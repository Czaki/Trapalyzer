import operator
import typing
from abc import ABC
from itertools import chain, product
from math import pi
from typing import Callable

import numpy as np
import SimpleITK as sitk
from napari.layers import Image
from napari.types import LayerDataTuple

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from PartSegCore.analysis.measurement_calculation import Diameter, get_border
from PartSegCore.autofit import density_mass_center
from PartSegCore.channel_class import Channel
from PartSegCore.segmentation import RestartableAlgorithm
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription, SegmentationResult
from PartSegCore.segmentation.noise_filtering import noise_filtering_dict
from PartSegCore.segmentation.threshold import BaseThreshold, threshold_dict

from .widgets import TrapezoidWidget

ALIVE_VAL = 1
DECONDENSED_VAL = 1
DEAD_VAL = 3
BACTERIA_VAL = 4
OTHER_VAL = 5
NET_VAL = 6
LABELING_NAME = "Labeling"
SCORE_SUFFIX = "_score"
COMPONENT_DICT = {"Alive": ALIVE_VAL, "Decondensed": DECONDENSED_VAL, "Dead": DEAD_VAL, "Bacteria": BACTERIA_VAL}
COMPONENT_SCORE_LIST = list(COMPONENT_DICT.keys())
PARAMETER_TYPE_LIST = ["voxels", "ext. brightness", "sharpness"]  # "brightness", "roundness"


class NeutrofileSegmentationBase(RestartableAlgorithm, ABC):
    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return False

    def _calculate_mask(self, channel, threshold_name):
        thr: BaseThreshold = threshold_dict[self.new_parameters[threshold_name]["name"]]
        mask, thr_val = thr.calculate_mask(
            channel, self.mask, self.new_parameters[threshold_name]["values"], operator.gt
        )
        return mask, thr_val

    @staticmethod
    def _calc_components(arr, min_size):
        return sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(arr)),
                min_size,
            )
        )


class TrapezoidNeutrofileSegmentation(NeutrofileSegmentationBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.count_dict = {ALIVE_VAL: 0, DEAD_VAL: 0, BACTERIA_VAL: 0, DECONDENSED_VAL: 0}
        self.nets = 0
        self.other = 0
        self.net_size = 0

    def classify_nets(self, outer_dna_mask, net_size, net_sharpness, net_ext_brightness):
        nets = self._calc_components(outer_dna_mask, net_size)
        inner_dna_channel = self.get_channel(self.new_parameters["inner_dna"])
        outer_dna_channel = self.get_channel(self.new_parameters["outer_dna"])
        nets_border = get_border(nets)
        laplacian_image = _laplacian_estimate(inner_dna_channel, 1.3)
        laplacian_outer_image = _laplacian_estimate(outer_dna_channel, 1.3)
        annotation = {}
        i = 1
        for val in np.unique(nets):
            if val == 0:
                continue
            component = np.array(nets == val)
            sharpness = np.mean(laplacian_image[component])
            brightness = np.mean(inner_dna_channel[component])
            ext_brightness = np.mean(outer_dna_channel[component])
            if sharpness > net_sharpness or ext_brightness < net_ext_brightness:
                nets[component] = 0
                continue
            component_border = nets_border == val
            component_border_coords = np.nonzero(component_border)
            diameter = Diameter.calculate_property(component, voxel_size=(1,) * component.ndim, result_scalar=1)
            voxels = np.count_nonzero(component)
            data_dict = {
                "component_id": i,
                "category": "Net",
                "voxels": voxels,
                "sharpness": sharpness,
                "sharpness outer": np.mean(laplacian_outer_image[component]),
                "brightness": brightness,
                "homogenity": np.mean(inner_dna_channel[component]) / np.std(inner_dna_channel[component]),
                "ext. brightness": ext_brightness,
                "roundness": new_sphericity(component, self.image.voxel_size),
                "roundness2": voxels / ((diameter ** 2 / 4) * pi),
                "diameter": diameter,
                "border_size": component_border_coords[0].size,
            }
            annotation[i] = data_dict
            nets[component] = i
            i += 1
        return nets, annotation

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        self.count_dict = {ALIVE_VAL: 0, DEAD_VAL: 0, BACTERIA_VAL: 0, DECONDENSED_VAL: 0}
        self.nets = 0
        self.other = 0
        self.net_size = 0
        inner_dna_channel = self.get_channel(self.new_parameters["inner_dna"])
        inner_noise_filtering_parameters = self.new_parameters["inner_dna_noise_filtering"]
        cleaned_inner = noise_filtering_dict[inner_noise_filtering_parameters["name"]].noise_filter(
            inner_dna_channel, self.image.spacing, inner_noise_filtering_parameters["values"]
        )
        inner_dna_mask, thr_val = self._calculate_mask(cleaned_inner, "inner_threshold")

        outer_dna_channel = self.get_channel(self.new_parameters["outer_dna"])
        outer_noise_filtering_parameters = self.new_parameters["outer_dna_noise_filtering"]
        cleaned_outer = noise_filtering_dict[outer_noise_filtering_parameters["name"]].noise_filter(
            outer_dna_channel, self.image.spacing, outer_noise_filtering_parameters["values"]
        )
        outer_dna_mask, dead_thr_val = self._calculate_mask(cleaned_outer, "outer_threshold")
        outer_dna_components, net_annotation = self.classify_nets(
            outer_dna_mask,
            self.new_parameters["net_size"],
            self.new_parameters["net_sharpness"],
            self.new_parameters["net_ext_brightness"],
        )
        inner_dna_mask[outer_dna_components > 0] = 0
        size_param_array = [self.new_parameters[x.lower() + "_voxels"] for x in COMPONENT_SCORE_LIST]
        min_object_size = int(
            max(
                5,
                min(
                    x["lower_bound"]
                    - (x["upper_bound"] - x["lower_bound"])
                    * self.new_parameters["softness"]
                    * x["lower_bound"]
                    / (x["upper_bound"] - x["lower_bound"])
                    for x in size_param_array
                ),
                self.new_parameters["minimum_size"],
            )
        )
        inner_dna_components = self._calc_components(inner_dna_mask, min_object_size)

        result_labeling, roi_annotation = self._classify_neutrofile(inner_dna_components, outer_dna_mask)

        self.nets = len(np.unique(outer_dna_components[outer_dna_components > 0]))

        result_labeling[outer_dna_components > 0] = NET_VAL
        max_component = inner_dna_components.max()
        inner_dna_components[outer_dna_components > 0] = outer_dna_components[outer_dna_components > 0] + max_component
        for value in np.unique(inner_dna_components[outer_dna_components > 0]):
            roi_annotation[value] = net_annotation[value - max_component]
            roi_annotation[value]["component_id"] = value
        alternative_representation = {LABELING_NAME: result_labeling}
        for name, val in COMPONENT_DICT.items():
            alternative_representation[name] = (result_labeling == val).astype(np.uint8) * val
        alternative_representation["Nets"] = (result_labeling == NET_VAL).astype(np.uint8)
        alternative_representation["Others"] = (result_labeling == OTHER_VAL).astype(np.uint8) * OTHER_VAL
        self.net_size = np.count_nonzero(alternative_representation["Nets"])
        return SegmentationResult(
            inner_dna_components,
            self.get_segmentation_profile(),
            alternative_representation=alternative_representation,
            roi_annotation=roi_annotation,
            additional_layers={
                "inner_mask": AdditionalLayerDescription(inner_dna_mask.astype(np.uint8), "labels", "inner_mask")
            },
        )

    def _classify_neutrofile(self, inner_dna_components, out_dna_mask):
        inner_dna_channel = self.get_channel(self.new_parameters["inner_dna"])
        outer_dna_channel = self.get_channel(self.new_parameters["outer_dna"])
        laplacian_image = _laplacian_estimate(inner_dna_channel, 1.3)
        annotation = {}
        labeling = np.zeros(inner_dna_components.shape, dtype=np.uint16)
        inner_dna_components_border = get_border(inner_dna_components)
        for val in np.unique(inner_dna_components):
            if val == 0:
                continue
            component = np.array(inner_dna_components == val)
            component_border = inner_dna_components_border == val
            component_border_coords = np.nonzero(component_border)
            # need to assert there are no holes - or get separate borders
            diameter = Diameter.calculate_property(component, voxel_size=(1,) * component.ndim, result_scalar=1)
            voxels = np.count_nonzero(component)
            colocalization1 = np.mean((inner_dna_channel[component] - 22) * (outer_dna_channel[component] - 80))

            if voxels == 0 or diameter == 0:
                continue
            data_dict = {
                "voxels": voxels,
                "sharpness": np.mean(laplacian_image[component]),
                "brightness": np.mean(inner_dna_channel[component]),
                "homogenity": np.mean(inner_dna_channel[component]) / np.std(inner_dna_channel[component]),
                "ext. brightness": np.mean(outer_dna_channel[component]),
                #                 "roundness": new_sphericity(component, self.image.voxel_size),
                #                 "roundness2": voxels / ((diameter ** 2 / 4) * pi),
                #                 "diameter": diameter,
                "curvature": np.std(curvature(component_border_coords[0], component_border_coords[1])),
                #                 "border_size": component_border_coords[0].size,
                "circumference": len(component_border_coords[0]),
                "area to circumference": voxels / len(component_border_coords[0]),
                "colocalization1": colocalization1,
            }
            annotation[val] = dict(
                {"component_id": val, "category": "Unknown"},
                **data_dict,
                **{
                    f"{prefix} {suffix}": sine_score_function(
                        data_dict[suffix],
                        softness=self.new_parameters["softness"],
                        **self.new_parameters[f"{prefix.lower()}_{suffix}"],
                    )
                    for prefix, suffix in product(COMPONENT_SCORE_LIST, PARAMETER_TYPE_LIST)
                },
            )
            score_list = []
            for component_name in COMPONENT_SCORE_LIST:
                score = 1
                for parameter in PARAMETER_TYPE_LIST:
                    score *= annotation[val][f"{component_name} {parameter}"]
                score_list.append((score, component_name))
            for el in score_list:
                annotation[val][el[1] + SCORE_SUFFIX] = el[0]
            score_list = sorted(score_list)
            if (
                score_list[-1][0] < self.new_parameters["minimum_score"]
                or score_list[-2][0] > self.new_parameters["maximum_other"]
            ):
                labeling[component] = OTHER_VAL
                self.other += 1
            else:
                labeling[component] = COMPONENT_DICT[score_list[-1][1]]
                self.count_dict[COMPONENT_DICT[score_list[-1][1]]] += 1
                annotation[val]["category"] = score_list[-1][1]

        return labeling, annotation

    def get_info_text(self):
        return (
            f"Alive: {self.count_dict[ALIVE_VAL]}, Decondensed: {self.count_dict[DECONDENSED_VAL]}, Dead: {self.count_dict[DEAD_VAL]}, Bacteria: {self.count_dict[BACTERIA_VAL]}, "
            f"Nets: {self.nets}, Nets voxels: {self.net_size} Other: {self.other}"
        )

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), dict(self.new_parameters))

    @classmethod
    def get_name(cls) -> str:
        return "Trapezoid Segment Neutrofile"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        initial = list(
            chain(
                *[
                    [
                        AlgorithmProperty(
                            f"{prefix.lower()}_{suffix}",
                            f"{prefix} {suffix}",
                            {"lower_bound": 10, "upper_bound": 50},
                            property_type=TrapezoidWidget,
                        )
                        for suffix in PARAMETER_TYPE_LIST
                    ]
                    + ["-------------------"]
                    for prefix in COMPONENT_SCORE_LIST
                ]
            )
        ) + [
            AlgorithmProperty("minimum_score", "Minimum score", 0.8),
            AlgorithmProperty("maximum_other", "Maximum other score", 0.4),
            AlgorithmProperty("minimum_size", "Min component size", 40, (1, 9999)),
            AlgorithmProperty("softness", "Softness", 0.1, (0, 1)),
        ]

        thresholds = [
            AlgorithmProperty("inner_dna", "Inner DNA", 1, property_type=Channel),
            AlgorithmProperty(
                "inner_dna_noise_filtering",
                "Filter inner",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "inner_threshold",
                "Inner threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("outer_dna", "Outer DNA", 1, property_type=Channel),
            AlgorithmProperty(
                "outer_dna_noise_filtering",
                "Filter outer",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "outer_threshold",
                "Outer Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("net_size", "net voxels", 500, (0, 10 ** 6), 100),
            AlgorithmProperty("net_sharpness", "net sharpness", 1.0, (0, 10 ** 2), 1),
            AlgorithmProperty("net_ext_brightness", "net_ext_brightness", 21.0, (0, 10 ** 3), 1),
            "-----------------------",
        ]

        return thresholds + initial


# class NeutrofileSegmentation(NeutrofileSegmentationBase):
#     def __init__(self):
#         super().__init__()
#         self.good, self.bad, self.nets, self.bacteria, self.net_area = 0, 0, 0, 0, 0
#
#     @staticmethod
#     def _range_threshold(data: np.ndarray, thr):
#         thr1, thr2 = thr
#         return (data > thr1) * (data < thr2)
#
#     def _calculate_mask(self, channel, threshold_name):
#         thr: BaseThreshold = threshold_dict[self.new_parameters[threshold_name]["name"]]
#         mask, thr_val = thr.calculate_mask(
#             channel, self.mask, self.new_parameters[threshold_name]["values"], operator.gt
#         )
#         return mask, thr_val
#
#     @staticmethod
#     def _calc_components(arr, min_size):
#         return sitk.GetArrayFromImage(
#             sitk.RelabelComponent(
#                 sitk.ConnectedComponent(sitk.GetImageFromArray(arr)),
#                 min_size,
#             )
#         )
#
#     @staticmethod
#     def _remove_net_components(channel, components, min_reach_thr):
#         for val in np.unique(components):
#             if val == 0:
#                 continue
#             mask = components == val
#             if np.max(channel[mask]) < min_reach_thr:
#                 components[mask] = 0
#         return components
#
#     def _classify_neutrofile(self, inner_dna_components, out_dna_mask):
#         dna_channel = self.get_channel(self.new_parameters["dna_marker"])
#         _mask, alive_thr_val = self._calculate_mask(dna_channel, "threshold2")
#         result_labeling = np.zeros(inner_dna_components.shape, dtype=np.uint8)
#
#         self.good, self.bad, self.nets, self.bacteria, self.net_area = 0, 0, 0, 0, 0
#         sizes = np.bincount(inner_dna_components.flat)
#         ratio = self.new_parameters["dead_ratio"]
#         max_size = min(self.new_parameters["maximum_neutrofile_size"], self.new_parameters["net_size"])
#         roi_annotation = {}
#         for val in np.unique(inner_dna_components):
#             if val == 0:
#                 continue
#             mm = inner_dna_components == val
#             count = np.sum(out_dna_mask[mm])
#
#             mean_brightness = np.mean(dna_channel[mm])
#             if sizes[val] < max_size:
#                 if count < ratio * sizes[val]:
#                     if mean_brightness > alive_thr_val:
#                         roi_annotation[val] = {
#                             "type": "Alive neutrofile",
#                             "Mean brightness": mean_brightness,
#                             "Voxels": sizes[val],
#                             "Dead marker ratio": count / sizes[val],
#                         }
#                         result_labeling[mm] = ALIVE_VAL
#                         self.good += 1
#                     else:
#                         roi_annotation[val] = {
#                             "type": "Bacteria group",
#                             "Mean brightness": mean_brightness,
#                             "Voxels": sizes[val],
#                             "Dead marker ratio": count / sizes[val],
#                         }
#                         result_labeling[mm] = BACTERIA_VAL
#                         self.bacteria += 1
#                 else:
#                     roi_annotation[val] = {
#                         "type": "Dead neutrofile",
#                         "Voxels": sizes[val],
#                         "Dead marker ratio": count / sizes[val],
#                     }
#                     result_labeling[mm] = DEAD_VAL
#                     self.bad += 1
#             elif count < ratio * sizes[val]:
#                 roi_annotation[val] = {
#                     "type": "Bacteria group",
#                     "Voxels": sizes[val],
#                     "Dead marker ratio": count / sizes[val],
#                 }
#                 result_labeling[mm] = BACTERIA_VAL
#                 self.bacteria += 1
#             else:
#                 roi_annotation[val] = {
#                     "type": "Bacteria group",
#                     "Voxels": sizes[val],
#                     "Dead marker ratio": count / sizes[val],
#                 }
#                 result_labeling[mm] = BACTERIA_VAL
#                 self.bacteria += 1
#                 print("problem here", val, sizes[val])
#         return result_labeling, roi_annotation
#
#     def calculation_run(self, report_fun: Callable[[str, int], None] = None) -> SegmentationResult:
#         for key in self.new_parameters.keys():
#             self.parameters[key] = deepcopy(self.new_parameters[key])
#         dna_channel = self.get_channel(self.new_parameters["dna_marker"])
#         dead_dna_channel = self.get_channel(self.new_parameters["dead_dna_marker"])
#         inner_dna_mask, thr_val = self._calculate_mask(dna_channel, "threshold")
#         out_dna_mask, dead_thr_val = self._calculate_mask(dead_dna_channel, "dead_threshold")
#         _mask, net_thr_val = self._calculate_mask(dead_dna_channel, "net_threshold")
#
#         out_dna_components = self._calc_components(out_dna_mask, self.new_parameters["net_size"])
#         out_dna_components = self._remove_net_components(dead_dna_channel, out_dna_components, net_thr_val)
#
#         _inner_dna_mask = inner_dna_mask.copy()
#
#         inner_dna_mask[out_dna_components > 0] = 0
#         inner_dna_components = self._calc_components(inner_dna_mask, self.new_parameters["minimum_size"])
#
#         idn = inner_dna_components.copy()
#         odn = out_dna_components.copy()
#
#         result_labeling, roi_annotation = self._classify_neutrofile(inner_dna_components, out_dna_mask)
#         self.nets = np.max(out_dna_components)
#         self.net_area = np.count_nonzero(out_dna_components)
#         if self.new_parameters["separate_nets"]:
#             result_labeling[out_dna_components > 0] = out_dna_components[out_dna_components > 0] + (NET_VAL - 1)
#         else:
#             result_labeling[out_dna_components > 0] = NET_VAL
#         max_component = inner_dna_components.max()
#         inner_dna_components[out_dna_components > 0] = out_dna_components[out_dna_components > 0] + max_component
#         for val in np.unique(out_dna_components[out_dna_components > 0]):
#             roi_annotation[val + max_component] = {"type": "Neutrofile net"}
#         return SegmentationResult(
#             inner_dna_components,
#             self.get_segmentation_profile(),
#             {
#                 "inner dna base": AdditionalLayerDescription(_inner_dna_mask, "labels", "inner dna base"),
#                 "inner dna": AdditionalLayerDescription(inner_dna_mask, "labels", "inner dna"),
#                 "inner dna com": AdditionalLayerDescription(idn, "labels", "inner dna com"),
#                 "outer dna": AdditionalLayerDescription(out_dna_mask, "labels", "out dna"),
#                 "outer dna com": AdditionalLayerDescription(odn, "labels", "outer dna com"),
#             },
#             alternative_representation={LABELING_NAME: result_labeling},
#             roi_annotation=roi_annotation,
#         )
#
#     def get_info_text(self):
#         return (
#             f"Alive: {self.good}, dead: {self.bad}, nets: {self.nets}, bacteria groups: {self.bacteria}, "
#             f"net size {self.net_area}"
#         )
#
#     @classmethod
#     def get_name(cls) -> str:
#         return "Segment Neutrofile"
#
#     @staticmethod
#     def single_channel():
#         return False
#
#     @classmethod
#     def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
#         return [
#             AlgorithmProperty("dna_marker", "DNA marker", 1, property_type=Channel),
#             AlgorithmProperty(
#                 "threshold",
#                 "Threshold",
#                 next(iter(threshold_dict.keys())),
#                 possible_values=threshold_dict,
#                 property_type=AlgorithmDescribeBase,
#             ),
#             AlgorithmProperty(
#                 "threshold2",
#                 "Mean alive value",
#                 next(iter(threshold_dict.keys())),
#                 possible_values=threshold_dict,
#                 property_type=AlgorithmDescribeBase,
#             ),
#             AlgorithmProperty("dead_dna_marker", "Dead DNA marker", 0, property_type=Channel),
#             AlgorithmProperty(
#                 "dead_threshold",
#                 "Dead Threshold",
#                 next(iter(threshold_dict.keys())),
#                 possible_values=threshold_dict,
#                 property_type=AlgorithmDescribeBase,
#             ),
#             AlgorithmProperty(
#                 "net_threshold",
#                 "Net reach Threshold",
#                 next(iter(threshold_dict.keys())),
#                 possible_values=threshold_dict,
#                 property_type=AlgorithmDescribeBase,
#             ),
#             AlgorithmProperty("minimum_size", "Minimum size (px)", 20, (0, 10 ** 6)),
#             AlgorithmProperty(
#                 "maximum_neutrofile_size",
#                 "Maximum neutrofile size (px)",
#                 100,
#                 (0, 10 ** 6),
#             ),
#             AlgorithmProperty("net_size", "net size (px)", 500, (0, 10 ** 6), 100),
#             AlgorithmProperty("separate_nets", "Mark nets separate", True),
#             AlgorithmProperty("dead_ratio", "Dead marker ratio", 0.5, (0, 1)),
#         ]
#
#     @classmethod
#     def get_default_values(cls):
#         val = super().get_default_values()
#         val["threshold"] = {
#             "name": ManualThreshold.get_name(),
#             "values": {"threshold": 120},
#         }
#         val["threshold2"] = {
#             "name": ManualThreshold.get_name(),
#             "values": {"threshold": 100},
#         }
#         val["dead_threshold"] = {
#             "name": ManualThreshold.get_name(),
#             "values": {"threshold": 20},
#         }
#         val["net_threshold"] = {
#             "name": ManualThreshold.get_name(),
#             "values": {"threshold": 34},
#         }
#         return val


def trapezoid_score_function(x, lower_bound, upper_bound, softness=0.5):
    """
    Compute a score on a scale from 0 to 1 that indicate whether values from x belong
    to the interval (lbound, ubound) with a softened boundary.
    If a point lies inside the interval, its score is equal to 1.
    If the point is further away than the interval length multiplied by the softness parameter,
    its score is equal to zero.
    Otherwise the score is given by a linear function.
    """
    interval_width = upper_bound - lower_bound
    subound = upper_bound + softness * interval_width
    slbound = lower_bound - (softness * interval_width) * 0.1
    swidth = softness * interval_width  # width of the soft boundary
    if lower_bound <= x <= upper_bound:
        return 1.0
    elif x <= slbound or x >= subound:
        return 0.0
    elif slbound <= x <= lower_bound:
        return 1.0 - (lower_bound - x) / swidth
    elif upper_bound <= x <= subound:
        return 1.0 - (x - upper_bound) / swidth


def gaussian_score_function(x, lbound, ubound, softness=0.5):
    """
    Computes a score on a scale from 0 to 1 whether x lies within the [lbound, ubound] interval.
    Fuzzified by a gaussian function.
    Softness controls the extension of the interval.
    """
    assert ubound >= lbound, "ubound needs to be >= than lbound"
    interval_mean = ubound + lbound
    interval_length = ubound - lbound
    if lbound <= x <= ubound:
        return 1.0
    sd_l = np.log(softness * interval_length * lbound / interval_mean) - np.log(3)  # to avoid raising to small powers
    sd_l = np.exp(sd_l)
    sd_u = np.log(softness * interval_length * ubound / interval_mean) - np.log(3)
    sd_u = np.exp(sd_u)
    if x <= lbound - 3.5 * sd_l or x >= ubound + 3.5 * sd_u:
        return 0.0
    if lbound - 3.5 * sd_l <= x <= lbound:
        logscore = -0.5 * (lbound - x) ** 2 / sd_l ** 2
        return np.exp(logscore)
    if ubound <= x <= ubound + 3.5 * sd_u:
        logscore = -0.5 * (x - ubound) ** 2 / sd_u ** 2
        return np.exp(logscore)


def sine_score_function(x, lower_bound, upper_bound, softness=0.5):
    """
    Computes a score on a scale from 0 to 1 whether x lies within the [lbound, ubound] interval.
    Extended with a sine function.
    Softness controls the extension of the interval.
    """
    assert upper_bound >= lower_bound, "upper_bound needs to be >= than lower_bound"
    interval_mean = upper_bound + lower_bound
    interval_length = upper_bound - lower_bound
    if lower_bound <= x <= upper_bound:
        return 1.0
    extension_span_left = softness * interval_length * lower_bound / interval_mean
    extension_span_right = softness * interval_length * upper_bound / interval_mean
    if x <= lower_bound - extension_span_left or x >= upper_bound + extension_span_right:
        return 0.0
    if lower_bound - extension_span_left <= x <= lower_bound:
        coord_transform = 2 * (x - lower_bound + extension_span_left) / extension_span_left - 1
    elif upper_bound <= x <= upper_bound + extension_span_right:
        coord_transform = 2 * (x - upper_bound + extension_span_right) / extension_span_right - 1
    else:
        raise RuntimeError("Something went terribly wrong!")
    return 0.5 + 0.5 * np.sin(0.5 * np.pi * coord_transform)


def new_sphericity(component: np.ndarray, voxel_size):
    component = component.squeeze()
    voxel_area = np.prod(voxel_size)
    center = np.array([density_mass_center(component, voxel_size)])
    area = np.count_nonzero(component) * voxel_area
    coordinates = np.transpose(np.nonzero(component)) * voxel_size - center
    radius_square = area / np.pi
    return np.sum(np.sum(coordinates ** 2, axis=1) <= radius_square) / (
        area / voxel_area + np.sum(np.sum(coordinates ** 2, axis=1) > radius_square)
    )


def laplacian_estimate(image: Image, radius=1.30, clip_bellow_0=True) -> LayerDataTuple:
    res = _laplacian_estimate(image.data[0], radius=radius)
    if clip_bellow_0:
        res[res < 0] = 0
    res = res.reshape(image.data.shape)
    return res, {"colormap": "magma", "scale": image.scale, "name": "Laplacian estimate"}


def _laplacian_estimate(channel: np.ndarray, radius=1.30) -> np.ndarray:
    data = channel.astype(np.float64).squeeze()
    return -sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(sitk.GetImageFromArray(data), radius)).reshape(
        channel.shape
    )


def curvature(x, y, *args):
    assert len(x) == len(y)
    n = len(x)
    k = np.zeros(n)
    for i in range(n):
        ddx = x[(i + 1) % n] + x[(i - 1) % n] - 2 * x[i]
        ddx *= n ** 2
        ddy = y[(i + 1) % n] + y[(i - 1) % n] - 2 * y[i]
        ddy *= n ** 2
        k[i] = np.sqrt(ddx ** 2 + ddy ** 2)
    return k
