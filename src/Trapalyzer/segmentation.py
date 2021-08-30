import operator
import typing
from abc import ABC
from collections import Counter
from enum import Enum
from itertools import chain, product
from typing import Callable

import numpy as np
import SimpleITK as sitk
from napari.layers import Image
from napari.types import LayerDataTuple

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from PartSegCore.analysis.measurement_calculation import Diameter, get_border
from PartSegCore.autofit import density_mass_center
from PartSegCore.channel_class import Channel
from PartSegCore.class_generator import enum_register
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation import RestartableAlgorithm
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription, SegmentationResult
from PartSegCore.segmentation.noise_filtering import noise_filtering_dict
from PartSegCore.segmentation.threshold import BaseThreshold, threshold_dict

from .widgets import TrapezoidWidget


class NeuType(Enum):
    PMN_neu = 1
    DEC_neu = 2
    NER_neu = 3
    PMP_neu = 4
    Bacteria = 5
    NET = 8
    Unknown_intra = 10
    Unknown_extra = 11

    def __str__(self):
        return self.name.replace("_", " ")

    def __lt__(self, other):
        return self.value < other.value

    @classmethod
    def known_components(cls):
        return (x for x in cls.__members__.values() if x.value < 10)

    @classmethod
    def neutrofile_components(cls):
        return (x for x in cls.__members__.values() if x.value < 8)

    @classmethod
    def all_components(cls):
        return cls.__members__.values()


try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    reloading
except NameError:
    reloading = False  # means the module is being imported
    enum_register.register_class(NeuType)

LABELING_NAME = "Labeling"
SCORE_SUFFIX = "_score"
CATEGORY_STR = "category"
PARAMETER_TYPE_LIST = [
    "pixel count",
    "brightness",
    "ext. brightness",
    "brightness gradient",
]  # "brightness", "roundness"
DESCRIPTION_DICT = {
    NeuType.PMN_neu: "polymorphonuclear neutrophils",
    NeuType.DEC_neu: "decondensed chromatin neutrophils",
    NeuType.NER_neu: "ruptured nuclear envelope neutrophils",
    NeuType.PMP_neu: "plasma membrane permeabilized neutrophils",
    NeuType.NET: "neutrophil extracellular trap",
    NeuType.Unknown_intra: "unclassified intracellurar component",
    NeuType.Unknown_extra: "unclassified extracellurar component",
    NeuType.Bacteria: "bacteria",
}


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
    def _calc_components(arr, min_size, close_holes=0):
        components = sitk.RelabelComponent(
            sitk.ConnectedComponent(sitk.GetImageFromArray(arr)),
            min_size,
        )

        if close_holes:
            binary1 = sitk.BinaryThreshold(components, 0, 0)
            removed_small = sitk.RelabelComponent(sitk.ConnectedComponent(binary1), close_holes)
            binary2 = sitk.BinaryThreshold(removed_small, 0, 0)
            components = sitk.RelabelComponent(
                sitk.ConnectedComponent(binary2),
                min_size,
            )
        return sitk.GetArrayFromImage(components)


class Trapalyzer(NeutrofileSegmentationBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.count_dict = Counter()
        self.nets = 0
        self.other = 0
        self.net_size = 0
        self.quality = 0

    def classify_nets(self, outer_dna_mask, net_size):
        nets = self._calc_components(outer_dna_mask, int(net_size["lower_bound"]))
        inner_dna_channel = self.get_channel(self.new_parameters["inner_dna"])
        outer_dna_channel = self.get_channel(self.new_parameters["outer_dna"])
        laplacian_image = _laplacian_estimate(inner_dna_channel, 1.3)
        laplacian_outer_image = _laplacian_estimate(outer_dna_channel, 1.3)
        annotation = {}
        i = 1
        nets[nets > 0] += 1
        for val in np.unique(nets):
            if val == 0:
                continue
            component = np.array(nets == val)
            brightness_gradient = np.mean(laplacian_outer_image[component])
            brightness_gradient_score = sine_score_function(
                brightness_gradient,
                softness=self.new_parameters["softness"],
                **self.new_parameters["net_brightness_gradient"],
            )
            brightness = np.quantile(inner_dna_channel[component], 0.9)
            ext_brightness = np.quantile(outer_dna_channel[component], 0.9)
            ext_brightness_score = sine_score_function(
                ext_brightness, softness=self.new_parameters["softness"], **self.new_parameters["net_ext_brightness"]
            )
            voxels = np.count_nonzero(component)
            voxels_score = sine_score_function(
                voxels, softness=self.new_parameters["softness"], **self.new_parameters["net_size"]
            )

            if voxels_score * ext_brightness_score * brightness_gradient_score < self.new_parameters["minimum_score"]:
                if not self.new_parameters["unknown_net"]:
                    nets[component] = 0
                    continue
                else:
                    category = NeuType.Unknown_extra
            else:
                category = NeuType.NET

            data_dict = {
                "component_id": i,
                CATEGORY_STR: category,
                "pixel count": voxels,
                "brightness": brightness,
                "ext. brightness": ext_brightness,
                "brightness gradient": np.mean(laplacian_image[component]),
                "ext. brightness gradient": brightness_gradient,
                "ext. brightness std": np.std(outer_dna_channel[component]),
            }
            annotation[i] = data_dict
            nets[component] = i
            i += 1
        return nets, annotation

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        self.count_dict = Counter()
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
        outer_dna_components, net_annotation = self.classify_nets(outer_dna_mask, self.new_parameters["net_size"])
        inner_dna_mask[outer_dna_components > 0] = 0
        size_param_array = [
            self.new_parameters[x.name.lower() + "_pixel count"] for x in NeuType.neutrofile_components()
        ]
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
        inner_dna_components = self._calc_components(inner_dna_mask, min_object_size, close_holes=100)

        result_labeling, roi_annotation = self._classify_neutrofile(inner_dna_components, cleaned_inner, cleaned_outer)

        nets_components = [k for k, v in net_annotation.items() if v[CATEGORY_STR] == NeuType.NET]
        unknown_net_components = [k for k, v in net_annotation.items() if v[CATEGORY_STR] != NeuType.NET]

        self.count_dict[NeuType.NET] = len(nets_components)
        self.count_dict[NeuType.Unknown_extra] = len(unknown_net_components)

        result_labeling[np.isin(outer_dna_components, nets_components)] = NeuType.NET.value
        result_labeling[np.isin(outer_dna_components, unknown_net_components)] = NeuType.Unknown_extra.value
        max_component = inner_dna_components.max()
        inner_dna_components[outer_dna_components > 0] = outer_dna_components[outer_dna_components > 0] + max_component
        for value in np.unique(inner_dna_components[outer_dna_components > 0]):
            roi_annotation[value] = net_annotation[value - max_component]
            roi_annotation[value]["component_id"] = value
        alternative_representation = {LABELING_NAME: result_labeling}
        for val in NeuType.all_components():
            alternative_representation[str(val)] = (result_labeling == val.value).astype(np.uint8) * val.value
        self.net_size = np.count_nonzero(alternative_representation[str(NeuType.NET)])
        from .measurement import QualityMeasure

        self.quality = QualityMeasure.calculate_property(inner_dna_components, roi_annotation)
        return SegmentationResult(
            inner_dna_components,
            self.get_segmentation_profile(),
            alternative_representation=alternative_representation,
            roi_annotation=roi_annotation,
            additional_layers={
                "inner_mask": AdditionalLayerDescription(inner_dna_mask.astype(np.uint8), "labels", "inner_mask")
            },
        )

    def _classify_neutrofile(self, inner_dna_components, cleaned_inner, cleaned_outer):
        laplacian_image = _laplacian_estimate(cleaned_inner, 1.3)
        annotation = {}
        labeling = np.zeros(inner_dna_components.shape, dtype=np.uint16)
        bounds = ROIInfo.calc_bounds(inner_dna_components)
        inner_dna_components_border = get_border(inner_dna_components)

        for val in np.unique(inner_dna_components):
            if val == 0:
                continue
            slices = tuple(bounds[val].get_slices(margin=5))
            component = np.array(inner_dna_components[slices] == val)
            component_border = inner_dna_components_border[slices] == val
            voxels = np.count_nonzero(component)
            perimeter = np.count_nonzero(component_border)
            diameter = Diameter.calculate_property(component, voxel_size=(1,) * component.ndim, result_scalar=1)

            if perimeter == 0:
                continue
            data_dict = {
                "pixel count": voxels,
                "brightness gradient": np.mean(laplacian_image[slices][component]),
                "brightness": np.quantile(cleaned_inner[slices][component], 0.9),
                "intensity": np.sum(cleaned_inner[slices][component]),
                # "homogenity": np.mean(inner_dna_channel[component]) / np.std(inner_dna_channel[component]),
                "ext. brightness": np.quantile(cleaned_outer[slices][component], 0.9),
                #                  "roundness": new_sphericity(component, self.image.voxel_size),
                #                 "roundness2": voxels / ((diameter ** 2 / 4) * pi),
                #                 "diameter": diameter,
                # "curvature": np.std(curvature(component_border_coords[0], component_border_coords[1])),
                #                 "border_size": component_border_coords[0].size,
                # "circumference": len(component_border_coords[0]),
                # "area to circumference": voxels / len(component_border_coords[0]),
                # "signal colocalization": colocalization1,
                "circularity": 3 * np.pi * voxels / perimeter ** 2,  # inspired by NETQUANT approach
                "circularity2": voxels / ((diameter ** 2 / 4) * np.pi),
            }
            annotation[val] = dict(
                {"component_id": val, CATEGORY_STR: NeuType.Unknown_intra},
                **data_dict,
                **{
                    f"{prefix} {suffix}": sine_score_function(
                        data_dict[suffix],
                        softness=self.new_parameters["softness"],
                        **self.new_parameters[f"{prefix.name.lower()}_{suffix}"],
                    )
                    for prefix, suffix in product(NeuType.neutrofile_components(), PARAMETER_TYPE_LIST)
                },
            )
            score_list = []
            for component_name in NeuType.neutrofile_components():
                score = 1
                for parameter in PARAMETER_TYPE_LIST:
                    score *= annotation[val][f"{component_name} {parameter}"]
                score_list.append((score, component_name))
                annotation[val][str(component_name) + SCORE_SUFFIX] = score

            score_list = sorted(score_list)
            if (
                score_list[-1][0] < self.new_parameters["minimum_score"]
                or score_list[-2][0] > self.new_parameters["maximum_other"]
            ):
                labeling[inner_dna_components == val] = NeuType.Unknown_intra.value
                self.count_dict[NeuType.Unknown_intra] += 1
            else:
                labeling[inner_dna_components == val] = score_list[-1][1].value
                self.count_dict[score_list[-1][1]] += 1
                annotation[val][CATEGORY_STR] = score_list[-1][1]

        return labeling, annotation

    def get_info_text(self):
        return (
            ", ".join(f"{name}: {self.count_dict[name]}" for name in NeuType.all_components())
            + f"\nQuality: {self.quality}"
        )

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), dict(self.new_parameters))

    @classmethod
    def get_name(cls) -> str:
        return "Trapalyzer"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        initial = list(
            chain(
                *(
                    [
                        AlgorithmProperty(
                            f"{prefix.name.lower()}_{suffix}",
                            f"{prefix} {suffix}",
                            {"lower_bound": 0, "upper_bound": 1},
                            property_type=TrapezoidWidget,
                            help_text=f"{DESCRIPTION_DICT[prefix]} {suffix}",
                        )
                        for suffix in PARAMETER_TYPE_LIST
                    ]
                    + ["-------------------"]
                    for prefix in NeuType.neutrofile_components()
                )
            )
        ) + [
            AlgorithmProperty("minimum_score", "Minimum score", 0.8),
            AlgorithmProperty("maximum_other", "Maximum competing score", 0.4),
            AlgorithmProperty(
                "minimum_size", "Minimum comp. pixel count", 40, (1, 9999), help_text="Minimum component pixel count"
            ),
            AlgorithmProperty("softness", "Error margin coeficient", 0.1, (0, 1)),
        ]

        thresholds = [
            AlgorithmProperty("inner_dna", "All DNA channel", 1, property_type=Channel),
            AlgorithmProperty(
                "inner_dna_noise_filtering",
                "Filter type",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "inner_threshold",
                "Threshold type",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("outer_dna", "Extracellural DNA channel", 1, property_type=Channel),
            AlgorithmProperty(
                "outer_dna_noise_filtering",
                "Filter type",
                next(iter(noise_filtering_dict.keys())),
                possible_values=noise_filtering_dict,
                value_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "outer_threshold",
                "Threshold type",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "net_size",
                "NET pixel count",
                {"lower_bound": 850, "upper_bound": 999999},
                property_type=TrapezoidWidget,
            ),
            AlgorithmProperty(
                "net_ext_brightness",
                "NET extracellular brightness",
                {"lower_bound": 21, "upper_bound": 100},
                property_type=TrapezoidWidget,
            ),
            AlgorithmProperty(
                "net_brightness_gradient",
                "NET brightness gradient",
                {"lower_bound": 0.0, "upper_bound": 1.0},
                property_type=TrapezoidWidget,
            ),
            AlgorithmProperty(
                "unknown_net", "detect extracellular artifacts", True, help_text="If mark unknown net components"
            ),
            "-----------------------",
        ]

        return thresholds + initial


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
    return LayerDataTuple((res, {"colormap": "magma", "scale": image.scale, "name": "Laplacian estimate"}))


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
