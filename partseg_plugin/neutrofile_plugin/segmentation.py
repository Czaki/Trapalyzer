import operator
import typing
from copy import deepcopy
from typing import Callable

import numpy as np
import SimpleITK as sitk

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from PartSegCore.channel_class import Channel
from PartSegCore.segmentation import RestartableAlgorithm, SegmentationAlgorithm
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription, SegmentationResult
from PartSegCore.segmentation.threshold import BaseThreshold, ManualThreshold, threshold_dict

from .widgets import TrapezoidWidget

ALIVE_VAL = 1
DEAD_VAL = 2
BACTERIA_VAL = 3
OTHER_VAL = 4
NET_VAL = 5
LABELING_NAME = "labeling"


class TrapezoidNeutrofileSegmentation(SegmentationAlgorithm):
    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return False

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        pass

    def get_info_text(self):
        return ""

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.get_name(), dict(self.new_parameters))

    @classmethod
    def get_name(cls) -> str:
        return "Trapezoid Segment Neutrofile"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty(
                "alive_volume",
                "Alive volume",
                {"lower_bound": 10, "upper_bound": 500, "softness": 0.5},
                property_type=TrapezoidWidget,
            ),
            AlgorithmProperty(
                "alive_brightness",
                "Alive br",
                {"lower_bound": 10, "upper_bound": 500, "softness": 0.5},
                property_type=TrapezoidWidget,
            ),
        ]


class NeutrofileSegmentation(RestartableAlgorithm):
    def __init__(self):
        super().__init__()
        self.good, self.bad, self.nets, self.bacteria, self.net_area = 0, 0, 0, 0, 0

    @staticmethod
    def _range_threshold(data: np.ndarray, thr):
        thr1, thr2 = thr
        return (data > thr1) * (data < thr2)

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

    @staticmethod
    def _remove_net_components(channel, components, min_reach_thr):
        for val in np.unique(components):
            if val == 0:
                continue
            mask = components == val
            if np.max(channel[mask]) < min_reach_thr:
                components[mask] = 0
        return components

    def _classify_neutrofile(self, inner_dna_components, out_dna_mask):
        dna_channel = self.get_channel(self.new_parameters["dna_marker"])
        _mask, alive_thr_val = self._calculate_mask(dna_channel, "threshold2")
        result_labeling = np.zeros(inner_dna_components.shape, dtype=np.uint8)

        self.good, self.bad, self.nets, self.bacteria, self.net_area = 0, 0, 0, 0, 0
        sizes = np.bincount(inner_dna_components.flat)
        ratio = self.new_parameters["dead_ratio"]
        max_size = min(self.new_parameters["maximum_neutrofile_size"], self.new_parameters["net_size"])
        roi_annotation = {}
        for val in np.unique(inner_dna_components):
            if val == 0:
                continue
            mm = inner_dna_components == val
            count = np.sum(out_dna_mask[mm])

            mean_brightness = np.mean(dna_channel[mm])
            if sizes[val] < max_size:
                if count < ratio * sizes[val]:
                    if mean_brightness > alive_thr_val:
                        roi_annotation[val] = {
                            "type": "Alive neutrofile",
                            "Mean brightness": mean_brightness,
                            "Voxels": sizes[val],
                            "Dead marker ratio": count / sizes[val],
                        }
                        result_labeling[mm] = ALIVE_VAL
                        self.good += 1
                    else:
                        roi_annotation[val] = {
                            "type": "Bacteria group",
                            "Mean brightness": mean_brightness,
                            "Voxels": sizes[val],
                            "Dead marker ratio": count / sizes[val],
                        }
                        result_labeling[mm] = BACTERIA_VAL
                        self.bacteria += 1
                else:
                    roi_annotation[val] = {
                        "type": "Dead neutrofile",
                        "Voxels": sizes[val],
                        "Dead marker ratio": count / sizes[val],
                    }
                    result_labeling[mm] = DEAD_VAL
                    self.bad += 1
            elif count < ratio * sizes[val]:
                roi_annotation[val] = {
                    "type": "Bacteria group",
                    "Voxels": sizes[val],
                    "Dead marker ratio": count / sizes[val],
                }
                result_labeling[mm] = BACTERIA_VAL
                self.bacteria += 1
            else:
                roi_annotation[val] = {
                    "type": "Bacteria group",
                    "Voxels": sizes[val],
                    "Dead marker ratio": count / sizes[val],
                }
                result_labeling[mm] = BACTERIA_VAL
                self.bacteria += 1
                print("problem here", val, sizes[val])
        return result_labeling, roi_annotation

    def calculation_run(self, report_fun: Callable[[str, int], None] = None) -> SegmentationResult:
        for key in self.new_parameters.keys():
            self.parameters[key] = deepcopy(self.new_parameters[key])
        dna_channel = self.get_channel(self.new_parameters["dna_marker"])
        dead_dna_channel = self.get_channel(self.new_parameters["dead_dna_marker"])
        inner_dna_mask, thr_val = self._calculate_mask(dna_channel, "threshold")
        out_dna_mask, dead_thr_val = self._calculate_mask(dead_dna_channel, "dead_threshold")
        _mask, net_thr_val = self._calculate_mask(dead_dna_channel, "net_threshold")

        out_dna_components = self._calc_components(out_dna_mask, self.new_parameters["net_size"])
        out_dna_components = self._remove_net_components(dead_dna_channel, out_dna_components, net_thr_val)

        _inner_dna_mask = inner_dna_mask.copy()

        inner_dna_mask[out_dna_components > 0] = 0
        inner_dna_components = self._calc_components(inner_dna_mask, self.new_parameters["minimum_size"])

        idn = inner_dna_components.copy()
        odn = out_dna_components.copy()

        result_labeling, roi_annotation = self._classify_neutrofile(inner_dna_components, out_dna_mask)
        self.nets = np.max(out_dna_components)
        self.net_area = np.count_nonzero(out_dna_components)
        if self.new_parameters["separate_nets"]:
            result_labeling[out_dna_components > 0] = out_dna_components[out_dna_components > 0] + (NET_VAL - 1)
        else:
            result_labeling[out_dna_components > 0] = NET_VAL
        max_component = inner_dna_components.max()
        inner_dna_components[out_dna_components > 0] = out_dna_components[out_dna_components > 0] + max_component
        for val in np.unique(out_dna_components[out_dna_components > 0]):
            roi_annotation[val + max_component] = {"type": "Neutrofile net"}
        return SegmentationResult(
            inner_dna_components,
            self.get_segmentation_profile(),
            {
                "inner dna base": AdditionalLayerDescription(_inner_dna_mask, "labels", "inner dna base"),
                "inner dna": AdditionalLayerDescription(inner_dna_mask, "labels", "inner dna"),
                "inner dna com": AdditionalLayerDescription(idn, "labels", "inner dna com"),
                "outer dna": AdditionalLayerDescription(out_dna_mask, "labels", "out dna"),
                "outer dna com": AdditionalLayerDescription(odn, "labels", "outer dna com"),
            },
            alternative_representation={LABELING_NAME: result_labeling},
            roi_annotation=roi_annotation,
        )

    def get_info_text(self):
        return (
            f"Alive: {self.good}, dead: {self.bad}, nets: {self.nets}, bacteria groups: {self.bacteria}, "
            f"net size {self.net_area}"
        )

    @classmethod
    def get_name(cls) -> str:
        return "Segment Neutrofile"

    @staticmethod
    def single_channel():
        return False

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("dna_marker", "DNA marker", 1, property_type=Channel),
            AlgorithmProperty(
                "threshold",
                "Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "threshold2",
                "Mean alive value",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("dead_dna_marker", "Dead DNA marker", 0, property_type=Channel),
            AlgorithmProperty(
                "dead_threshold",
                "Dead Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty(
                "net_threshold",
                "Net reach Threshold",
                next(iter(threshold_dict.keys())),
                possible_values=threshold_dict,
                property_type=AlgorithmDescribeBase,
            ),
            AlgorithmProperty("minimum_size", "Minimum size (px)", 20, (0, 10 ** 6)),
            AlgorithmProperty(
                "maximum_neutrofile_size",
                "Maximum neutrofile size (px)",
                100,
                (0, 10 ** 6),
            ),
            AlgorithmProperty("net_size", "net size (px)", 500, (0, 10 ** 6), 100),
            AlgorithmProperty("separate_nets", "Mark nets separate", True),
            AlgorithmProperty("dead_ratio", "Dead marker ratio", 0.5, (0, 1)),
        ]

    @classmethod
    def get_default_values(cls):
        val = super().get_default_values()
        val["threshold"] = {
            "name": ManualThreshold.get_name(),
            "values": {"threshold": 120},
        }
        val["threshold2"] = {
            "name": ManualThreshold.get_name(),
            "values": {"threshold": 100},
        }
        val["dead_threshold"] = {
            "name": ManualThreshold.get_name(),
            "values": {"threshold": 20},
        }
        val["net_threshold"] = {
            "name": ManualThreshold.get_name(),
            "values": {"threshold": 34},
        }
        return val


class NeutrofileOnlySegmentation(RestartableAlgorithm):
    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        pass

    @classmethod
    def get_name(cls) -> str:
        return "Neutrofile only"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        pass
