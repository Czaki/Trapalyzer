import operator
import typing
from copy import deepcopy
from typing import Callable
import SimpleITK as sitk
import numpy as np

from PartSegCore.algorithm_describe_base import AlgorithmProperty, AlgorithmDescribeBase
from PartSegCore.analysis.measurement_calculation import (
    MeasurementResult,
    MeasurementProfile,
)
from PartSegCore.channel_class import Channel
from PartSegCore.segmentation import RestartableAlgorithm
from PartSegCore.segmentation.algorithm_base import SegmentationResult, AdditionalLayerDescription
from PartSegCore.segmentation.threshold import (
    threshold_dict,
    BaseThreshold,
    DoubleThreshold,
    ManualThreshold,
)


class NeutrofileSegmentation(RestartableAlgorithm):
    def __init__(self):
        super().__init__()
        self.good, self.bad, self.nets, self.bacteria, self.net_area = 0, 0, 0, 0, 0

    @staticmethod
    def _range_threshold(data: np.ndarray, thr):
        thr1, thr2 = thr
        return (data > thr1) * (data < thr2)

    def calculation_run(self, report_fun: Callable[[str, int], None] = None) -> SegmentationResult:
        for key in self.new_parameters.keys():
            self.parameters[key] = deepcopy(self.new_parameters[key])
        channel = self.get_channel(self.new_parameters["dna_marker"])
        thr: BaseThreshold = threshold_dict[self.new_parameters["threshold"]["name"]]
        inner_dna_mask, thr_val = thr.calculate_mask(
            channel, self.mask, self.new_parameters["threshold"]["values"], operator.gt
        )
        _mask, alive_thr_val = thr.calculate_mask(
            channel, self.mask, self.new_parameters["threshold2"]["values"], operator.gt
        )
        dead_channel = self.get_channel(self.new_parameters["dead_dna_marker"])
        thr2: BaseThreshold = threshold_dict[self.new_parameters["dead_threshold"]["name"]]
        out_dna_mask, dead_thr_val = thr2.calculate_mask(
            dead_channel,
            self.mask,
            self.new_parameters["dead_threshold"]["values"],
            operator.gt,
        )

        inner_dna_components = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(inner_dna_mask)),
                self.new_parameters["minimum_size"],
            )
        )

        idn = inner_dna_components.copy()

        out_dna_components = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(out_dna_mask)),
                self.new_parameters["net_size"],
            )
        )

        odn = out_dna_components.copy()

        profile_info = MeasurementProfile.get_segmentation_to_mask_component(inner_dna_components, out_dna_components)

        result = np.zeros(inner_dna_components.shape, dtype=np.uint8)
        self.good, self.bad, self.nets, self.bacteria, self.net_area = 0, 0, 0, 0, 0
        sizes = np.bincount(inner_dna_components.flat)
        ratio = self.new_parameters["dead_ratio"]
        max_size = min(self.new_parameters["maximum_neutrofile_size"], self.new_parameters["net_size"])

        for val in np.unique(inner_dna_components):
            if val == 0:
                continue
            mm = inner_dna_components == val
            count = np.sum(out_dna_mask[mm])
            if profile_info.components_translation[val] and count >= ratio * sizes[val]:
                el = profile_info.components_translation[val][0]
                out_dna_components[mm] = el
                continue

            if sizes[val] < max_size:
                if count < ratio * sizes[val]:
                    result[mm] = 1
                    self.good += 1
                elif np.mean(channel[mm]) < alive_thr_val:
                    result[mm] = 3
                    self.bacteria += 1
                else:
                    result[mm] = 2
                    self.bad += 1
            elif sizes[val] < self.new_parameters["net_size"]:
                result[mm] = 3
                self.bacteria += 1
            else:
                print("problem here", val, sizes[val])
                self.nets += 1
                self.net_area += np.sum(mm)
        self.nets = np.max(out_dna_components)
        self.net_area = np.count_nonzero(out_dna_components)
        if self.new_parameters["separate_nets"]:
            result[out_dna_components > 0] = out_dna_components[out_dna_components > 0] + 4
        else:
            result[out_dna_components > 0] = 4
        return SegmentationResult(
            result,
            self.get_segmentation_profile(),
            {
                "inner dna": AdditionalLayerDescription(inner_dna_mask, "labels", "inner dna"),
                "inner dna com": AdditionalLayerDescription(idn, "labels", "inner dna com"),
                "outer dna": AdditionalLayerDescription(out_dna_mask, "labels", "out dna"),
                "outer dna com": AdditionalLayerDescription(odn, "labels", "outer dna com"),
            },
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
        return val
