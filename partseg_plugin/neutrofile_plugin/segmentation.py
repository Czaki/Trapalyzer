import operator
import typing
from copy import deepcopy
from typing import Callable
import SimpleITK as sitk
import numpy as np

from PartSeg.utils.algorithm_describe_base import AlgorithmProperty, AlgorithmDescribeBase
from PartSeg.utils.channel_class import Channel
from PartSeg.utils.segmentation import RestartableAlgorithm
from PartSeg.utils.segmentation.algorithm_base import SegmentationResult
from PartSeg.utils.segmentation.threshold import threshold_dict, BaseThreshold, DoubleThreshold


class NeutrofileSegmentation(RestartableAlgorithm):
    def __init__(self):
        super().__init__()
        self.good, self.bad, self.nets, self.bacteria = 0, 0, 0, 0

    @staticmethod
    def _range_threshold(data: np.ndarray, thr):
        thr1, thr2 = thr
        return (data > thr1) * (data < thr2)

    def calculation_run(self, report_fun: Callable[[str, int], None] = None) -> SegmentationResult:
        for key in self.new_parameters.keys():
            self.parameters[key] = deepcopy(self.new_parameters[key])
        channel = self.get_channel(self.new_parameters["dna_marker"])
        dead_channel = self.get_channel(self.new_parameters["dead_dna_marker"])
        mask, _ = DoubleThreshold.calculate_mask(
            channel, self.mask, {"core_threshold": self.new_parameters["threshold"],
                                 "base_threshold": self.new_parameters["threshold2"]}, operator.gt)
        thr2: BaseThreshold = threshold_dict[self.new_parameters["dead_threshold"]["name"]]
        dead_mask, dead_thr_val = thr2.calculate_mask(dead_channel, self.mask,
                                                      self.new_parameters["dead_threshold"]["values"],
                                                      operator.gt)

        connect = sitk.ConnectedComponent(sitk.GetImageFromArray((mask == 2).astype(np.uint8)))
        self.segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(connect, self.new_parameters["minimum_size"]))
        net_mask = (mask > 0) * (self.segmentation == 0) * dead_mask
        connect_net = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.GetImageFromArray(net_mask.astype(np.uint8))), self.new_parameters["net_size"]))

        result = np.zeros(self.segmentation.shape, dtype=np.uint8)
        self.good, self.bad, self.nets, self.bacteria = 0, 0, 0, 0
        sizes = np.bincount(self.segmentation.flat)
        ratio = self.new_parameters["dead_ratio"]
        for val in np.unique(self.segmentation)[1:]:
            mm = self.segmentation == val
            count = np.sum(dead_mask[mm])
            if sizes[val] > self.new_parameters["net_size"]:
                result[mm] = 3
                self.bacteria += 1
            elif count < ratio * sizes[val]:
                result[mm] = 1
                self.good += 1
            else:
                result[mm] = 2
                self.bad += 1
        self.nets = np.max(connect_net)
        if self.new_parameters["separate_nets"]:
            connect_net[connect_net > 0] += 3
            result[connect_net > 0] = connect_net[connect_net > 0]
        else:
            result[connect_net > 0] = 4
        return SegmentationResult(result, self.get_segmentation_profile(), self.segmentation, None)

    def get_info_text(self):
        return f"Alive: {self.good}, dead: {self.bad}, nets: {self.nets}, bacteria groups{self.bacteria}"

    def set_parameters(self, dna_marker, threshold, threshold2, dead_dna_marker, dead_threshold, minimum_size, net_size,
                       separate_nets, dead_ratio):
        self.new_parameters["dna_marker"] = dna_marker
        self.new_parameters["threshold"] = threshold
        self.new_parameters["threshold2"] = threshold2
        self.new_parameters["dead_dna_marker"] = dead_dna_marker
        self.new_parameters["dead_threshold"] = dead_threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["net_size"] = net_size
        self.new_parameters["separate_nets"] = separate_nets
        self.new_parameters["dead_ratio"] = dead_ratio

    @classmethod
    def get_name(cls) -> str:
        return "Segment Neutrofile"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [AlgorithmProperty("dna_marker", "DNA marker", 1, property_type=Channel),
                AlgorithmProperty("threshold", "Threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("threshold2", "Threshold dead area", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("dead_dna_marker", "Dead DNA marker", 0, property_type=Channel),
                AlgorithmProperty("dead_threshold", "Dead Threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("minimum_size", "Cell Min size (px)", 20, (0, 10 ** 6), 1000),
                AlgorithmProperty("net_size", "Cell max size (px)", 500, (0, 10 ** 6), 100),
                AlgorithmProperty("separate_nets", "Mark nets separate", True),
                AlgorithmProperty("dead_ratio", "Dead marker ratio", 0.5, (0, 1))
                ]
