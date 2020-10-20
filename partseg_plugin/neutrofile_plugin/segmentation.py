import operator
import typing
from copy import deepcopy
from typing import Callable
import SimpleITK as sitk
import numpy as np

from PartSegCore.algorithm_describe_base import AlgorithmProperty, AlgorithmDescribeBase
from PartSegCore.channel_class import Channel
from PartSegCore.segmentation import RestartableAlgorithm
from PartSegCore.segmentation.algorithm_base import SegmentationResult
from PartSegCore.segmentation.threshold import threshold_dict, BaseThreshold


class NeutrodfileSegmentation(RestartableAlgorithm):
    def __init__(self):
        super().__init__()
        self.good, self.bad, self.nets = 0, 0, 0

    def calculation_run(self, report_fun: Callable[[str, int], None] = None) -> SegmentationResult:
        for key in self.new_parameters.keys():
            self.parameters[key] = deepcopy(self.new_parameters[key])
        channel = self.get_channel(self.new_parameters["dna_marker"])
        dead_channel = self.get_channel(self.new_parameters["dead_dna_marker"])
        thr: BaseThreshold = threshold_dict[self.new_parameters["threshold"]["name"]]
        mask, thr_val = thr.calculate_mask(channel, self.mask, self.new_parameters["threshold"]["values"],
                                           operator.gt)
        dead_mask, dead_thr_val = thr.calculate_mask(dead_channel, self.mask,
                                                     self.new_parameters["dead_threshold"]["values"],
                                                     operator.gt)
        connect = sitk.ConnectedComponent(sitk.GetImageFromArray(mask))
        self.segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(connect, self.new_parameters["minimum_size"]))
        result = np.zeros(self.segmentation.shape, dtype=np.uint8)
        self.good, self.bad, self.nets = 0, 0, 0
        for val in np.unique(self.segmentation)[1:]:
            mm = self.segmentation == val
            count = np.sum(dead_mask[mm])
            if count < 20:
                result[mm] = 1
                self.good += 1
            elif count < self.new_parameters["net_size"]:
                result[mm] = 2
                self.bad += 1
            else:
                result[mm] = 3
                self.nets += 1
        dead_mask[result > 0] = 0
        dead_mask = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.GetImageFromArray(dead_mask)), self.new_parameters["net_size"]//2))
        result[dead_mask > 0] = 4
        nets_components = sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.GetImageFromArray((result > 2).astype(np.uint8))), self.new_parameters["net_size"]))
        self.nets = nets_components.max()
        if self.new_parameters["separate_nets"]:
            nets_components[nets_components > 0] += 2
            result[nets_components > 0] = nets_components[nets_components > 0]
        return SegmentationResult(result, self.get_segmentation_profile())

    def get_info_text(self):
        return f"Alive: {self.good}, dead: {self.bad}, nets: {self.nets}"

    def set_parameters(self, dna_marker, threshold, dead_dna_marker, dead_threshold, minimum_size, net_size,
                       separate_nets):
        self.new_parameters["dna_marker"] = dna_marker
        self.new_parameters["threshold"] = threshold
        self.new_parameters["dead_dna_marker"] = dead_dna_marker
        self.new_parameters["dead_threshold"] = dead_threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["net_size"] = net_size
        self.new_parameters["separate_nets"] = separate_nets

    @classmethod
    def get_name(cls) -> str:
        return "Segment Neutrofile"

    @staticmethod
    def single_channel():
        return False

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [AlgorithmProperty("dna_marker", "DNA marker", 1, property_type=Channel),
                AlgorithmProperty("threshold", "Threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("dead_dna_marker", "Dead DNA marker", 0, property_type=Channel),
                AlgorithmProperty("dead_threshold", "Dead Threshold", next(iter(threshold_dict.keys())),
                                  possible_values=threshold_dict, property_type=AlgorithmDescribeBase),
                AlgorithmProperty("minimum_size", "Minimum size (px)", 20, (0, 10 ** 6), 1000),
                AlgorithmProperty("net_size", "net size (px)", 500, (0, 10 ** 6), 100),
                AlgorithmProperty("separate_nets", "Mark nets separate", True)
                ]
