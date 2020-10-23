from abc import ABC
from typing import Union
from PartSegCore.analysis import measurement_calculation
from PartSegCore.analysis.measurement_base import Leaf, AreaType, PerComponent, MeasurementMethodBase
import SimpleITK as sitk
import numpy as np

from .segmentation import BACTERIA_VAL, NET_VAL, DEAD_VAL, ALIVE_VAL

def count_components(area_array: Union[np.ndarray, bool]) -> int:
    return sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
        sitk.GetImageFromArray(np.array(area_array > 0).astype(np.uint8))))).max()


class AreaBase(MeasurementMethodBase, ABC):
    @classmethod
    def get_units(cls, ndim):
        return measurement_calculation.Volume.get_units(ndim)

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.No)


class NetArea(AreaBase):
    text_info = "Neutrofile net area", "Calculate area of neutrofile nets"

    @staticmethod
    def calculate_property(area_array, **kwargs):
        return measurement_calculation.Volume.calculate_property(area_array >= NET_VAL, **kwargs)


class BacteriaArea(AreaBase):
    text_info = "Bacteria groups area", "Calculate area of bacteria groups"

    @staticmethod
    def calculate_property(area_array, **kwargs):
        return measurement_calculation.Volume.calculate_property(area_array == BACTERIA_VAL, **kwargs)


class VoxelBase(AreaBase, ABC):
    @classmethod
    def get_units(cls, ndim):
        return measurement_calculation.Voxels.get_units(ndim)


class NetVoxels(AreaBase):
    text_info = "Neutrofile net pixels", "Calculate number of voxels of neutrofile nets"

    @staticmethod
    def calculate_property(area_array, **kwargs):
        return measurement_calculation.Voxels.calculate_property(area_array >= NET_VAL, **kwargs)


class BacteriaVoxels(AreaBase):
    text_info = "Bacteria groups pixels", "Calculate number of voxels of bacteria groups"

    @staticmethod
    def calculate_property(area_array, **kwargs):
        return measurement_calculation.Voxels.calculate_property(area_array == BACTERIA_VAL, **kwargs)


class NetPercent(MeasurementMethodBase):
    text_info = "Neutrofile net percent", "Total percentage occupied by neutrofile nets"
    @classmethod
    def get_units(cls, ndim):
        return "%"

    @staticmethod
    def calculate_property(area_array, **kwargs):
        return measurement_calculation.Volume.calculate_property(area_array >=NET_VAL, **kwargs)/\
               measurement_calculation.Volume.calculate_property(area_array >=0, **kwargs)*100

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.No)


class CountBase(MeasurementMethodBase, ABC):
    @classmethod
    def get_units(cls, ndim):
        return 1

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.No)


class AliveCount(CountBase):
    text_info = "Neutrofile alive count", "Count alive cells in neutrofiles"

    @classmethod
    def calculate_property(cls, area_array, **kwargs):
        return sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.GetImageFromArray(np.array(area_array == ALIVE_VAL).astype(np.uint8))))).max()


class DeadCount(CountBase):
    text_info = "Neutrofile dead count", "Count dead cells in neutrofiles"

    @classmethod
    def calculate_property(cls, area_array, **kwargs):
        return count_components(area_array==DEAD_VAL)


class BacteriaCount(CountBase):
    text_info = "Bacteria count", "Count groups in neutrofiles"

    @classmethod
    def calculate_property(cls, area_array, **kwargs):
        return count_components(area_array==BACTERIA_VAL)


class NetCount(CountBase):
    text_info = "Neutrofile net count", "Count net components in neutrofiles"

    @classmethod
    def calculate_property(cls, area_array, **kwargs):
        return count_components(area_array>=NET_VAL)
