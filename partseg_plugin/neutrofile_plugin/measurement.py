from PartSeg.utils.analysis import statistics_calculation
from PartSeg.utils.analysis.measurement_base import Leaf, AreaType, PerComponent, StatisticMethodBase
import SimpleITK as sitk
import numpy as np


class NetArea(StatisticMethodBase):
    text_info = "Neutrofile net area", "Calculate area of neutrofile nets"
    @classmethod
    def get_units(cls, ndim):
        return statistics_calculation.Volume.get_units(ndim)

    @staticmethod
    def calculate_property(area_array, **kwargs):
        return statistics_calculation.Volume.calculate_property(area_array > 2, **kwargs)

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Segmentation, per_component=PerComponent.No)


class NetPercent(StatisticMethodBase):
    text_info = "Neutrofile net percent", "Total percentage occupied by neutrofile nets"
    @classmethod
    def get_units(cls, ndim):
        return "%"

    @staticmethod
    def calculate_property(area_array, **kwargs):
        return statistics_calculation.Volume.calculate_property(area_array > 2, **kwargs)/\
               statistics_calculation.Volume.calculate_property(area_array >=0, **kwargs)*100

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Segmentation, per_component=PerComponent.No)


class AliveCount(StatisticMethodBase):
    text_info = "Neutrofile Alive count", "Count alive cells in neutrofiles"

    @classmethod
    def get_units(cls, ndim):
        return 1

    @classmethod
    def calculate_property(cls, area_array, **kwargs):
        return sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.GetImageFromArray(np.array(area_array == 1).astype(np.uint8))))).max()

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Segmentation, per_component=PerComponent.No)


class DeadCount(StatisticMethodBase):
    text_info = "Neutrofile dead count", "Count dead cells in neutrofiles"

    @classmethod
    def get_units(cls, ndim):
        return 1

    @classmethod
    def calculate_property(cls, area_array, **kwargs):
        return sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.GetImageFromArray(np.array(area_array == 2).astype(np.uint8))))).max()

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Segmentation, per_component=PerComponent.No)


class NetCount(StatisticMethodBase):
    text_info = "Neutrofile net count", "Count net components in neutrofiles"

    @classmethod
    def get_units(cls, ndim):
        return 1

    @classmethod
    def calculate_property(cls, area_array, **kwargs):
        return sitk.GetArrayFromImage(sitk.RelabelComponent(sitk.ConnectedComponent(
            sitk.GetImageFromArray(np.array(area_array > 2).astype(np.uint8))))).max()

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Segmentation, per_component=PerComponent.No)