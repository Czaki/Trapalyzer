from abc import ABC

import numpy as np
from sympy import symbols

from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.analysis import measurement_calculation
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementMethodBase, PerComponent

from .segmentation import LABELING_NAME, PARAMETER_TYPE_LIST, SCORE_SUFFIX, NeuType


class ComponentArea(MeasurementMethodBase):
    text_info = "Componet type area", "Calculate area of given component type"

    @classmethod
    def get_units(cls, ndim):
        return measurement_calculation.Volume.get_units(ndim)

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.No)

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("component_type", "Component type", NeuType.PMN_neu, possible_values=NeuType)]

    @staticmethod
    def calculate_property(roi_alternative, component_type, **kwargs):
        area_array = roi_alternative[LABELING_NAME]
        kwargs = dict(kwargs)
        del kwargs["area_array"]
        return measurement_calculation.Volume.calculate_property(area_array == component_type.value, **kwargs)


class ComponentVoxels(MeasurementMethodBase):
    text_info = "Component type pixels", "Calculate number of voxels of given component type"

    @classmethod
    def get_units(cls, ndim):
        return measurement_calculation.Voxels.get_units(ndim)

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("component_type", "Component type", NeuType.PMN_neu, possible_values=NeuType)]

    @staticmethod
    def calculate_property(roi_alternative, component_type, **kwargs):
        area_array = roi_alternative[LABELING_NAME]
        kwargs = dict(kwargs)
        del kwargs["area_array"]
        return measurement_calculation.Voxels.calculate_property(area_array == component_type.value, **kwargs)


class ComponentCount(MeasurementMethodBase):
    text_info = "component type count", "Count elements of given component type"

    @classmethod
    def get_units(cls, ndim):
        return 1

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.No)

    @classmethod
    def calculate_property(cls, area_array, roi_alternative, component_type, **kwargs):
        return len(np.unique(area_array[roi_alternative[LABELING_NAME] == component_type.value]))

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("component_type", "Component type", NeuType.PMN_neu, possible_values=NeuType)]


class ClassifyNeutrofile(MeasurementMethodBase, ABC):
    text_info = "Classify neutrophil", "Classify if component is alive orr dead neutrophil, bacteria group or net"

    @classmethod
    def get_units(cls, ndim):
        return symbols("Text")

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.Yes)

    @staticmethod
    def calculate_property(area_array, roi_alternative, **kwargs):
        labels = roi_alternative[LABELING_NAME]
        numbers = np.unique(labels[area_array > 0])
        if numbers.size != 1:
            raise ValueError(f"Component need {np.unique(labels)} to have single label not {numbers}")
        return str(NeuType(numbers[0]))


class NeutrofileScore(MeasurementMethodBase):
    text_info = "Get score", "Get score for given type of components"

    @classmethod
    def get_units(cls, ndim):
        return symbols("Text")

    @classmethod
    def get_fields(cls):
        names = [str(x) + SCORE_SUFFIX for x in NeuType.all_components()]
        return [AlgorithmProperty("score_type", "Score type", names[0], possible_values=names)]

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.Yes)

    @staticmethod
    def calculate_property(roi_annotation, _component_num, score_type, **kwargs):
        return roi_annotation[_component_num].get(score_type)


class NeutrofileParameter(MeasurementMethodBase):
    text_info = "Get parameter", "Get parameter of components"

    @classmethod
    def get_units(cls, ndim):
        return symbols("Text")

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty(
                "parameter_name", "Parameter Name", PARAMETER_TYPE_LIST[0], possible_values=PARAMETER_TYPE_LIST
            )
        ]

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.Yes)

    @staticmethod
    def calculate_property(roi_annotation, _component_num, parameter_name, **kwargs):
        return roi_annotation[_component_num].get(parameter_name)


class NetPercent(MeasurementMethodBase):
    text_info = "NET percent coverage", "Total percentage occupied by neutrophil nets"

    @classmethod
    def get_units(cls, ndim):
        return "%"

    @staticmethod
    def calculate_property(roi_alternative, **kwargs):
        area_array = roi_alternative[LABELING_NAME]
        return (
            measurement_calculation.Volume.calculate_property(area_array == NeuType.NET, **kwargs)
            / measurement_calculation.Volume.calculate_property(area_array >= 0, **kwargs)
            * 100
        )

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.No)


class ComponentMid(MeasurementMethodBase):
    text_info = "Component position", "Position of component as string"

    @classmethod
    def get_units(cls, ndim):
        return "str"

    @staticmethod
    def calculate_property(bounds_info, _component_num, **kwargs):
        return str(bounds_info[_component_num])

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.Yes)
