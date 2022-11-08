import os

from napari_plugin_engine import napari_hook_implementation

from .napari_functions import count_size, load_annnotation
from .segmentation import laplacian_estimate

try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    reloading
except NameError:
    reloading = False
else:
    reloading = True


def register():
    from PartSegCore.register import RegisterEnum
    from PartSegCore.register import register as register_fun

    from . import measurement, segmentation, widgets

    if reloading:
        import importlib

        importlib.reload(segmentation)
        importlib.reload(measurement)
        importlib.reload(widgets)
    # register_fun(segmentation.NeutrofileSegmentation, RegisterEnum.analysis_algorithm)
    register_fun(segmentation.Trapalyzer, RegisterEnum.analysis_algorithm)
    register_fun(measurement.ComponentArea, RegisterEnum.analysis_measurement)
    register_fun(measurement.NetPercent, RegisterEnum.analysis_measurement)
    register_fun(measurement.ComponentCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.ClassifyNeutrofile, RegisterEnum.analysis_measurement)
    register_fun(measurement.NeutrofileScore, RegisterEnum.analysis_measurement)
    register_fun(measurement.NeutrofileParameter, RegisterEnum.analysis_measurement)
    register_fun(measurement.ComponentMid, RegisterEnum.analysis_measurement)
    register_fun(measurement.QualityMeasure, RegisterEnum.analysis_measurement)
    register_fun(widgets.qss_file, RegisterEnum._qss_register)


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget2():
    return count_size, {"name": "Components size help"}


@napari_hook_implementation(specname="napari_experimental_provide_function")
def napari_experimental_provide_function3():
    return laplacian_estimate


@napari_hook_implementation
def napari_get_reader(path: str):
    if os.path.splitext(path)[1] == ".xml":
        return load_annnotation
