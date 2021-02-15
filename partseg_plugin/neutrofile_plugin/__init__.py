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
    register_fun(segmentation.NeutrofileSegmentation, RegisterEnum.analysis_algorithm)
    register_fun(segmentation.TrapezoidNeutrofileSegmentation, RegisterEnum.analysis_algorithm)
    register_fun(measurement.NetArea, RegisterEnum.analysis_measurement)
    register_fun(measurement.NetVoxels, RegisterEnum.analysis_measurement)
    register_fun(measurement.BacteriaArea, RegisterEnum.analysis_measurement)
    register_fun(measurement.BacteriaVoxels, RegisterEnum.analysis_measurement)
    register_fun(measurement.NetPercent, RegisterEnum.analysis_measurement)
    register_fun(measurement.AliveCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.DeadCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.NetCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.BacteriaCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.ClassifyNeutrofile, RegisterEnum.analysis_measurement)
    register_fun(widgets.qss_file, RegisterEnum._qss_register)
