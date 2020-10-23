try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    reloading
except NameError:
    reloading = False
else:
    reloading = True


def register():
    from PartSegCore.register import RegisterEnum, register as register_fun
    from . import segmentation
    from . import measurement
    if reloading:
        import importlib
        importlib.reload(segmentation)
        importlib.reload(measurement)
    register_fun(segmentation.NeutrofileSegmentation, RegisterEnum.analysis_algorithm)
    register_fun(measurement.NetArea, RegisterEnum.analysis_measurement)
    register_fun(measurement.NetVoxels, RegisterEnum.analysis_measurement)
    register_fun(measurement.BacteriaArea, RegisterEnum.analysis_measurement)
    register_fun(measurement.BacteriaVoxels, RegisterEnum.analysis_measurement)
    register_fun(measurement.NetPercent, RegisterEnum.analysis_measurement)
    register_fun(measurement.AliveCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.DeadCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.NetCount, RegisterEnum.analysis_measurement)
    register_fun(measurement.BacteriaCount, RegisterEnum.analysis_measurement)
