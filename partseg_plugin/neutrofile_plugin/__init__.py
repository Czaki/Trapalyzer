def register():
    from PartSeg.utils.register import RegisterEnum, register as register_fun
    from .segmentation import NeutrodfileSegmentation
    register_fun(NeutrodfileSegmentation, RegisterEnum.analysis_algorithm)

