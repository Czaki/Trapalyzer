import operator
import typing
from abc import ABC
from collections import Counter
from enum import Enum
from itertools import product

import numpy as np
import SimpleITK as sitk
from napari.layers import Image
from napari.types import LayerDataTuple
from nme import REGISTER, class_to_str, register_class
from pydantic import Field

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.measurement_calculation import Diameter, get_border
from PartSegCore.autofit import density_mass_center
from PartSegCore.channel_class import Channel
from PartSegCore.class_generator import enum_register
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation import RestartableAlgorithm
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription, SegmentationResult
from PartSegCore.segmentation.noise_filtering import NoiseFilterSelection
from PartSegCore.segmentation.threshold import BaseThreshold, ThresholdSelection, threshold_dict
from PartSegCore.utils import BaseModel

from .widgets import TrapezoidRange


class NeuType(Enum):
    PMN_neu = 1
    RND_neu = 2
    NER_neu = 3
    PMP_neu = 4
    Bacteria = 5
    NET = 8
    Unknown_intra = 10
    Unknown_extra = 11

    def __str__(self):
        return self.name.replace("_", " ")

    def __lt__(self, other):
        return self.value < other.value

    @classmethod
    def known_components(cls):
        return (x for x in cls.__members__.values() if x.value < 10)

    @classmethod
    def neutrofile_components(cls):
        return (x for x in cls.__members__.values() if x.value < 8)

    @classmethod
    def all_components(cls):
        return cls.__members__.values()


try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    reloading
except NameError:
    reloading = False  # means the module is being imported
    enum_register.register_class(NeuType)

LABELING_NAME = "Labeling"
SCORE_SUFFIX = "_score"
CATEGORY_STR = "category"
PARAMETER_TYPE_LIST = [
    "pixel count",
    "brightness",
    "ext. brightness",
    "brightness gradient",
]  # "brightness", "roundness"
DESCRIPTION_DICT = {
    NeuType.PMN_neu: "polymorphonuclear neutrophils",
    NeuType.RND_neu: "decondensed chromatin neutrophils",
    NeuType.NER_neu: "ruptured nuclear envelope neutrophils",
    NeuType.PMP_neu: "plasma membrane permeabilized neutrophils",
    NeuType.NET: "neutrophil extracellular trap",
    NeuType.Unknown_intra: "unclassified intracellurar component",
    NeuType.Unknown_extra: "unclassified extracellurar component",
    NeuType.Bacteria: "bacteria",
}


class NeutrofileSegmentationBase(RestartableAlgorithm, ABC):
    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return False

    def _calculate_mask(self, channel, threshold_info):
        thr: BaseThreshold = threshold_dict[threshold_info.name]
        mask, thr_val = thr.calculate_mask(channel, self.mask, threshold_info.values, operator.gt)
        return mask, thr_val

    @staticmethod
    def _calc_components(arr, min_size, close_holes=0):
        components = sitk.RelabelComponent(
            sitk.ConnectedComponent(sitk.GetImageFromArray(arr)),
            min_size,
        )

        if close_holes:
            binary1 = sitk.BinaryThreshold(components, 0, 0)
            removed_small = sitk.RelabelComponent(sitk.ConnectedComponent(binary1), close_holes)
            binary2 = sitk.BinaryThreshold(removed_small, 0, 0)
            components = sitk.RelabelComponent(
                sitk.ConnectedComponent(binary2),
                min_size,
            )
        return sitk.GetArrayFromImage(components)


def _migrate_trapalyzer_sub_fields(dkt):
    dkt = dkt.copy()
    cmp_name = [
        ("pmn_neu", "PMN_neu"),
        ("rnd_neu", "RND_neu"),
        ("ner_neu", "NER_neu"),
        ("pmp_neu", "PMP_neu"),
        ("bacteria", "Bacteria"),
    ]
    parameter_type_list = [
        ("pixel count", "pixel_count"),
        ("brightness", "brightness"),
        ("ext. brightness", "extracellular_brightness"),
        ("brightness gradient", "brightness_gradient"),
    ]
    for cmp_old, cmp_new in cmp_name:
        data = {new: dkt.pop(f"{cmp_old}_{old}") for old, new in parameter_type_list}
        data = REGISTER.migrate_data(class_to_str(NeutrophileComponentParameters), {}, data)
        dkt[cmp_new] = NeutrophileComponentParameters(**data)

    return dkt


@register_class(old_paths=["Trapalyzer.segmentation.NeutropeniaComponentParameters"])
class NeutrophileComponentParameters(BaseModel):
    pixel_count: TrapezoidRange = Field(default=(0, 1001), title="pixel count")
    brightness: TrapezoidRange = Field(default=(0, 1000), title="brightness")
    extracellular_brightness: TrapezoidRange = Field(default=(0, 1000), title="ext. brightness")
    brightness_gradient: TrapezoidRange = Field(default=(0, 1000), title="brightness gradient")


@register_class(version="0.1.0", migrations=[("0.1.0", _migrate_trapalyzer_sub_fields)])
class TrapalyzerParameters(BaseModel):
    inner_dna: Channel = Field(1, title="All DNA channel")
    inner_dna_noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Filter type")
    inner_threshold: ThresholdSelection = Field(ThresholdSelection.get_default(), title="Threshold type")
    outer_dna: Channel = Field(1, title="Extracellural DNA channel")
    outer_dna_noise_filtering: NoiseFilterSelection = Field(NoiseFilterSelection.get_default(), title="Filter type")
    outer_threshold: ThresholdSelection = Field(ThresholdSelection.get_default(), title="Threshold type")
    net_size: TrapezoidRange = Field((850, 999999), title="NET pixel count")
    net_ext_brightness: TrapezoidRange = Field((21, 100), title="NET extracellular brightness")
    net_ext_brightness_std: TrapezoidRange = Field((0.0, 1.0), title="NET ext. brightness SD")
    unknown_net: bool = Field(
        True, title="detect extracellular artifacts", description="If mark unknown net components"
    )
    PMN_neu: NeutrophileComponentParameters = Field(
        NeutrophileComponentParameters(), help_text=DESCRIPTION_DICT[NeuType.PMN_neu], title="<strong>PMN neu</strong>"
    )
    RND_neu: NeutrophileComponentParameters = Field(
        NeutrophileComponentParameters(), help_text=DESCRIPTION_DICT[NeuType.RND_neu], title="<strong>RND neu</strong>"
    )
    NER_neu: NeutrophileComponentParameters = Field(
        NeutrophileComponentParameters(), help_text=DESCRIPTION_DICT[NeuType.NER_neu], title="<strong>NER neu</strong>"
    )
    PMP_neu: NeutrophileComponentParameters = Field(
        NeutrophileComponentParameters(), help_text=DESCRIPTION_DICT[NeuType.PMP_neu], title="<strong>PMP neu</strong>"
    )
    Bacteria: NeutrophileComponentParameters = Field(
        NeutrophileComponentParameters(),
        help_text=DESCRIPTION_DICT[NeuType.Bacteria],
        title="<strong>Bacteria</strong>",
    )
    minimum_score: float = 0.8
    maximum_other: float = Field(0.4, title="Maximum competing score")
    minimum_size: int = Field(
        40, title="Minimum comp. pixel count", description="Minimum component pixel count", ge=1, le=9999
    )
    softness: float = Field(0.1, title="Error margin coeficient", ge=0, le=1)


class Trapalyzer(NeutrofileSegmentationBase):
    __argument_class__ = TrapalyzerParameters
    new_parameters = TrapalyzerParameters

    def __init__(self):
        super().__init__()
        self.count_dict = Counter()
        self.area_dict = Counter()
        self.nets = 0
        self.other = 0
        self.net_size = 0
        self.quality = 0
        self.image_size = 0

    def classify_nets(self, outer_dna_mask, cleaned_inner, cleaned_outer, net_size: TrapezoidRange):
        nets = self._calc_components(outer_dna_mask, int(net_size.lower_bound))
        laplacian_outer_image = _laplacian_estimate(cleaned_outer, 1.3)
        annotation = {}
        i = 1
        nets[nets > 0] += 1
        for val in np.unique(nets):
            if val == 0:
                continue
            component = np.array(nets == val)
            brightness_gradient = np.mean(laplacian_outer_image[component])
            # brightness_gradient_score = sine_score_function(
            #     brightness_gradient,
            #     softness=self.new_parameters["softness"],
            #     **self.new_parameters["net_brightness_gradient"],
            # )
            brightness_std = np.std(cleaned_outer[component])
            brightness_std_score = sine_score_function(
                np.std(cleaned_outer[component]),
                softness=self.new_parameters.softness,
                **self.new_parameters.net_ext_brightness_std.as_dict(),
            )
            brightness = np.quantile(cleaned_inner[component], 0.9)
            ext_brightness = np.quantile(cleaned_outer[component], 0.9)
            ext_brightness_score = sine_score_function(
                ext_brightness,
                softness=self.new_parameters.softness,
                **self.new_parameters.net_ext_brightness.as_dict(),
            )
            voxels = np.count_nonzero(component)
            voxels_score = sine_score_function(
                voxels, softness=self.new_parameters.softness, **self.new_parameters.net_size.as_dict()
            )

            if voxels_score * ext_brightness_score * brightness_std_score < self.new_parameters.minimum_score:
                if not self.new_parameters.unknown_net:
                    nets[component] = 0
                    continue
                else:
                    category = NeuType.Unknown_extra
            else:
                category = NeuType.NET

            data_dict = {
                "component_id": i,
                CATEGORY_STR: category,
                "pixel count": voxels,
                "brightness": brightness,
                "ext. brightness": ext_brightness,
                # "brightness gradient": np.mean(laplacian_image[component]),
                "ext. brightness gradient": brightness_gradient,
                "ext. brightness SD": brightness_std,
            }
            annotation[i] = data_dict
            nets[component] = i
            i += 1
        return nets, annotation

    def calculation_run(self, report_fun: typing.Callable[[str, int], None]) -> SegmentationResult:
        self.count_dict = Counter()
        self.area_dict = Counter()
        self.nets = 0
        self.other = 0
        self.net_size = 0
        inner_dna_channel = self.get_channel(self.new_parameters.inner_dna)
        self.image_size = inner_dna_channel.shape[0] * inner_dna_channel.shape[1] * inner_dna_channel.shape[2]
        inner_noise_filtering_parameters = self.new_parameters.inner_dna_noise_filtering
        cleaned_inner = NoiseFilterSelection[inner_noise_filtering_parameters.name].noise_filter(
            inner_dna_channel, self.image.spacing, inner_noise_filtering_parameters.values
        )
        inner_dna_mask, thr_val = self._calculate_mask(cleaned_inner, self.new_parameters.inner_threshold)

        outer_dna_channel = self.get_channel(self.new_parameters.outer_dna)
        outer_noise_filtering_parameters = self.new_parameters.outer_dna_noise_filtering
        cleaned_outer = NoiseFilterSelection[outer_noise_filtering_parameters.name].noise_filter(
            outer_dna_channel, self.image.spacing, outer_noise_filtering_parameters.values
        )
        outer_dna_mask, dead_thr_val = self._calculate_mask(cleaned_outer, self.new_parameters.outer_threshold)
        outer_dna_components, net_annotation = self.classify_nets(
            outer_dna_mask, cleaned_inner, cleaned_outer, self.new_parameters.net_size
        )
        inner_dna_mask[outer_dna_components > 0] = 0
        size_param_array = [getattr(self.new_parameters, x.name).pixel_count for x in NeuType.neutrofile_components()]

        min_object_size = int(
            max(
                5,
                min(
                    x.lower_bound
                    - (x.upper_bound - x.lower_bound)
                    * self.new_parameters.softness
                    * x.lower_bound
                    / (x.upper_bound - x.lower_bound)
                    for x in size_param_array
                ),
                self.new_parameters.minimum_size,
            )
        )
        inner_dna_components = self._calc_components(inner_dna_mask, min_object_size, close_holes=100)

        result_labeling, roi_annotation = self._classify_neutrofile(inner_dna_components, cleaned_inner, cleaned_outer)

        nets_components = [k for k, v in net_annotation.items() if v[CATEGORY_STR] == NeuType.NET]
        unknown_net_components = [k for k, v in net_annotation.items() if v[CATEGORY_STR] != NeuType.NET]

        self.count_dict[NeuType.NET] = len(nets_components)

        self.count_dict[NeuType.Unknown_extra] = len(unknown_net_components)

        result_labeling[np.isin(outer_dna_components, nets_components)] = NeuType.NET.value
        result_labeling[np.isin(outer_dna_components, unknown_net_components)] = NeuType.Unknown_extra.value
        max_component = inner_dna_components.max()
        inner_dna_components[outer_dna_components > 0] = outer_dna_components[outer_dna_components > 0] + max_component
        for value in np.unique(inner_dna_components[outer_dna_components > 0]):
            roi_annotation[value] = net_annotation[value - max_component]
            roi_annotation[value]["component_id"] = value
        alternative_representation = {LABELING_NAME: result_labeling}
        for val in NeuType.all_components():
            alternative_representation[str(val)] = (result_labeling == val.value).astype(np.uint8) * val.value
        self.net_size = np.count_nonzero(alternative_representation[str(NeuType.NET)])
        self.area_dict[NeuType.NET] = self.net_size

        from .measurement import QualityMeasure

        self.quality = QualityMeasure.calculate_property(inner_dna_components, roi_annotation)
        return SegmentationResult(
            inner_dna_components,
            self.get_segmentation_profile(),
            alternative_representation=alternative_representation,
            roi_annotation=roi_annotation,
            additional_layers={
                "inner_mask": AdditionalLayerDescription(inner_dna_mask.astype(np.uint8), "labels", "inner_mask"),
                "cleaned outer brightness gradient": AdditionalLayerDescription(
                    _laplacian_estimate(cleaned_outer, 1.3), "image", "cleaned outer brightness gradient"
                ),
                "cleaned inner brightness gradient": AdditionalLayerDescription(
                    _laplacian_estimate(cleaned_inner, 1.3), "image", "cleaned inner brightness gradient"
                ),
            },
        )

    def _classify_neutrofile(self, inner_dna_components, cleaned_inner, cleaned_outer):
        self.cell_count = 0
        laplacian_image = _laplacian_estimate(cleaned_inner, 1.3)
        annotation = {}
        labeling = np.zeros(inner_dna_components.shape, dtype=np.uint16)
        bounds = ROIInfo.calc_bounds(inner_dna_components)
        inner_dna_components_border = get_border(inner_dna_components)

        for val in np.unique(inner_dna_components):
            if val == 0:
                continue
            slices = tuple(bounds[val].get_slices(margin=5))
            component = np.array(inner_dna_components[slices] == val)
            component_border = inner_dna_components_border[slices] == val
            voxels = np.count_nonzero(component)
            perimeter = np.count_nonzero(component_border)
            diameter = Diameter.calculate_property(component, voxel_size=(1,) * component.ndim, result_scalar=1)

            if perimeter == 0:
                continue
            data_dict = {
                "pixel count": voxels,
                "brightness gradient": np.mean(laplacian_image[slices][component]),
                "brightness": np.quantile(cleaned_inner[slices][component], 0.9),
                "intensity": np.sum(cleaned_inner[slices][component]),
                # "homogenity": np.mean(inner_dna_channel[component]) / np.std(inner_dna_channel[component]),
                "ext. brightness": np.quantile(cleaned_outer[slices][component], 0.9),
                "ext. brightness SD": np.std(cleaned_outer[slices][component]),
                #                  "roundness": new_sphericity(component, self.image.voxel_size),
                #                 "roundness2": voxels / ((diameter ** 2 / 4) * pi),
                #                 "diameter": diameter,
                # "curvature": np.std(curvature(component_border_coords[0], component_border_coords[1])),
                #                 "border_size": component_border_coords[0].size,
                # "circumference": len(component_border_coords[0]),
                # "area to circumference": voxels / len(component_border_coords[0]),
                # "signal colocalization": colocalization1,
                "circularity": 3 * np.pi * voxels / perimeter**2,  # inspired by NETQUANT approach
                # "circularity2": voxels / ((diameter ** 2 / 4) * np.pi),
            }
            annotation[val] = dict(
                {"component_id": val, CATEGORY_STR: NeuType.Unknown_intra},
                **data_dict,
                **{
                    f"{prefix} {field.field_info.title}": sine_score_function(
                        data_dict[field.field_info.title],
                        softness=self.new_parameters.softness,
                        **getattr(getattr(self.new_parameters, prefix.name), suffix).as_dict(),
                    )
                    for prefix, (suffix, field) in product(
                        NeuType.neutrofile_components(), NeutrophileComponentParameters.__fields__.items()
                    )
                },
            )
            score_list = []
            for component_name in NeuType.neutrofile_components():
                score = 1
                for parameter in NeutrophileComponentParameters.__fields__.values():
                    score *= annotation[val][f"{component_name} {parameter.field_info.title}"]
                score_list.append((score, component_name))
                annotation[val][str(component_name) + SCORE_SUFFIX] = score

            score_list = sorted(score_list)
            if (
                score_list[-1][0] < self.new_parameters.minimum_score
                or score_list[-2][0] > self.new_parameters.maximum_other
            ):
                labeling[inner_dna_components == val] = NeuType.Unknown_intra.value
                self.count_dict[NeuType.Unknown_intra] += 1
            else:
                labeling[inner_dna_components == val] = score_list[-1][1].value
                self.count_dict[score_list[-1][1]] += 1
                self.cell_count += 1
                self.area_dict[score_list[-1][1]] += voxels
                annotation[val][CATEGORY_STR] = score_list[-1][1]

        return labeling, annotation

    def get_info_text(self):
        image_size = self.image_size or 1
        return (
            f"Annotation quality: {self.quality}"
            + "\n"
            + "\n".join(
                f"{name}: {self.count_dict[name]}, {self.area_dict[name]} px"
                f" ({round(100*self.area_dict[name]/image_size, 2)}% image area)"
                for name in [NeuType.NET, NeuType.Unknown_extra]
            )
            + "\n"
            + "\n".join(
                f"{name}: {self.count_dict[name]} ({round(100*self.count_dict[name] / (self.cell_count or 1), 2)}"
                f" % cells), {self.area_dict[name]} px ({round(100*self.area_dict[name]/image_size, 2)}%"
                f" image area)"
                for name in NeuType.neutrofile_components()
            )
        )

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile(name="", algorithm=self.get_name(), values=self.new_parameters.copy())

    @classmethod
    def get_name(cls) -> str:
        return "Trapalyzer"

    # @classmethod
    # def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
    #     initial = list(
    #         chain(
    #             *(
    #                 [
    #                     AlgorithmProperty(
    #                         f"{prefix.name.lower()}_{suffix}",
    #                         f"{prefix} {suffix}",
    #                         {"lower_bound": 0, "upper_bound": 1},
    #                         property_type=TrapezoidWidget,
    #                         help_text=f"{DESCRIPTION_DICT[prefix]} {suffix}",
    #                     )
    #                     for suffix in PARAMETER_TYPE_LIST
    #                 ]
    #                 + ["-------------------"]
    #                 for prefix in NeuType.neutrofile_components()
    #             )
    #         )
    #     ) + [
    #         AlgorithmProperty("minimum_score", "Minimum score", 0.8),
    #         AlgorithmProperty("maximum_other", "Maximum competing score", 0.4),
    #         AlgorithmProperty(
    #             "minimum_size", "Minimum comp. pixel count", 40, (1, 9999), help_text="Minimum component pixel count"
    #         ),
    #         AlgorithmProperty("softness", "Error margin coeficient", 0.1, (0, 1)),
    #     ]
    #
    #     thresholds = [
    #         AlgorithmProperty("inner_dna", "All DNA channel", 1, property_type=Channel),
    #         AlgorithmProperty(
    #             "inner_dna_noise_filtering",
    #             "Filter type",
    #             next(iter(noise_filtering_dict.keys())),
    #             possible_values=noise_filtering_dict,
    #             value_type=AlgorithmDescribeBase,
    #         ),
    #         AlgorithmProperty(
    #             "inner_threshold",
    #             "Threshold type",
    #             next(iter(threshold_dict.keys())),
    #             possible_values=threshold_dict,
    #             property_type=AlgorithmDescribeBase,
    #         ),
    #         AlgorithmProperty("outer_dna", "Extracellural DNA channel", 1, property_type=Channel),
    #         AlgorithmProperty(
    #             "outer_dna_noise_filtering",
    #             "Filter type",
    #             next(iter(noise_filtering_dict.keys())),
    #             possible_values=noise_filtering_dict,
    #             value_type=AlgorithmDescribeBase,
    #         ),
    #         AlgorithmProperty(
    #             "outer_threshold",
    #             "Threshold type",
    #             next(iter(threshold_dict.keys())),
    #             possible_values=threshold_dict,
    #             property_type=AlgorithmDescribeBase,
    #         ),
    #         AlgorithmProperty(
    #             "net_size",
    #             "NET pixel count",
    #             {"lower_bound": 850, "upper_bound": 999999},
    #             property_type=TrapezoidWidget,
    #         ),
    #         AlgorithmProperty(
    #             "net_ext_brightness",
    #             "NET extracellular brightness",
    #             {"lower_bound": 21, "upper_bound": 100},
    #             property_type=TrapezoidWidget,
    #         ),
    #         # AlgorithmProperty(
    #         #     "net_brightness_gradient",
    #         #     "NET ext. brightness gradient",
    #         #     {"lower_bound": -1.0, "upper_bound": 1.0},
    #         #     property_type=TrapezoidWidget,
    #         # ),
    #         AlgorithmProperty(
    #             "net_ext_brightness_std",
    #             "NET ext. brightness SD",
    #             {"lower_bound": 0.0, "upper_bound": 1.0},
    #             property_type=TrapezoidWidget,
    #         ),
    #         AlgorithmProperty(
    #             "unknown_net", "detect extracellular artifacts", True, help_text="If mark unknown net components"
    #         ),
    #         "-----------------------",
    #     ]
    #
    #     return thresholds + initial


def trapezoid_score_function(x, lower_bound, upper_bound, softness=0.5):
    """
    Compute a score on a scale from 0 to 1 that indicate whether values from x belong
    to the interval (lbound, ubound) with a softened boundary.
    If a point lies inside the interval, its score is equal to 1.
    If the point is further away than the interval length multiplied by the softness parameter,
    its score is equal to zero.
    Otherwise the score is given by a linear function.
    """
    interval_width = upper_bound - lower_bound
    subound = upper_bound + softness * interval_width
    slbound = lower_bound - (softness * interval_width) * 0.1
    swidth = softness * interval_width  # width of the soft boundary
    if lower_bound <= x <= upper_bound:
        return 1.0
    elif x <= slbound or x >= subound:
        return 0.0
    elif slbound <= x <= lower_bound:
        return 1.0 - (lower_bound - x) / swidth
    elif upper_bound <= x <= subound:
        return 1.0 - (x - upper_bound) / swidth


def gaussian_score_function(x, lbound, ubound, softness=0.5):
    """
    Computes a score on a scale from 0 to 1 whether x lies within the [lbound, ubound] interval.
    Fuzzified by a gaussian function.
    Softness controls the extension of the interval.
    """
    assert ubound >= lbound, "ubound needs to be >= than lbound"
    interval_mean = ubound + lbound
    interval_length = ubound - lbound
    if lbound <= x <= ubound:
        return 1.0
    sd_l = np.log(softness * interval_length * lbound / interval_mean) - np.log(3)  # to avoid raising to small powers
    sd_l = np.exp(sd_l)
    sd_u = np.log(softness * interval_length * ubound / interval_mean) - np.log(3)
    sd_u = np.exp(sd_u)
    if x <= lbound - 3.5 * sd_l or x >= ubound + 3.5 * sd_u:
        return 0.0
    if lbound - 3.5 * sd_l <= x <= lbound:
        logscore = -0.5 * (lbound - x) ** 2 / sd_l**2
        return np.exp(logscore)
    if ubound <= x <= ubound + 3.5 * sd_u:
        logscore = -0.5 * (x - ubound) ** 2 / sd_u**2
        return np.exp(logscore)


def sine_score_function(x, lower_bound, upper_bound, softness=0.5):
    """
    Computes a score on a scale from 0 to 1 whether x lies within the [lbound, ubound] interval.
    Extended with a sine function.
    Softness controls the extension of the interval.
    """
    assert upper_bound >= lower_bound, "upper_bound needs to be >= than lower_bound"
    interval_mean = upper_bound + lower_bound
    interval_length = upper_bound - lower_bound
    if lower_bound <= x <= upper_bound:
        return 1.0
    extension_span_left = softness * interval_length * lower_bound / interval_mean
    extension_span_right = softness * interval_length * upper_bound / interval_mean
    if x <= lower_bound - extension_span_left or x >= upper_bound + extension_span_right:
        return 0.0
    if lower_bound - extension_span_left <= x <= lower_bound:
        coord_transform = 2 * (x - lower_bound + extension_span_left) / extension_span_left - 1
    elif upper_bound <= x <= upper_bound + extension_span_right:
        coord_transform = 2 * (x - upper_bound + extension_span_right) / extension_span_right - 1
    else:
        raise RuntimeError("Something went terribly wrong!")
    return 0.5 + 0.5 * np.sin(0.5 * np.pi * coord_transform)


def new_sphericity(component: np.ndarray, voxel_size):
    component = component.squeeze()
    voxel_area = np.prod(voxel_size)
    center = np.array([density_mass_center(component, voxel_size)])
    area = np.count_nonzero(component) * voxel_area
    coordinates = np.transpose(np.nonzero(component)) * voxel_size - center
    radius_square = area / np.pi
    return np.sum(np.sum(coordinates**2, axis=1) <= radius_square) / (
        area / voxel_area + np.sum(np.sum(coordinates**2, axis=1) > radius_square)
    )


def laplacian_estimate(image: Image, radius=1.30, clip_bellow_0=True) -> LayerDataTuple:
    res = _laplacian_estimate(image.data[0], radius=radius)
    if clip_bellow_0:
        res[res < 0] = 0
    res = res.reshape(image.data.shape)
    return LayerDataTuple((res, {"colormap": "magma", "scale": image.scale, "name": "Laplacian estimate"}))


def _laplacian_estimate(channel: np.ndarray, radius=1.30) -> np.ndarray:
    data = channel.astype(np.float64).squeeze()
    return -sitk.GetArrayFromImage(sitk.LaplacianRecursiveGaussian(sitk.GetImageFromArray(data), radius)).reshape(
        channel.shape
    )


def curvature(x, y, *args):
    assert len(x) == len(y)
    n = len(x)
    k = np.zeros(n)
    for i in range(n):
        ddx = x[(i + 1) % n] + x[(i - 1) % n] - 2 * x[i]
        ddx *= n**2
        ddy = y[(i + 1) % n] + y[(i - 1) % n] - 2 * y[i]
        ddy *= n**2
        k[i] = np.sqrt(ddx**2 + ddy**2)
    return k
