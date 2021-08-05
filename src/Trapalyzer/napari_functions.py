import numpy as np
from magicgui import magic_factory
from napari.layers import Labels


@magic_factory(info={"widget_type": "TextEdit"}, call_button=True)
def count_size(
    segmentation: Labels,
    info: str = "",
) -> None:
    count = np.bincount(segmentation.data.flat)
    count_size.info.value = "\n".join(f"{x}: {y}" for x, y in enumerate(count))
