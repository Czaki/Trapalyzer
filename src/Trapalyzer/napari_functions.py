from collections import defaultdict
from xml.etree import ElementTree as ET

import numpy as np
from magicgui import magic_factory
from napari.layers import Labels

COLOR_DICT = {"decondensed": (0, 1, 0), "NET": (0, 0, 1), "dead": (1, 0, 0)}


@magic_factory(info={"widget_type": "TextEdit"}, call_button=True)
def count_size(
    segmentation: Labels,
    info: str = "",
) -> None:
    count = np.bincount(segmentation.data.flat)
    count_size.info.value = "\n".join(f"{x}: {y}" for x, y in enumerate(count))


def load_annnotation(file_path):
    data = ET.parse(file_path)

    objects = data.findall("object")

    data = defaultdict(list)

    for object in objects:
        box = object.find("bndbox")
        data[object.find("name").text].append([int(box.find(x).text) for x in ["ymin", "xmin", "ymax", "xmax"]])

    res = defaultdict(list)

    for name, values in data.items():
        for box in values:
            res[name].append([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]), (box[2], box[1])])

    return [
        (
            values,
            {
                "name": name,
                "shape_type": "polygon",
                "opacity": 0.3,
                "edge_color": [COLOR_DICT.get(name, "white")],
                "edge_width": 4,
                "face_color": [COLOR_DICT.get(name, (1, 1, 1)) + (0.2,)],
            },
            "shapes",
        )
        for name, values in res.items()
    ]
