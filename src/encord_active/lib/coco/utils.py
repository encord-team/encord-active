import uuid
from typing import Dict, List, Optional, Tuple, Union

from pycocotools.mask import decode, frPyObjects, merge

from encord_active.lib.common.time import get_timestamp
from encord_active.lib.labels.object import BoxShapes, ObjectShape

Point = Tuple[float, float]
BBOX_KEYS = {"x", "y", "h", "w"}


def __lower_snake_case(s: str):
    return "_".join(s.lower().split())


def make_object_dict(
    ontology_object: dict,
    object_data: Union[Point, List[Point], Dict[str, float], None] = None,
    object_hash: Optional[str] = None,
) -> dict:
    """
    :type ontology_object: The ontology object to associate with the ``object_data``.
    :type object_data: The data to put in the object dictionary. This has to conform
        with the ``shape`` parameter defined in
        ``ontology_object["shape"]``.
        - ``shape == "point"``: For key-points, the object data should be
          a tuple with x, y coordinates as floats.
        - ``shape == "bounding_box"``: For bounding boxes, the
          ``object_data`` needs to be a dict with info:
          {"x": float, "y": float, "h": float, "w": float} specifying the
          top right corner of the box (x, y) and the height and width of
          the bounding box.
        - ``shape in ("polygon", "polyline")``: For polygons and
          polylines, the format is an iterable  of points:
          [(x, y), ...] specifying (ordered) points in the
          polygon/polyline.
          If ``object_hash`` is none, a new hash will be generated.
    :type object_hash: If you want the object to have the same id across frames (for
        videos only), you can specify the object hash, which need to be
        an eight-character hex string (e.g., use
        ``str(uuid.uuid4())[:8]`` or the ``objectHash`` from an
        associated object.
    :returns: An object dictionary conforming with the Encord label row data format.
    """
    if object_hash is None:
        object_hash = str(uuid.uuid4())[:8]

    timestamp: str = get_timestamp()
    shape: str = ontology_object["shape"]

    object_dict = {
        "name": ontology_object["name"],
        "color": ontology_object["color"],
        "value": __lower_snake_case(ontology_object["name"]),
        "createdAt": timestamp,
        "createdBy": "robot@cord.tech",
        "confidence": 1,
        "objectHash": object_hash,
        "featureHash": ontology_object["featureNodeHash"],
        "lastEditedAt": timestamp,
        "lastEditedBy": "robot@encord.com",
        "shape": shape,
        "manualAnnotation": False,
        "reviews": [],
    }

    if shape in [ObjectShape.POLYGON, ObjectShape.POLYLINE] and object_data:
        if not isinstance(object_data, list):
            raise ValueError(f"The `object_data` for {shape} should be a list of points.")

        object_dict[shape] = {str(i): {"x": round(x, 4), "y": round(y, 4)} for i, (x, y) in enumerate(object_data)}

    elif shape == ObjectShape.KEY_POINT:
        if not isinstance(object_data, tuple):
            raise ValueError(f"The `object_data` for {shape} should be a tuple.")
        if len(object_data) != 2:
            raise ValueError(f"The `object_data` for {shape} should have two coordinates.")
        if not isinstance(object_data[0], float):
            raise ValueError(f"The `object_data` for {shape} should contain floats.")

        object_dict[shape] = {"0": {"x": round(object_data[0], 4), "y": round(object_data[1], 4)}}

    elif shape in BoxShapes:
        if not isinstance(object_data, dict):
            raise ValueError(f"The `object_data` for {shape} should be a dictionary.")
        if len(BBOX_KEYS.intersection(object_data.keys())) != 4:
            raise ValueError(f"The `object_data` for {shape} should have keys {BBOX_KEYS}.")
        if not isinstance(object_data["x"], float):
            raise ValueError(f"The `object_data` for {shape} should float values.")

        box = {k: round(v, 4) for k, v in object_data.items()}
        if shape == ObjectShape.ROTATABLE_BOUNDING_BOX:
            if "theta" not in object_data:
                raise ValueError(f"The `object_data` for {shape} should contain a `theta` field.")

            object_dict["rotatableBoundingBox"] = {**box, "theta": object_data["theta"]}
        else:
            object_dict["boundingBox"] = box

    return object_dict


def find_ontology_object(ontology: dict, encord_name: str):
    try:
        obj = next((o for o in ontology["objects"] if o["name"].lower() == encord_name.lower()))
    except StopIteration:
        raise ValueError(f"Couldn't match Encord ontology name `{encord_name}` to objects in the " f"Encord ontology.")
    return obj


def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann["segmentation"]
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = frPyObjects(segm, h, w)
        rle = merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann["segmentation"]
    return rle


def annToMask(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, h, w)
    return decode(rle)
