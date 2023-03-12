from pycocotools.mask import decode, frPyObjects, merge


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
