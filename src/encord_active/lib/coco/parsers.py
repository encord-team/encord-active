from typing import Dict, List

from tqdm.auto import tqdm

from encord_active.lib.coco.datastructure import (
    CocoAnnotation,
    CocoCategory,
    CocoImage,
    CocoInfo,
    CocoResult,
)
from encord_active.lib.coco.utils import annToMask
from encord_active.lib.common.utils import mask_to_polygon


def parse_info(info: Dict) -> CocoInfo:
    return CocoInfo(
        contributor=info["contributor"],
        date_created=info["date_created"],
        url=info["url"],
        version=info["version"],
        year=info["year"],
        description=info["description"],
    )


def parse_categories(categories: Dict) -> List[CocoCategory]:
    return [
        CocoCategory(
            supercategory=category["supercategory"] if "supercategory" in category.keys() else "",
            id_=category["id"],
            name=category["name"],
        )
        for category in tqdm(categories, desc="Parsing categories")
    ]


def parse_images(images: Dict) -> Dict[str, CocoImage]:
    return {
        str(image["id"]): CocoImage(
            id_=image["id"],
            height=image["height"],
            width=image["width"],
            file_name=image["file_name"],
            coco_url=image.get("coco_url"),
            flickr_url=image.get("flickr_url"),
        )
        for image in tqdm(images, desc="Parsing images")
    }


def parse_annotations(annotations: List[Dict]) -> Dict[int, List[CocoAnnotation]]:
    annot_dict: Dict[int, List[CocoAnnotation]] = {}
    for annotation in tqdm(annotations, desc="Parsing annotations"):
        if annotation["iscrowd"] == 1:
            continue

        segmentations = annotation["segmentation"]

        if not segmentations:
            segmentations = [[]]
        elif isinstance(segmentations, list) and not isinstance(segmentations[0], list):
            segmentations = [segmentations]
        elif isinstance(segmentations, dict):
            h, w = segmentations["size"]
            mask = annToMask(annotation, h=h, w=w)
            poly, inferred_bbox = mask_to_polygon(mask)
            if poly is None or inferred_bbox != annotation["bbox"]:
                print(f"Annotation '{annotation['id']}', contains an invalid polygon. Skipping ...")
                continue
            segmentations = [poly]

        annot_dict.setdefault(annotation["image_id"], [])

        for segment in segmentations:
            annot_dict[annotation["image_id"]].append(
                CocoAnnotation(
                    area=annotation["area"],
                    bbox=annotation["bbox"],
                    category_id=annotation["category_id"],
                    id_=annotation["id"],
                    image_id=annotation["image_id"],
                    iscrowd=annotation["iscrowd"],
                    segmentation=segment,
                    rotation=annotation.get("attributes", {}).get("rotation"),
                )
            )

    return annot_dict


def parse_results(results: List[Dict]):
    coco_results: List[CocoResult] = []
    for result in tqdm(results, desc="Parsing results"):
        segmentations = result["segmentation"]
        bbox = result.get("bbox")

        if isinstance(segmentations, list) and not isinstance(segmentations[0], list):
            segmentations = [segmentations]
        elif isinstance(segmentations, dict):
            h, w = segmentations["size"]
            mask = annToMask(result, h=h, w=w)
            poly, inferred_bbox = mask_to_polygon(mask)
            if poly is None or (bbox is not None and inferred_bbox != bbox):
                print(f"Annotation '{result['id']}', contains an invalid polygon. Skipping ...")
                continue
            bbox = inferred_bbox
            segmentations = [poly]

        for segment in segmentations:
            coco_results.append(
                CocoResult(
                    bbox=bbox,
                    category_id=result["category_id"],
                    image_id=result["image_id"],
                    segmentation=segment,
                    score=result["score"],
                )
            )

    return coco_results
