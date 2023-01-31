import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml
from encord.objects.common import Shape
from encord.objects.ontology_object import Object
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.dataset import Image
from encord.utilities import label_utilities
from PIL import Image as pil_image
from PIL import ImageOps
from tqdm.auto import tqdm

from encord_active.lib.coco.datastructure import CocoAnnotation, CocoImage
from encord_active.lib.coco.parsers import (
    parse_annotations,
    parse_categories,
    parse_images,
    parse_info,
)
from encord_active.lib.coco.utils import make_object_dict
from encord_active.lib.encord.local_sdk import (
    FileTypeNotSupportedError,
    LocalDataRow,
    LocalDataset,
    LocalOntology,
    LocalProject,
    LocalUserClient,
)

IMAGE_DATA_UNIT_FILENAME = "image_data_unit.json"


def upload_img(
    dataset_tmp: LocalDataset,
    coco_image: CocoImage,
    image_path: Path,
) -> Optional[Image]:
    file_path = image_path / coco_image.file_name

    if not file_path.is_file():
        url = coco_image.coco_url or coco_image.flickr_url
        if url:
            img_data = requests.get(url).content
            file_path.write_bytes(img_data)
        else:
            print(f"Image {coco_image.file_name} not found")
            return None

    try:
        img = pil_image.open(file_path)
    except Exception:
        return None
    img_exif = img.getexif()
    if img_exif and (274 in img_exif):  # 274 corresponds to orientation key for EXIF metadata
        temp_file_name = "temp_image" + file_path.suffix
        img = ImageOps.exif_transpose(img)
        img.save(temp_file_name)

        try:
            encord_image = dataset_tmp.upload_image(
                title=str(coco_image.id_),
                file_path=temp_file_name,
            )
        except FileTypeNotSupportedError as e:
            print(f"{file_path} will be skipped as it doesn't seem to be an image.")
            encord_image = None
        finally:
            os.remove(temp_file_name)

        return encord_image
    else:
        try:
            return dataset_tmp.upload_image(
                title=str(coco_image.id_),
                file_path=file_path,
            )
        except FileTypeNotSupportedError as e:
            print(f"{file_path} will be skipped as it doesn't seem to be an image.")
            return None


def upload_annotation(
    project: LocalProject,
    annotations: Dict[int, List[CocoAnnotation]],
    data_row: LocalDataRow,
    id_to_obj: Dict[int, Object],
):
    data_hash = data_row.uid
    lr_dict = project.create_label_row(data_hash)
    label_hash = lr_dict["label_hash"]
    lr = project.get_label_row(label_hash, get_signed_url=False)
    try:
        annotation_list = annotations[int(data_row.title)]
    except KeyError:
        print(f"No annotations for {data_row.title}")
        annotation_list = []
    data_unit = lr["data_units"][data_hash]
    img_w, img_h = data_unit["width"], data_unit["height"]

    objects = []
    for annot in annotation_list:
        obj = id_to_obj[annot.category_id]
        polygon = annot.segmentation
        polygon_points = [(polygon[i] / img_w, polygon[i + 1] / img_h) for i in range(0, len(polygon), 2)]
        objects.append(make_object_dict(ontology_object=obj.to_dict(), object_data=polygon_points))

    lr["data_units"][data_hash]["labels"] = {"objects": objects, "classifications": []}
    updated_lr = label_utilities.construct_answer_dictionaries(lr)
    project.save_label_row(uid=label_hash, label=updated_lr)

    return data_hash, updated_lr


class CocoImporter:
    def __init__(
        self, images_dir_path: Path, annotations_file_path: Path, destination_dir: Path, use_symlinks: bool = False
    ):
        """
        Importer for COCO datasets.
        Args:
            images_dir_path (Path): Path where images are stored
            annotations_file_path (Path): The COCO JSON annotation file
            destination_dir (Path): Where to store the data
            use_symlinks (bool): If False, the importer will copy images.
                Otherwise, symlinks will be used to save disk space.
        """
        self.images_dir = images_dir_path
        self.annotations_file_path = annotations_file_path
        self.use_symlinks: bool = use_symlinks

        if not self.images_dir.is_dir():
            raise NotADirectoryError(f"Images directory '{self.images_dir}' doesn't exist")
        if not self.annotations_file_path.is_file():
            raise FileNotFoundError(f"Annotation file '{self.annotations_file_path}' doesn't exist")

        annotations_file = json.loads(self.annotations_file_path.read_text(encoding="utf-8"))
        self.info = parse_info(annotations_file["info"])
        self.categories = parse_categories(annotations_file["categories"])
        self.images = parse_images(annotations_file["images"])
        self.annotations = parse_annotations(annotations_file["annotations"])

        title = self.info.description.replace(" ", "-")
        self.title = f"[EA] {title}"
        self.project_dir = destination_dir / self.title

        self.user_client = LocalUserClient(self.project_dir)

    def create_dataset(self) -> LocalDataset:
        print(f"Creating a new dataset: {self.title}")
        dataset: LocalDataset = self.user_client.create_dataset(self.title, use_symlinks=self.use_symlinks)

        for _, coco_image in tqdm(self.images.items(), desc="Uploading images"):
            upload_img(dataset, coco_image, self.images_dir)

        return dataset

    def create_ontology(self) -> LocalOntology:
        print(f"Creating a new dataset: {self.title}")
        ontology_structure = OntologyStructure()
        for cat in self.categories:
            ontology_structure.add_object(name=cat.name, shape=Shape.POLYGON, uid=cat.id_)

        ontology: LocalOntology = self.user_client.create_ontology(
            self.title,
            description=self.info.description,
            structure=ontology_structure,
        )

        return ontology

    def create_project(
        self, dataset: LocalDataset, ontology: LocalOntology, ssh_key_path: Optional[Path]
    ) -> LocalProject:
        print(f"Creating a new project: {self.title}")
        project = self.user_client.create_project(
            project_title=self.title,
            dataset_hashes=[dataset.dataset_hash],
            ontology_hash=ontology.ontology_hash,
        )

        self.project_dir.mkdir(exist_ok=True)

        ontology_file = self.project_dir / "ontology.json"
        ontology_file.write_text(json.dumps(project.ontology))

        project_meta = {
            "project_title": project.title,
            "project_description": project.description,
            "project_hash": project.project_hash,
        }
        if ssh_key_path:
            project_meta["ssh_key_path"] = ssh_key_path.as_posix()
        meta_file_path = self.project_dir / "project_meta.yaml"
        meta_file_path.write_text(yaml.dump(project_meta), encoding="utf-8")

        id_to_obj = {obj.uid: obj for obj in ontology.structure.objects}

        label_row_meta = {lr["label_hash"]: lr for lr in project.label_rows}
        label_row_meta_file_path = self.project_dir / "label_row_meta.json"
        label_row_meta_file_path.write_text(json.dumps(label_row_meta, indent=2), encoding="utf-8")

        image_to_du = {}

        for data_unit in tqdm(dataset.data_rows, desc="Uploading annotations"):
            data_hash, lr = upload_annotation(project, self.annotations, data_unit, id_to_obj)
            image_id = lr["data_title"]
            image = self.images[image_id]
            image_to_du[image_id] = {"data_hash": data_hash, "height": image.height, "width": image.width}

        (Path(self.project_dir) / IMAGE_DATA_UNIT_FILENAME).write_text(json.dumps(image_to_du))

        return project
