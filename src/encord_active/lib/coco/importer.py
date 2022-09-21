import json
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional

import requests
import typer
import yaml
from encord import Dataset, EncordUserClient, Project
from encord.objects.common import Shape
from encord.objects.ontology_object import Object
from encord.objects.ontology_structure import OntologyStructure
from encord.ontology import Ontology
from encord.orm.dataset import CreateDatasetResponse, DataRow, Image, StorageLocation
from encord.utilities import label_utilities
from tqdm import tqdm

from encord_active.lib.coco.datastructure import CocoAnnotation, CocoImage
from encord_active.lib.coco.parsers import (
    parse_annotations,
    parse_categories,
    parse_images,
    parse_info,
)
from encord_active.lib.coco.utils import make_object_dict

IMAGE_DATA_UNIT_FILENAME = "image_data_unit.json"


def upload_img(
    dataset_tmp: Dataset,
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

    return dataset_tmp.upload_image(
        title=str(coco_image.id_),
        file_path=file_path,
    )


def upload_annotation(
    project: Project,
    annotations: Dict[int, List[CocoAnnotation]],
    data_unit: DataRow,
    id_to_obj: Dict[int, Object],
):
    data_hash = data_unit["data_hash"]
    lr_dict = project.create_label_row(data_hash)
    label_hash = lr_dict["label_hash"]
    lr = project.get_label_row(label_hash, get_signed_url=False)
    try:
        annotation_list = annotations[int(data_unit["data_title"])]
    except KeyError:
        print(f"No annotations for {data_unit['data_title']}")
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


def build_local_lr(
    image: CocoImage,
    image_path: Path,
    data_hash: str,
    lr: label_utilities.LabelRow,
    lr_path: Path,
):
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.write_text(json.dumps(lr, indent=2), encoding="utf-8")

    dst_dir_path = lr_path.parent / "images"
    dst_dir_path.mkdir(exist_ok=True)
    img_file = Path(image.file_name)

    data_type = lr["data_units"][data_hash]["data_type"]
    new_suffix = f".{data_type.split('/')[1]}"
    copyfile(
        image_path / img_file,
        dst_dir_path / f"{data_hash}{new_suffix}",
    )


class CocoImporter:
    def __init__(self, images_dir_path: Path, annotations_file_path: Path, ssh_key_path: Path, destination_dir: Path):
        self.ssh_key_path = ssh_key_path
        self.images_dir = images_dir_path
        self.annotations_file_path = annotations_file_path

        if not self.images_dir.is_dir():
            raise NotADirectoryError(f"Images directory '{self.images_dir}' doesn't exist")
        if not self.annotations_file_path.is_file():
            raise FileNotFoundError(f"Annotation file '{self.annotations_file_path}' doesn't exist")

        self.user_client = EncordUserClient.create_with_ssh_private_key(ssh_key_path.read_text())

        annotations_file = json.loads(self.annotations_file_path.read_text(encoding="utf-8"))
        self.info = parse_info(annotations_file["info"])
        self.categories = parse_categories(annotations_file["categories"])
        self.images = parse_images(annotations_file["images"])
        self.annotations = parse_annotations(annotations_file["annotations"])

        title = self.info.description.replace(" ", "-")
        with_prefix = f"[EA]{title}"
        existing_projects = {o["project"].title: o["project"] for o in self.user_client.get_projects()}

        while with_prefix in existing_projects:
            title = typer.prompt(
                f"We found a project named '{title}', please provide a new name for this project", type=str
            )
            with_prefix = f"[EA]{title}"

        self.title = with_prefix
        self.project_dir = destination_dir / self.title

    def create_dataset(self) -> Dataset:
        print(f"Creating a new dataset: {self.title}")
        dataset_response: CreateDatasetResponse = self.user_client.create_dataset(
            self.title, StorageLocation.CORD_STORAGE
        )
        dataset: Dataset = self.user_client.get_dataset(dataset_response["dataset_hash"])

        for _, coco_image in tqdm(self.images.items(), desc="Uploading images"):
            upload_img(dataset, coco_image, self.images_dir)

        return dataset

    def create_ontology(self) -> Ontology:
        print(f"Creating a new dataset: {self.title}")
        ontology_structure = OntologyStructure()
        for cat in self.categories:
            ontology_structure.add_object(name=cat.name, shape=Shape.POLYGON, uid=cat.id_)

        ontology: Ontology = self.user_client.create_ontology(
            self.title,
            description=self.info.description,
            structure=ontology_structure,
        )

        return ontology

    def create_project(self, dataset: Dataset, ontology: Ontology) -> Project:
        print(f"Creating a new project: {self.title}")
        project_hash = self.user_client.create_project(
            project_title=self.title,
            dataset_hashes=[dataset.dataset_hash],
            ontology_hash=ontology.ontology_hash,
        )

        project = self.user_client.get_project(project_hash)

        self.project_dir.mkdir(exist_ok=True)

        ontology_file = self.project_dir / "ontology.json"
        ontology_file.write_text(json.dumps(project.ontology))

        project_meta = {
            "project_title": project.title,
            "project_description": project.description,
            "project_hash": project.project_hash,
            "ssh_key_path": self.ssh_key_path.as_posix(),
        }
        meta_file_path = self.project_dir / "project_meta.yaml"
        meta_file_path.write_text(yaml.dump(project_meta), encoding="utf-8")

        id_to_obj = {obj.uid: obj for obj in ontology.structure.objects}

        image_to_du = {}

        for data_unit in tqdm(dataset.data_rows, desc="Uploading annotations"):
            data_hash, lr = upload_annotation(project, self.annotations, data_unit, id_to_obj)
            image_id = lr["data_title"]
            image = self.images[image_id]
            image_to_du[image_id] = {"data_hash": data_hash, "height": image.height, "width": image.width}

            lr_path = self.project_dir / "data" / lr["label_hash"] / "label_row.json"
            build_local_lr(image, self.images_dir, data_hash, lr, lr_path)

        (Path(self.project_dir) / IMAGE_DATA_UNIT_FILENAME).write_text(json.dumps(image_to_du))

        return project
