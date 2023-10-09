import json
import re
import tempfile
from abc import abstractmethod
from collections.abc import Sized
from copy import deepcopy
from dataclasses import asdict
from functools import reduce
from itertools import chain
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from encord.exceptions import EncordException
from encord.orm.label_log import LabelLog
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm

from encord_active.lib.common.data_utils import (
    download_file,
    download_image,
    extract_frames,
)
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project import Project, ProjectFileStructure


class Iterator(Sized):
    def __init__(self, cache_dir: Path, subset_size: Optional[int] = None, **kwargs):
        self.cache_dir = cache_dir
        self.subset_size = subset_size
        self.dataset_title = ""
        self.label_hash = ""
        self.du_hash = ""
        self.frame = -1
        self.num_frames = -1
        self.project: Project
        self.project_file_structure = ProjectFileStructure(cache_dir)
        if "project" in kwargs and kwargs["project"]:
            self.project = Project(cache_dir).from_encord_project(kwargs["project"])
        else:
            self.project = Project(cache_dir).load(subset_size)
        self.label_rows = self.project.label_rows

    @abstractmethod
    def iterate(self, desc: str = "") -> Generator[Tuple[dict, Optional[Image.Image]], None, None]:
        pass

    @abstractmethod
    def get_identifier(self, object: Union[dict, list[dict], None] = None, frame: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def get_data_url(self) -> str:
        pass

    @abstractmethod
    def get_label_logs(self, object_hash: Optional[str] = None, refresh: bool = False) -> List[dict]:
        pass

    @staticmethod
    @abstractmethod
    def update_cache_dir(cache_dir: Path) -> Path:
        pass


class DatasetIterator(Iterator):
    def __init__(self, cache_dir: Path, subset_size: Optional[int] = None, skip_labeled_data: bool = False, **kwargs):
        super().__init__(cache_dir, subset_size, **kwargs)
        self.key = ""
        self._skip_labeled_data = skip_labeled_data
        self.length = reduce(
            lambda s, lr: s
            + sum(map(lambda du: 1 if "objects" in du["labels"] else len(du["labels"]), lr["data_units"].values())),
            self.label_rows.values(),
            0,
        )

    def iterate(self, desc: str = "") -> Generator[Tuple[dict, Optional[Image.Image]], None, None]:
        with PrismaConnection(self.project_file_structure) as cache_db:
            pbar = tqdm(total=self.length, desc=desc, leave=False)
            for label_hash, label_row in self.label_rows.items():
                self.dataset_title = label_row["dataset_title"]
                self.label_hash = label_hash
                if label_row.data_type in {"img_group", "image"}:
                    self.num_frames = len(label_row.data_units)
                    data_units = sorted(label_row.data_units.values(), key=lambda du: int(du["data_sequence"]))
                    for data_unit in data_units:

                        if self._skip_labeled_data:
                            du_label = data_unit.get("labels", {})
                            if du_label.get("objects", []) != [] or du_label.get("classifications", []) != []:
                                pbar.update(1)
                                continue

                        self.du_hash = data_unit["data_hash"]
                        self.frame = int(data_unit["data_sequence"])
                        try:
                            img_metadata = next(
                                self.project_file_structure.label_row_structure(label_hash).iter_data_unit(
                                    self.du_hash, cache_db=cache_db
                                ),
                                None,
                            )
                            image = None
                            if img_metadata is not None:
                                image = download_image(
                                    img_metadata.signed_url,
                                    project_dir=self.project_file_structure.project_dir,
                                )
                            yield data_unit, image
                        except KeyError:
                            logger.error(
                                f"There was an issue finding the path for label row: `{label_hash}` and data unit: `{self.du_hash}`"
                            )
                            continue
                        pbar.update(1)
                elif label_row.data_type == "video":
                    data_unit, *_ = label_row["data_units"].values()
                    self.du_hash = data_unit["data_hash"]
                    video_metadata = next(
                        self.project_file_structure.label_row_structure(label_hash).iter_data_unit(
                            self.du_hash, cache_db=cache_db
                        ),
                        None,
                    )
                    if video_metadata is None:
                        continue  # SKIP

                    # Create temporary folder containing the video
                    with tempfile.TemporaryDirectory() as working_dir:
                        working_path = Path(working_dir)
                        safe_data_title = re.sub(r'[\\/:*?"<>|\x00-\x1F\x7F ]', "_", data_unit["data_title"])
                        video_path = working_path / safe_data_title
                        video_images_dir = working_path / "images"
                        download_file(
                            video_metadata.signed_url,
                            project_dir=self.project_file_structure.project_dir,
                            destination=video_path,
                        )
                        extract_frames(video_path, video_images_dir, self.du_hash, data_unit["data_fps"])

                        fake_data_unit = deepcopy(data_unit)

                        for frame_id in range(len(list(video_images_dir.glob("*_*.*")))):
                            self.frame = frame_id
                            fake_data_unit["labels"] = data_unit["labels"].get(str(frame_id), {})

                            if self._skip_labeled_data:
                                if (
                                    fake_data_unit["labels"].get("objects", []) != []
                                    or fake_data_unit["labels"].get("classifications", []) != []
                                ):
                                    pbar.update(1)
                                    continue

                            image_path = next(video_images_dir.glob(f"{self.du_hash}_{frame_id}.*"), None)
                            if image_path:
                                yield fake_data_unit, Image.open(image_path)
                            else:
                                yield fake_data_unit, None
                            pbar.update(1)

                else:
                    logger.error(f"Label row '{label_hash}' with data type '{label_row.data_type}' is not recognized")

    def __len__(self):
        return self.length

    def get_identifier(self, object: Union[dict, list[dict], None] = None, frame: Optional[int] = None):
        frame_idx = frame if frame is not None else self.frame
        key = f"{self.label_hash}_{self.du_hash}_{frame_idx:05d}"

        if object is not None:
            if isinstance(object, dict):
                objects = [object]
            else:
                objects = object  # object is expected to be a list[dict]
            hashes = [obj["objectHash"] if "objectHash" in obj else obj["featureHash"] for obj in objects]
            return "_".join(chain([key], hashes))
        return key

    def get_data_url(self) -> str:
        label_row_meta = self.project.label_row_metas.get(self.label_hash, None)
        data_hash_meta = label_row_meta.data_hash if label_row_meta is not None else self.du_hash
        base_url = "https://app.encord.com/label_editor/"
        data_url = f"{base_url}{data_hash_meta}&{self.project.project_hash}"
        if isinstance(self.frame, int):
            data_url += f"/{self.frame}"
        return data_url

    @staticmethod
    def __filter_logs_on_object_hash(logs: List[dict], object_hash: Optional[str] = None) -> List[dict]:
        if object_hash is None:
            return logs

        return list(filter(lambda x: x["annotation_hash"] == object_hash, logs))

    def get_label_logs(self, object_hash: Optional[str] = None, refresh: bool = False) -> List[dict]:
        """
        Fetches label logs from the sdk if they are not cached locally. The `refresh=True` will fetch a new version
        of the label logs.

        :param object_hash: If specified, only label logs associated with the given object will be returned.
        :param refresh: flag to  fetch the most recent label logs instead of using the cached version.
        :return: A list of label logs for the
        """
        if not hasattr(self, "du_hash"):
            raise ValueError(
                "Label logs are only available when you are iterating data.\n"
                "Consider using `iterator.project.get_label_logs()` if you want all (non-cached) label logs for the "
                "project."
            )

        no_logs_file = self.cache_dir / "label_logs" / self.label_hash / "no_logs.txt"
        if (no_logs_file).exists():
            return []

        label_logs_file = self.cache_dir / "label_logs" / self.label_hash / f"{self.frame:05d}.json"
        if label_logs_file.parent.exists() and not refresh:
            if label_logs_file.exists():
                try:
                    with label_logs_file.open("r", encoding="utf-8") as f:
                        json_logs: List[dict] = json.load(f)["logs"]
                        return self.__filter_logs_on_object_hash(json_logs, object_hash=object_hash)
                except JSONDecodeError:
                    pass
            return []

        # Load label logs and store them according to frames.
        data_hash = self.project.label_row_metas[self.label_hash].data_hash
        try:
            # logs = self.project.get_label_logs(data_hash=data_hash)
            # TODO commented while we figure out how to address this from cache only
            logs: List[LabelLog] = []
            return logs
        except EncordException as e:
            no_logs_file.parent.mkdir(parents=True, exist_ok=True)
            no_logs_file.touch(exist_ok=True)
            return []

        frame_logs: Dict[int, List[LabelLog]] = {}
        for log in logs:
            frame_logs.setdefault(log.frame, []).append(log)

        out_logs: List[dict] = []
        for frame, tmp_logs in frame_logs.items():
            frame_name = f"{frame:05d}.json" if frame is not None else "No-frame.json"
            tmp_log_file = self.cache_dir / "label_logs" / self.label_hash / frame_name
            tmp_log_file.parent.mkdir(parents=True, exist_ok=True)

            json_logs = list(map(asdict, tmp_logs))
            with tmp_log_file.open("w") as f:
                json.dump({"logs": json_logs}, f)

            if frame == self.frame:
                out_logs = json_logs

        return self.__filter_logs_on_object_hash(out_logs, object_hash=object_hash)

    @staticmethod
    def update_cache_dir(cache_dir: Path) -> Path:
        # Just use the root
        return cache_dir
