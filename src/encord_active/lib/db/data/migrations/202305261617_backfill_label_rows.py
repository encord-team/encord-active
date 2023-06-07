from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from PIL import Image
from prisma.types import DataUnitCreateInput, LabelRowCreateInput
from tqdm.auto import tqdm

from encord_active.lib.common.data_utils import get_frames_per_second
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project.project_file_structure import (
    DataUnitStructure,
    ProjectFileStructure,
)


def up(pfs: ProjectFileStructure):
    # Adds the content missing from the data units and label rows tables when projects with
    # older versions of Encord Active are handled with versions greater than 0.1.52.
    label_row_meta = json.loads(pfs.label_row_meta.read_text(encoding="utf-8"))
    with PrismaConnection(pfs) as conn:
        fill_label_rows = conn.labelrow.count() == 0
        fill_data_units = conn.dataunit.count() == 0
        old_style_data = conn.labelrow.find_first(where={"label_row_json": None}) is not None or fill_label_rows
        if not (fill_label_rows or fill_data_units or old_style_data):
            return
        migration_counter = tqdm(
            None,
            desc=f"Migrating Database: {pfs.project_dir}",
        )
        with conn.batch_() as batcher:
            migration_counter.update(1)
            for label_row in pfs.iter_labels():
                migration_counter.display(
                    f"Migrating label_row: {label_row.label_hash}: (fill={fill_label_rows}, update={old_style_data},"
                    f" add_data={fill_data_units})"
                )
                try:
                    label_row_dict = label_row.label_row_json
                except ValueError:
                    legacy_label_row_file = label_row.label_row_file_deprecated_for_migration().read_text("utf-8")
                    label_row_dict = json.loads(legacy_label_row_file)

                label_hash = label_row_dict["label_hash"]
                lr_data_hash = label_row_meta[label_hash]["data_hash"]
                data_type = label_row_dict["data_type"]

                if fill_label_rows:
                    batcher.labelrow.create(
                        LabelRowCreateInput(
                            label_hash=label_hash,
                            data_hash=lr_data_hash,
                            data_title=label_row_dict["data_title"],
                            data_type=data_type,
                            created_at=label_row_meta[label_hash].get("created_at", datetime.now()),
                            last_edited_at=label_row_meta[label_hash].get("last_edited_at", datetime.now()),
                            label_row_json=json.dumps(label_row_dict),
                        )
                    )
                elif old_style_data:
                    batcher.labelrow.update(
                        data={
                            "label_row_json": json.dumps(label_row_dict),
                        },
                        where={
                            "label_hash": label_hash,
                        },
                    )

                if fill_data_units or old_style_data:
                    data_units = label_row_dict["data_units"]
                    legacy_lr_path = label_row.label_row_file_deprecated_for_migration().parent / "images"
                    db_data_unit = list(label_row.iter_data_unit())
                    if len(db_data_unit) == 0:
                        # For migrating fully empty db condition
                        type_hack_optional_int_none: Optional[int] = None
                        db_data_unit = [
                            DataUnitStructure(
                                label_hash=label_hash,
                                du_hash=du["data_hash"],
                                frame=frame,
                                data_type=data_type,
                                # Not queried and hence can be left empty
                                signed_url="null",
                                data_hash_raw="null",
                                width=-1,
                                height=-1,
                                frames_per_second=-1,
                            )
                            for du in data_units.values()
                            for frame in (
                                [type_hack_optional_int_none]
                                if du["data_type"] != "video"
                                else [i for i, pth in enumerate(legacy_lr_path.glob(f"{du['data_hash']}_*"))]
                            )
                        ]

                    for data_unit in db_data_unit:
                        migration_counter.update(1)
                        migration_counter.display(
                            f"Migrating data_hash: {label_row.label_hash} / {data_unit.du_hash} / {data_unit.frame}"
                        )
                        if data_unit.frame is not None and data_unit.frame != -1 and data_type == "video":
                            legacy_du_path = next(legacy_lr_path.glob(f"{data_unit.du_hash}_{data_unit.frame}.*"))
                        else:
                            legacy_du_path = next(legacy_lr_path.glob(f"{data_unit.du_hash}.*"))

                        du = data_units[data_unit.du_hash]
                        frames_per_second = 0.0
                        if data_type == "video":
                            if "_" not in legacy_du_path.stem:
                                frame_str = "-1"  # To include a reference to the video location in the DataUnit table
                            else:
                                _, frame_str = legacy_du_path.stem.rsplit("_", 1)
                            # Lookup frames per second
                            legacy_du_video_path = next(legacy_lr_path.glob(f"{data_unit.du_hash}.*"))
                            frames_per_second = get_frames_per_second(legacy_du_video_path)
                            data_uri_path = legacy_du_video_path
                        else:
                            frame_str = du.get("data_sequence", 0)
                            data_uri_path = legacy_du_path
                        frame = int(frame_str)

                        if frame != -1 or data_type != "video":
                            image = Image.open(legacy_du_path)
                        else:
                            legacy_du_any_frame_path = next(legacy_lr_path.glob(f"{data_unit.du_hash}_*"))
                            image = Image.open(legacy_du_any_frame_path)

                        if fill_data_units:
                            batcher.dataunit.create(
                                DataUnitCreateInput(
                                    data_hash=data_unit.du_hash,
                                    data_title=du["data_title"],
                                    frame=frame,
                                    lr_data_hash=lr_data_hash,
                                    data_uri=data_uri_path.absolute().as_uri(),
                                    width=image.width,
                                    height=image.height,
                                    fps=frames_per_second,
                                )
                            )
                        else:
                            batcher.dataunit.update(
                                data={
                                    "data_uri": data_uri_path.absolute().as_uri(),
                                    "width": image.width,
                                    "height": image.height,
                                    "fps": frames_per_second,
                                },
                                where={
                                    "data_hash_frame": {
                                        "data_hash": data_unit.du_hash,
                                        "frame": frame,
                                    },
                                },
                            )
            batcher.commit()
