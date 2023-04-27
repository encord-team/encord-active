import json
import re

from prisma import models

from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project.metadata import fetch_project_meta

DATA_HASH_REGEX = r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})"


# To be deprecated when Encord Active version is >= 0.1.60.
def fill_missing_tables():
    with PrismaConnection() as conn:
        if conn.labelrow.count() == 0:
            fill_label_rows_table()
        if conn.dataunit.count() == 0:
            fill_data_units_table()


# To be deprecated when Encord Active version is >= 0.1.60.
def fill_data_units_table():
    # Adds the content missing from the data units table when projects with
    # older versions of Encord Active are handled with versions greater than 0.1.52.
    project_file_structure = PrismaConnection.project_file_structure()

    pattern = re.compile(DATA_HASH_REGEX)
    with PrismaConnection() as conn:
        with conn.batch_() as batcher:
            # fetch data units from local storage and store their references in the prisma db
            for label_hash in project_file_structure.data.iterdir():
                if pattern.match(label_hash.name) is None:  # avoid unexpected folders in the data directory
                    continue
                label_row_path = project_file_structure.data / label_hash / "label_row.json"
                label_row = json.loads(label_row_path.read_text(encoding="utf-8"))
                images_dir = project_file_structure.data / label_hash / "images"

                for data_unit in label_row["data_units"].values():
                    du_hash = data_unit["data_hash"]
                    du_title = data_unit["data_title"]

                    if label_row["data_type"] == "video":
                        for du_frame_path in images_dir.glob(f"{du_hash}_*"):
                            du_frame = int(du_frame_path.stem.rsplit("_", maxsplit=1)[-1])
                            batcher.dataunit.create(
                                models.DataUnit(
                                    id=-1,  # add dummy id value because `id` is a required attribute
                                    data_hash=du_hash,
                                    data_title=du_title,
                                    frame=du_frame,
                                    location=du_frame_path.resolve().as_posix(),
                                    lr_data_hash=label_row["data_hash"],
                                ).dict(exclude={"id", "label_row"})
                            )
                    else:
                        du_frame = int(data_unit["data_sequence"])
                        du_frame_path = next(images_dir.glob(f"{du_hash}.*"), None)
                        if du_frame_path is not None:
                            batcher.dataunit.create(
                                models.DataUnit(
                                    id=-1,  # add dummy id value because `id` is a required attribute
                                    data_hash=du_hash,
                                    data_title=du_title,
                                    frame=du_frame,
                                    location=du_frame_path.resolve().as_posix(),
                                    lr_data_hash=label_row["data_hash"],
                                ).dict(exclude={"id", "label_row"})
                            )
            batcher.commit()


# To be deprecated when Encord Active version is >= 0.1.60.
def fill_label_rows_table():
    # Adds the content missing from the label rows table when projects with
    # older versions of Encord Active are handled with versions greater than 0.1.52.
    project_file_structure = PrismaConnection.project_file_structure()
    project_meta = fetch_project_meta(PrismaConnection.project_file_structure().project_dir)

    pattern = re.compile(DATA_HASH_REGEX)
    with PrismaConnection() as conn:
        with conn.batch_() as batcher:
            # fetch label rows from local storage and store their references in the prisma db
            for label_hash in project_file_structure.data.iterdir():
                if pattern.match(label_hash.name) is None:  # avoid unexpected folders in the data directory
                    continue
                label_row_path = project_file_structure.data / label_hash / "label_row.json"
                label_row = json.loads(label_row_path.read_text(encoding="utf-8"))

                label_hash = label_row["label_hash"]
                # data_hash = label_row["data_hash"] # don't supported in previous sdk version
                data_hash = project_meta[label_hash]["data_hash"]
                data_title = label_row["data_title"]
                data_type = label_row["data_type"]
                created_at = label_row["created_at"]
                last_edited_at = label_row["last_edited_at"]

                batcher.labelrow.create(
                    models.LabelRow(
                        id=-1,  # add dummy id value because `id` is a required attribute
                        label_hash=label_hash,
                        data_hash=data_hash,
                        data_title=data_title,
                        data_type=data_type,
                        created_at=created_at,
                        last_edited_at=last_edited_at,
                        location=label_row_path.resolve().as_posix(),
                    ).dict(
                        exclude={"id", "data_units"}  # remove dummy values
                    )
                )
            batcher.commit()
