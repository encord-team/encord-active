import json
import re

from prisma.models import DataUnit

from encord_active.lib.db.connection import PrismaConnection

DATA_HASH_REGEX = r"([0-9a-f]{8})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{4})-([0-9a-f]{12})"


# To be deprecated when Encord Active version is >= 0.1.60.
def fill_data_units_table():
    # Adds the content missing from the data units table when projects with
    # older versions of Encord Active use versions greater than 0.1.52.
    # begin enforce backwards compatibility (read paths from old filesystem storage)
    project_file_structure = PrismaConnection.project_file_structure()
    data_units = []

    pattern = re.compile(DATA_HASH_REGEX)
    # fetch data units from local storage
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
                    data_units.append(
                        DataUnit(
                            id=-1,  # add dummy id value because `id` is a required attribute
                            hash=du_hash,
                            group_hash=label_row["data_hash"],
                            location=du_frame_path.resolve().as_posix(),
                            title=du_title,
                            frame=du_frame,
                        )
                    )
            else:
                du_frame = int(data_unit["data_sequence"])
                du_frame_path = next(images_dir.glob(f"{du_hash}.*"), None)
                if du_frame_path is not None:
                    data_units.append(
                        DataUnit(
                            id=-1,  # add dummy id value because `id` is a required attribute
                            hash=du_hash,
                            group_hash=label_row["data_hash"],
                            location=du_frame_path.resolve().as_posix(),
                            title=du_title,
                            frame=du_frame,
                        )
                    )

    # store data unit references in the db
    with PrismaConnection() as conn:
        for data_unit in data_units:
            conn.dataunit.create(data_unit.dict(exclude={"id"}))  # remove dummy id value
