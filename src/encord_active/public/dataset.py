from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from encord.objects import OntologyStructure
from encord.objects.attributes import RadioAttribute
from PIL import Image
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, select
from torch.utils.data import Dataset

from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectTag,
    ProjectTaggedDataUnit,
    get_engine,
)
from encord_active.lib.common.data_utils import url_to_file_path

P = Project
T = ProjectTaggedDataUnit
D = ProjectDataUnitMetadata
L = ProjectDataMetadata


class ActiveDataset(Dataset):
    def __init__(
        self,
        database_path: Path,
        project_hash: Union[set[str], str],
        tag_name: Optional[Union[set[str], str]] = None,
        ontology_hashes: Optional[list[str]] = None,
        transform=None,
        target_transform=None,
    ):
        database_path = database_path.expanduser().resolve()
        if not database_path.is_file():
            raise ValueError(f"DB doesn't exist `database_path`")
        self.root_path = database_path.parent
        self.engine = get_engine(database_path, use_alembic=False)
        self.project_hash = {UUID(project_hash)} if isinstance(project_hash, str) else set(map(UUID, project_hash))
        self.tag_name = {tag_name} if isinstance(tag_name, str) else tag_name
        self.ontology_hashes = ontology_hashes

        self.identifiers: list[tuple[UUID, int]] = []

        self.setup()

        self.transform = transform
        self.target_transform = target_transform

    def get_identifier_query(self, sess: Session):
        identifier_query = select(D.du_hash, D.frame)
        if self.tag_name is not None:
            in_uids = set()
            in_names = set()
            for name in self.tag_name:
                try:
                    in_uids.add(UUID(name))
                except ValueError:
                    in_names.add(name)

            where_clauses = [
                in_op(col, vals) for col, vals in [(ProjectTag.tag_hash, in_uids), (ProjectTag.name, in_names)] if vals
            ]
            tag_query = select(ProjectTag.tag_hash).where(
                in_op(ProjectTag.project_hash, self.project_hash), *where_clauses
            )
            tag_hash = set(sess.exec(tag_query).all())

            if tag_hash is None:
                valid_tag_names = sess.exec(
                    select(ProjectTag.name).where(in_op(ProjectTag.project_hash, self.project_hash))
                ).all()
                raise ValueError(
                    f"Couldn't find a data tag with either name or tag_hash `{self.tag_name}`. Valid tags for the specified project(s) are {valid_tag_names}."
                )

            identifier_query = identifier_query.join(
                T, onclause=((T.du_hash == D.du_hash) & (T.frame == D.frame))
            ).where(
                in_op(D.project_hash, self.project_hash),
                in_op(T.project_hash, self.project_hash),
                in_op(T.tag_hash, tag_hash),
            )
        else:
            identifier_query = identifier_query.where(in_op(D.project_hash, self.project_hash))

        return identifier_query

    def setup(self):
        with Session(self.engine) as sess:
            # Check that data is available locally
            identifier_query = self.get_identifier_query(sess)
            probe = sess.exec(identifier_query.add_columns(D.data_uri).limit(1)).first()
            if (
                probe is not None
                and probe[-1] is not None
                and url_to_file_path(probe[-1], self.root_path) is None  # type: ignore
            ):
                raise ValueError("Couldn't find data locally. Please execute `encord-active download-data` first.")

            # Check for videos
            video_probe = sess.exec(identifier_query.where(D.data_uri_is_video).limit(1)).first()
            if video_probe is not None:
                raise ValueError("Dataset contains videos. This is currently not supported for this dataloader.")

            # Load and validate ontology
            if len(self.project_hash) > 1:
                ontologies = list(map(OntologyStructure.from_dict, sess.exec(select(P.project_ontology)).all()))  # type: ignore
                first, *rest = [
                    tuple(
                        [o.feature_node_hash for o in ont.objects] + [c.feature_node_hash for c in ont.classifications]
                    )
                    for ont in ontologies
                ]
                assert all(
                    [first == next_ for next_ in rest]
                ), "Ontologies must match if you select multiple projects at once"

            ontology_dict = sess.exec(
                select(P.project_ontology).where(in_op(P.project_hash, self.project_hash)).limit(1)
            ).first()
            if ontology_dict is None:
                raise ValueError("Couldn't read project ontology")
            self.ontology = OntologyStructure.from_dict(ontology_dict)  # type: ignore

    def __len__(self):
        with Session(self.engine) as sess:
            return sess.query(self.get_identifier_query(sess)).count()

    def __getitem__(self, idx):
        ...


class ActiveClassificationDataset(ActiveDataset):
    def __init__(
        self,
        database_path: Path,
        project_hash: Union[set[str], str],
        tag_name: Optional[Union[set[str], str]] = None,
        ontology_hashes: Optional[list[str]] = None,
        transform=None,
        target_transform=None,
    ):
        """
        A dataset hooked up to an Encord Active database.
        The dataset can filter (image) data from Encord Active based on both `project_hash`es
        and `tag_hash`/`tag_name`s. For example, if you have added a tag called
        "train" within Encord Active, you can use that tag name by setting
        `tag_name="train"` option. You can also combine multiple tags into one
        dataset by providing a set of names. Similarly, you can use multiple
        projects if they share the same ontology. Just provide the relevant
        project hashes.

        Note: that this dataset requires that you have downloaded the data locally.
        This can be done with `encord-active project download-data`.

        ⚠️  Videos are not yet supported.

        Args:
            database_path: Path to where the `encord_active.sqlite` database lives
                on your system.
            project_hash: The project hash (or set of hashes) of the project(s)
                to load data from.
            tag_name: tag names (or hashes) for the tags you want to include.
                If no tags are specified, all images with labels from the project
                will be included.
            ontology_hashes: The `feature_node_hash` of the radio button classification
                question used for labels. The first radiobutton within that
                classification will be used.
            transform (): Data transform applied to PIL images
            target_transform (): Target transform applied to the uint label tensor.
        """
        assert (
            ontology_hashes is None or len(ontology_hashes) == 1
        ), "Either don't define ontology hashes to use first radio button in ontology or specify the feature node hash of the classification you want."

        super().__init__(
            database_path,
            project_hash,
            tag_name,
            ontology_hashes,
            transform,
            target_transform,
        )

    def setup(self):
        super().setup()

        with Session(self.engine) as sess:
            identifier_query = self.get_identifier_query(sess)
            identifier_query = identifier_query.join(L, onclause=(L.data_hash == D.data_hash)).where(
                in_op(L.project_hash, self.project_hash)
            )
            identifier_query = identifier_query.add_columns(D.data_uri, D.classifications, L.label_row_json)

            identifiers = sess.exec(identifier_query).all()
            ontology_pairs = [
                (c, a)
                for c in self.ontology.classifications
                for a in c.attributes
                if isinstance(a, RadioAttribute)
                and ((not self.ontology_hashes) or c.feature_node_hash in self.ontology_hashes)
            ]
            if len(ontology_pairs) == 0:
                raise ValueError(f"Found no ontology items to use for labels")
            classification, attribute = ontology_pairs[0]
            indices = {o.feature_node_hash: i for i, o in enumerate(attribute.options)}
            self.class_names = [o.title for o in attribute.options]

            self.uris = []
            self.labels = []
            for (  # type: ignore
                *_,
                data_uri,
                classifications,
                label_row_json,
            ) in identifiers:
                classification_answers = label_row_json["classification_answers"]

                clf_instance = next(
                    (c for c in classifications if c["featureHash"] == classification.feature_node_hash),
                    None,
                )
                if clf_instance is None:
                    continue
                clf_hash = clf_instance["classificationHash"]
                clf_classifications = classification_answers[clf_hash]["classifications"]
                clf_answers = next(
                    (a for a in clf_classifications if a["featureHash"] == attribute.feature_node_hash),
                    None,
                )
                if clf_answers is None:
                    continue
                clf_opt = next(
                    (o for o in clf_answers["answers"] if o["featureHash"] in indices),
                    None,
                )
                if clf_opt is None:
                    continue

                self.uris.append(url_to_file_path(data_uri, self.root_path))  # type: ignore
                self.labels.append(indices[clf_opt["featureHash"]])

    def __getitem__(self, idx):
        data_uri = self.uris[idx]

        img = Image.open(data_uri)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.labels)
