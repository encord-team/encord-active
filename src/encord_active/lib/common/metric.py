from abc import ABC, abstractmethod
from enum import Enum
from hashlib import md5
from typing import List, Optional, Union

from encord.project_ontology.classification_type import ClassificationType
from encord.project_ontology.object_type import ObjectShape

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.writer import CSVMetricWriter


class MetricType(Enum):
    SEMANTIC = "semantic"
    GEOMETRIC = "geometric"
    HEURISTIC = "heuristic"


class DataType(Enum):
    IMAGE = "image"
    SEQUENCE = "sequence"


class AnnotationType:
    NONE = None
    OBJECT = ObjectShape
    CLASSIFICATION = ClassificationType
    ALL = [OBJECT, CLASSIFICATION]


class EmbeddingType(Enum):
    CLASSIFICATION = "classification"
    OBJECT = "object"
    HU_MOMENTS = "hu_moments"
    NONE = "none"


class Metric(ABC):
    @abstractmethod
    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        """
        This is where you should perform your data indexing.

        :param iterator: The iterator with which you can iterate through the dataset as
            many times you like. The iterator scans through one label rows at a time,
            continuously indexing each frame of a video.

            Use::

                for data_unit, img_pth in iterator.iterate(desc="Progress bar description"):
                    pass


        :param writer:  The writer where you should store your scores.
            Use::

                writer.write(score, objects)


        :return:
        """
        pass

    @property
    @abstractmethod
    def TITLE(self) -> str:
        """
        Set the title of your metric. Aim for no more than 80 characters.
        The title will be used, e.g., when displaying results in the app.

        The property can, e.g., be set as a class property::

            class MyClass(Metric):
                TITLE = "This is the title"
                # ...

        """
        pass

    @property
    @abstractmethod
    def METRIC_TYPE(self) -> MetricType:
        """
        Type of the metric. Choose one of the following:
            - Geometric: metrics related to the geometric properties of annotations.
            This type includes size, shape, location, etc.
            - Semantic: metrics based on the *contents* of annotations, images or videos.
            This type includes ResNet embedding distances, image uncertainties, etc.
            - Heuristic: any other metric. For example, brightness, sharpness, object counts, etc.


        The property can, e.g., be set as a class property::

            class MyClass(Metric):
                METRIC_TYPE = MetricType.GEOMETRIC
                # ...

        """
        pass

    @property
    @abstractmethod
    def DATA_TYPE(self) -> Union[List[DataType], DataType]:
        """
        Type of data with which the metric operates. Choose one of the following:
            - Image: metric requires individual images. E.g. bounding box location
            - Sequence: metric requires sequences of images to harness temporal information
                (videos, e.g. bounding box IoU across video frames) or volumetric information
                (DICOM, e.g. bounding box IoU across slices).

        The property can, e.g., be set as a class property::

            class MyClass(Metric):
                DATA_TYPE = DataType.IMAGE
                # ...

        """
        pass

    @property
    @abstractmethod
    def ANNOTATION_TYPE(self) -> Optional[Union[List[AnnotationType], AnnotationType]]:
        """
        Type of annotations the metric operates needs. Choose one of the following:
            - Object: includes bounding box, polygon, polyline, keypoint and skeleton
            - Classification: includes text, radio button and checklist

        The property can, e.g., be set as a class property::

            class MyClass(Metric):
                ANNOTATION_TYPE = AnnotationType.OBJECT.POLYGON
                # ...

        """
        pass

    @property
    @abstractmethod
    def SHORT_DESCRIPTION(self) -> str:
        """
        Set a short description of what the metric does. Aim for at most 120
        characters. As for the title, this can be set as a class property.

        The short description will be used in lists and other places, where space is
        limited.
        """
        pass

    @property
    @abstractmethod
    def LONG_DESCRIPTION(self) -> str:
        """
        Set a verbose description of what the metric does. If it is based on ideas
        from papers, etc., this is where such links can be included. If you write
        markdown, this will be rendered appropriately in the app.

        The long description will be used when much space is available and when
        providing full details makes sense.
        """
        pass

    @property
    def SCORE_NORMALIZATION(self) -> bool:
        """
        If the score normalization will be applied to metrics from this metric as default in the app.
        :return: True or False
        """

        return False

    @property
    def NEEDS_IMAGES(self):
        """
        If the metric need to look at image content. This is automatically inferred from the metric type:
            - MetricType.GEOMETRIC: False
            - MetricType.SEMANTIC: True
            - MetricType.HEURISTIC: True

        """
        return False if self.METRIC_TYPE == MetricType.GEOMETRIC else True

    def get_unique_name(self):
        name_hash = md5((self.TITLE + self.SHORT_DESCRIPTION + self.LONG_DESCRIPTION).encode()).hexdigest()

        return f"{name_hash[:8]}_{self.TITLE.lower().replace(' ', '_')}"
