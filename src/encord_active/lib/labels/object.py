from enum import Enum


# copy from encord but as a string enum
class ObjectShape(str, Enum):
    POLYGON = "polygon"
    POLYLINE = "polyline"
    BOUNDING_BOX = "bounding_box"
    KEY_POINT = "point"
    SKELETON = "skeleton"
    ROTATABLE_BOUNDING_BOX = "rotatable_bounding_box"


SimpleShapes = {ObjectShape.POLYGON, ObjectShape.POLYLINE, ObjectShape.KEY_POINT, ObjectShape.SKELETON}
BoxShapes = {ObjectShape.BOUNDING_BOX, ObjectShape.ROTATABLE_BOUNDING_BOX}
