import Icon, {
  AppstoreAddOutlined,
  PartitionOutlined,
} from "@ant-design/icons";
import {
  BiPolygon,
  BiShapeSquare,
  BiShapeTriangle,
  BiShareAlt,
  BiTargetLock,
} from "react-icons/all";
import { AnnotationType } from "../../openapi/api";

export function AnnotationShapeIcon(props: {
  shape: AnnotationType;
  color: string;
}) {
  const { shape, color } = props;
  const className = "align-middle";
  const size = 2;
  if (shape === 0) {
    // classification
    return (
      <PartitionOutlined style={{ color }} className={className} size={size} />
    );
  } else if (shape === 1) {
    // bounding box
    return (
      <Icon
        style={{ color }}
        component={BiShapeSquare}
        className={className}
        size={size}
      />
    );
  } else if (shape === 2) {
    // rot bounding box
    return (
      <Icon
        style={{ color }}
        rotate={45}
        component={BiShapeSquare}
        className={className}
        size={size}
      />
    );
  } else if (shape === 3) {
    // point
    return (
      <Icon
        style={{ color }}
        component={BiTargetLock}
        className={className}
        size={size}
      />
    );
  } else if (shape === 4) {
    // polyline
    return (
      <Icon
        style={{ color }}
        rotate={90}
        component={BiShareAlt}
        className={className}
        size={size}
      />
    );
  } else if (shape === 5) {
    // polygon
    return (
      <Icon
        style={{ color }}
        component={BiShapeTriangle}
        className={className}
        size={size}
      />
    );
  } else if (shape === 6) {
    // skeleton
    return (
      <Icon
        style={{ color }}
        component={BiPolygon}
        className={className}
        size={size}
      />
    );
  } else if (shape === 7) {
    // bitmask
    return (
      <AppstoreAddOutlined
        rotate={270}
        style={{ color }}
        className={className}
        size={size}
      />
    );
  } else {
    // ERROR
    return null;
  }
}
