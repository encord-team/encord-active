import { QueryContext } from "../../hooks/Context";
import useResizeObserver from "use-resize-observer";
import { useRef } from "react";
import { ProjectItem } from "../../openapi/api";
import * as React from "react";
import { useImageSrc } from "../../hooks/useImageSrc";

export function AnnotatedImage(props: {
  queryContext: QueryContext;
  item: ProjectItem;
  annotationHash: string | undefined;
  className?: string | undefined;
}) {
  const { item, queryContext, className, annotationHash } = props;
  const {
    ref: image,
    width: imageWidth,
    height: imageHeight,
  } = useResizeObserver<HTMLImageElement>();
  const video = useRef<HTMLVideoElement>(null);
  const { width: videoWidth, height: videoHeight } =
    useResizeObserver<HTMLVideoElement>({
      ref: video,
    });
  const width = item.timestamp != null ? videoWidth : imageWidth;
  const height = item.timestamp != null ? videoHeight : imageHeight;
  const imageSrc = useImageSrc(queryContext, item.url);

  return (
    <figure
      className={"relative" + (className === undefined ? "" : " " + className)}
    >
      {item.timestamp != null ? (
        <video
          ref={video}
          className="rounded object-contain transition-opacity"
          src={imageSrc}
          muted
          controls={false}
          onLoadedMetadata={() => {
            const videoRef = video.current;
            if (videoRef != null) {
              videoRef.currentTime = item.timestamp || 0;
            }
          }}
        />
      ) : (
        <img
          ref={image}
          className="rounded object-contain transition-opacity"
          alt=""
          src={imageSrc}
        />
      )}
      {item.objects.length > 0 && width && height ? (
        <AnnotationRenderLayer
          objects={item.objects as AnnotationObject[]}
          width={width}
          height={height}
          annotationHash={annotationHash}
        />
      ) : null}
    </figure>
  );
}
type AnnotationObjectPolygon = {
  readonly shape: "polygon";
  readonly polygon: Record<string, { readonly x: number; readonly y: number }>;
};

type AnnotationObjectPoint = {
  readonly shape: "point";
  readonly point: {
    readonly "0": { readonly x: number; readonly y: number };
  };
};

type AnnotationObjectAABB = {
  readonly shape: "bounding_box";
  readonly bounding_box: {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
  };
};

type AnnotationObjectCommon = {
  readonly objectHash: string;
  readonly color: string;
  readonly confidence: number;
  readonly createdAt: string;
  readonly createdBy: string;
  readonly featureHash: string;
  readonly lastEditedAt: string;
  readonly lastEditedBy: string;
  readonly manualAnnotation: boolean;
  readonly name: string;
  readonly value: string;
};

type AnnotationObject = AnnotationObjectCommon &
  (AnnotationObjectPolygon | AnnotationObjectPoint | AnnotationObjectAABB);

function AnnotationRenderLayer(props: {
  objects: AnnotationObject[];
  width: number;
  height: number;
  annotationHash: string | undefined;
}) {
  const { objects, width, height, annotationHash } = props;
  console.log("render annotations", objects, annotationHash);

  const renderPolygon = (
    poly: AnnotationObjectCommon & AnnotationObjectPolygon,
    select: boolean
  ) => {
    return (
      <polygon
        key={poly.objectHash}
        style={{
          fillOpacity: select ? "40%" : "10%",
          strokeOpacity: select ? "100%" : "40%",
          stroke: poly.color,
          strokeWidth: select ? "2px" : "1px",
        }}
        points={Object.values(poly.polygon)
          .map(({ x, y }) => `${x * width},${y * height}`)
          .join(" ")}
      />
    );
  };

  const renderPoint = (
    poly: AnnotationObjectCommon & AnnotationObjectPoint,
    select: boolean
  ) => {
    const { x, y } = poly.point["0"];
    return (
      <g key={poly.objectHash}>
        <circle cx={x * width} cy={y * height} r="5px" fill={poly.color} />
        <circle
          cx={x * width}
          cy={y * height}
          r="7px"
          fill="none"
          stroke={poly.color}
          strokeWidth="1px"
        />
      </g>
    );
  };

  const renderBoundingBox = (
    poly: AnnotationObjectCommon & AnnotationObjectAABB,
    select: boolean
  ) => {
    return (
      <polygon
        key={poly.objectHash}
        style={{
          fillOpacity: ".40",
          stroke: poly.color,
          strokeWidth: "4px",
        }}
        points={""} // FIXME:
      />
    );
  };

  const renderObject = (object: AnnotationObject) => {
    const select =
      annotationHash === undefined || object.objectHash === annotationHash;
    if (object.shape === "point") {
      return renderPoint(object, select);
    } else if (object.shape === "polygon") {
      return renderPolygon(object, select);
    } else {
      throw Error("Unknown shape: " + object.shape);
    }
  };

  return (
    <svg className="absolute top-0 right-0 h-full w-full">
      {objects.map(renderObject)}
    </svg>
  );
}
