import useResizeObserver from "use-resize-observer";
import { useRef } from "react";
import * as React from "react";
import { ProjectItem } from "../../openapi/api";
import { QueryContext } from "../../hooks/Context";
import { useImageSrc } from "../../hooks/useImageSrc";

export function AnnotatedImage(props: {
  queryContext: QueryContext;
  item: ProjectItem;
  annotationHash: string | undefined;
  className?: string | undefined;
  width?: number | undefined;
  height?: number | undefined;
  fit?: boolean;
  children?: React.ReactNode | undefined;
}) {
  const {
    item,
    queryContext,
    className,
    annotationHash,
    children,
    width,
    height,
    fit,
  } = props;
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
  const contentWidth = item.timestamp != null ? videoWidth : imageWidth;
  const contentHeight = item.timestamp != null ? videoHeight : imageHeight;
  const imageSrc = useImageSrc(queryContext, item.url);

  const exactFit = width && height && fit;
  const contentStyle: React.CSSProperties = {
    // width: fitWidth ? width : undefined,
    // height: fitWidth ? undefined : height,
    flexShrink: exactFit ? 0 : undefined,
    minWidth: exactFit ? "100%" : undefined,
    minHeight: exactFit ? "100%" : undefined,
    position: exactFit ? "absolute" : undefined,
    overflow: exactFit ? "hidden" : undefined,
  };
  const figureStyle: React.CSSProperties = {
    display: exactFit ? "flex" : undefined,
    justifyContent: exactFit ? "center" : undefined,
    alignItems: exactFit ? "center" : undefined,
    overflow: exactFit ? "hidden" : undefined,
    width,
    height,
    position: "relative",
  };

  return (
    <figure className={className} style={figureStyle}>
      {children}
      {item.timestamp != null ? (
        <video
          ref={video}
          className="rounded object-contain transition-opacity"
          style={contentStyle}
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
          style={contentStyle}
          alt=""
          src={imageSrc}
        />
      )}
      {item.objects.length > 0 && contentWidth && contentHeight ? (
        <AnnotationRenderLayer
          layout={width && height ? "relative" : "absolute"}
          objects={item.objects as AnnotationObject[]}
          width={contentWidth}
          height={contentHeight}
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
  layout: "relative" | "absolute";
  width: number;
  height: number;
  annotationHash: string | undefined;
}) {
  const { objects, layout, width, height, annotationHash } = props;

  const renderPolygon = (
    poly: AnnotationObjectCommon & AnnotationObjectPolygon,
    select: boolean
  ) => (
    <polygon
      key={poly.objectHash}
      style={{
        fillOpacity: select && annotationHash !== undefined ? "50%" : "20%",
        fill: poly.color,
        strokeOpacity: select ? "100%" : "40%",
        stroke: poly.color,
        strokeWidth: select ? "2px" : "1px",
      }}
      points={Object.values(poly.polygon)
        .map(({ x, y }) => `${x * width},${y * height}`)
        .join(" ")}
    />
  );

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
  ) => (
    <polygon
      key={poly.objectHash}
      style={{
        fillOpacity: ".40",
        stroke: poly.color,
        strokeWidth: "4px",
      }}
      points="" // FIXME:
    />
  );

  const renderObject = (object: AnnotationObject) => {
    const select =
      annotationHash === undefined || object.objectHash === annotationHash;
    if (object.shape === "point") {
      return renderPoint(object, select);
    } else if (object.shape === "polygon") {
      return renderPolygon(object, select);
    } else {
      throw Error(`Unknown shape: ${object.shape}`);
    }
  };

  return (
    <svg
      className={`${layout} top-0 right-0 overflow-hidden`}
      style={{ width, height }}
    >
      {objects.map(renderObject)}
    </svg>
  );
}
