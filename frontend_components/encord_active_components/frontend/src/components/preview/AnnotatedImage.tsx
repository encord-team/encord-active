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
  hideExtraAnnotations?: boolean | undefined;
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
    hideExtraAnnotations,
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
          hideExtraAnnotations={hideExtraAnnotations ?? false}
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
  hideExtraAnnotations: boolean;
}) {
  const {
    objects,
    layout,
    width,
    height,
    annotationHash,
    hideExtraAnnotations,
  } = props;

  const fillOpacity = (select: boolean | undefined): string => {
    if (select === undefined) {
      return "20%";
    }
    return select ? "60%" : "10%";
  };

  const strokeOpacity = (select: boolean | undefined): string => "100%";

  const strokeWidth = (select: boolean | undefined): string => {
    if (select === undefined) {
      return "2px";
    }
    return select ? "4px" : "1px";
  };

  const renderPolygon = (
    poly: AnnotationObjectCommon & AnnotationObjectPolygon,
    select: boolean | undefined
  ) => (
    <polygon
      key={poly.objectHash}
      style={{
        fillOpacity: fillOpacity(select),
        fill: poly.color,
        strokeOpacity: strokeOpacity(select),
        stroke: poly.color,
        strokeWidth: strokeWidth(select),
      }}
      points={Object.values(poly.polygon)
        .map(({ x, y }) => `${x * width},${y * height}`)
        .join(" ")}
    />
  );

  const renderPoint = (
    poly: AnnotationObjectCommon & AnnotationObjectPoint,
    select: boolean | undefined
  ) => {
    const { x, y } = poly.point["0"];

    return (
      <g key={poly.objectHash}>
        <circle
          cx={x * width}
          cy={y * height}
          r="5px"
          fill={poly.color}
          fillOpacity={fillOpacity(select)}
        />
        <circle
          cx={x * width}
          cy={y * height}
          r="7px"
          fill="none"
          stroke={poly.color}
          strokeWidth={strokeWidth(select)}
          strokeOpacity={strokeOpacity(select)}
        />
      </g>
    );
  };

  const renderBoundingBox = (
    poly: AnnotationObjectCommon & AnnotationObjectAABB,
    select: boolean | undefined
  ) => (
    <polygon
      key={poly.objectHash}
      style={{
        fillOpacity: fillOpacity(select),
        fill: poly.color,
        strokeOpacity: strokeOpacity(select),
        stroke: poly.color,
        strokeWidth: strokeWidth(select),
      }}
      points="" // FIXME:
    />
  );

  const renderObject = (object: AnnotationObject) => {
    if (hideExtraAnnotations && object.objectHash !== annotationHash) {
      return null;
    }
    const select =
      annotationHash === undefined
        ? undefined
        : object.objectHash === annotationHash;
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
