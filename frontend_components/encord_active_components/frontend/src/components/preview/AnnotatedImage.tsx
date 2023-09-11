import useResizeObserver from "use-resize-observer";
import { useRef } from "react";
import * as React from "react";
import { ProjectItem } from "../../openapi/api";
import { useImageSrc } from "../../hooks/useImageSrc";
import { classy } from "../../helpers/classy";

export function AnnotatedImage(props: {
  item: ProjectItem;
  annotationHash: string | undefined;
  hideExtraAnnotations?: boolean | undefined;
  className?: string | undefined;
  mode: "preview" | "full" | "large";
  children?: React.ReactNode | undefined;
}) {
  const {
    item,
    className,
    annotationHash,
    hideExtraAnnotations,
    children,
    mode,
  } = props;

  const image = useRef<HTMLImageElement>(null);
  const { width: imageWidth, height: imageHeight } =
    useResizeObserver<HTMLImageElement>({ ref: image });

  const video = useRef<HTMLVideoElement>(null);
  const { width: videoWidth, height: videoHeight } =
    useResizeObserver<HTMLVideoElement>({
      ref: video,
    });

  const contentWidth = item.timestamp != null ? videoWidth : imageWidth;
  const contentHeight = item.timestamp != null ? videoHeight : imageHeight;

  const imageSrc = useImageSrc(item.url);

  // fit
  // width={240}
  // height={160}
  let contentStyle: React.CSSProperties = {};
  let figureStyle: React.CSSProperties = {};
  if (mode === "preview") {
    contentStyle = {
      flexShrink: 0,
      minWidth: "100%",
      minHeight: "100%",
      position: "absolute",
      overflow: "hidden",
    };
    figureStyle = {
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      overflow: "hidden",
      width: 240,
      height: 160,
      position: "relative",
    };
  }

  return (
    <figure
      className={classy(
        className,
        "relative m-0 flex h-full w-full items-center justify-center overflow-clip"
      )}
      style={figureStyle}
    >
      {children}
      {item.timestamp != null ? (
        <video
          ref={video}
          className="!rounded-none object-contain transition-opacity"
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
          className="!rounded-none object-contain transition-opacity"
          style={contentStyle}
          alt=""
          src={imageSrc}
        />
      )}
      {item.objects.length > 0 && contentWidth && contentHeight ? (
        <AnnotationRenderLayer
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

type AnnotationObjectRotBB = {
  readonly shape: "rotatable_bounding_box";
  readonly rotatable_bounding_box: {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
    readonly theta: number;
  };
};

type AnnotationObjectSkeleton = {
  readonly shape: "skeleton";
  readonly skeleton: object;
};

type AnnotationObjectBitmask = {
  readonly shape: "bitmask";
  readonly bitmask: string;
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
  (
    | AnnotationObjectPolygon
    | AnnotationObjectPoint
    | AnnotationObjectAABB
    | AnnotationObjectRotBB
    | AnnotationObjectSkeleton
    | AnnotationObjectBitmask
  );

function AnnotationRenderLayer({
  objects,
  width,
  height,
  annotationHash,
  hideExtraAnnotations,
}: {
  objects: AnnotationObject[];
  width: number;
  height: number;
  annotationHash: string | undefined;
  hideExtraAnnotations: boolean;
}) {
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
    poly: AnnotationObjectCommon &
      (AnnotationObjectAABB | AnnotationObjectRotBB),
    select: boolean | undefined
  ) => {
    const bb =
      poly.shape === "bounding_box"
        ? { ...poly.bounding_box, theta: 0 }
        : poly.rotatable_bounding_box;
    const x1 = bb.x * width;
    const y1 = bb.y * height;
    const x2 = (bb.x + bb.w) * width;
    const y2 = (bb.y + bb.h) * height;
    const c = Math.cos(bb.theta);
    const s = Math.sin(bb.theta);
    const rotate = (x: number, y: number): string => {
      const xr = x * c - y * s;
      const yr = x * s + y * c;

      return `${xr},${yr}`;
    };

    return (
      <polygon
        key={poly.objectHash}
        style={{
          fillOpacity: fillOpacity(select),
          fill: poly.color,
          strokeOpacity: strokeOpacity(select),
          stroke: poly.color,
          strokeWidth: strokeWidth(select),
        }}
        points={`${rotate(x1, y1)} ${rotate(x1, y2)} ${rotate(x2, y2)} ${rotate(
          x2,
          y1
        )}`}
      />
    );
  };

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
    } else if (
      object.shape === "bounding_box" ||
      object.shape === "rotatable_bounding_box"
    ) {
      return renderBoundingBox(object, select);
    } else {
      throw Error(`Unknown shape: ${object.shape}`);
    }
  };

  return (
    <svg
      className="absolute h-full w-full overflow-hidden"
      style={{ width, height }}
    >
      {objects.map(renderObject)}
    </svg>
  );
}
