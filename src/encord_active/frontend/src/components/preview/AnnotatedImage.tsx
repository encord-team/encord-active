import useResizeObserver from "use-resize-observer";
import { CSSProperties, ReactNode, memo, useRef, useEffect } from "react";

import ErrorBoundary from "antd/lib/alert/ErrorBoundary";
import { WarningOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { ProjectItem } from "../../openapi/api";
import { useImageSrc } from "../../hooks/useImageSrc";
import { classy } from "../../helpers/classy";

export function AnnotatedImage(props: {
  item: ProjectItem;
  annotationHash: string | undefined;
  hideExtraAnnotations?: boolean | undefined;
  className?: string | undefined;
  mode: "preview" | "full" | "large";
  predictionTruePositive: ReadonlySet<string> | undefined;
  children?: ReactNode | undefined;
}) {
  const {
    item,
    className,
    annotationHash,
    hideExtraAnnotations,
    children,
    mode,
    predictionTruePositive,
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
  let contentStyle: CSSProperties = {};
  let figureStyle: CSSProperties = {};
  if (mode === "preview") {
    contentStyle = {
      flexShrink: 0,
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

  // Ensure correctness if timestamp changes but video url does not.
  useEffect(() => {
    const videoRef = video.current;
    if (videoRef != null) {
      videoRef.currentTime = item.timestamp || 0;
    }
  }, [item.timestamp]);

  return (
    <figure
      className={classy(
        className,
        "m-0 flex h-full w-full items-center justify-center overflow-clip"
      )}
      style={figureStyle}
    >
      {children}
      {item.timestamp != null ? (
        <video
          ref={video}
          className="!rounded-none object-contain "
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
          // className="aspect-square !rounded-none object-cover transition-opacity"
          className="!rounded-none object-contain transition-opacity"
          style={contentStyle}
          alt=""
          src={imageSrc}
        />
      )}
      {item.objects.length > 0 && contentWidth && contentHeight ? (
        <ErrorBoundary
          message={
            <div>
              <WarningOutlined />
              Rendering Error
            </div>
          }
          description={null}
        >
          <AnnotationRenderLayer
            objects={item.objects as AnnotationObject[]}
            width={contentWidth}
            height={contentHeight}
            annotationHash={annotationHash}
            predictionTruePositive={predictionTruePositive}
            hideExtraAnnotations={hideExtraAnnotations ?? false}
          />
        </ErrorBoundary>
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
  readonly boundingBox: {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
  };
  readonly bounding_box?: {
    readonly x?: number;
    readonly y?: number;
    readonly w?: number;
    readonly h?: number;
  };
};

type AnnotationObjectRotBB = {
  readonly shape: "rotatable_bounding_box";
  readonly rotatableBoundingBox: {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
    readonly theta: number;
  };
  readonly rotatable_bounding_box?: {
    readonly x: number;
    readonly y: number;
    readonly w: number;
    readonly h: number;
    readonly theta: number;
  };
};

type AnnotationObjectPolyline = {
  readonly shape: "polyline";
  readonly polyline: Record<string, { readonly x: number; readonly y: number }>;
};

type AnnotationObjectSkeleton = {
  readonly shape: "skeleton";
  readonly skeleton: object;
};

type AnnotationObjectBitmask = {
  readonly shape: "bitmask";
  readonly bitmask: {
    readonly top: number;
    readonly left: number;
    readonly width: number;
    readonly height: number;
    readonly rleString: string;
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
  (
    | AnnotationObjectPolygon
    | AnnotationObjectPoint
    | AnnotationObjectAABB
    | AnnotationObjectRotBB
    | AnnotationObjectSkeleton
    | AnnotationObjectBitmask
    | AnnotationObjectPolyline
  );

const AnnotationRenderLayer = memo(AnnotationRenderLayerRaw);

function AnnotationRenderLayerRaw({
  objects,
  width,
  height,
  annotationHash,
  hideExtraAnnotations,
  predictionTruePositive,
}: {
  objects: AnnotationObject[];
  width: number;
  height: number;
  annotationHash: string | undefined;
  hideExtraAnnotations: boolean;
  predictionTruePositive: ReadonlySet<string> | undefined;
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

  const color = (poly: AnnotationObjectCommon): string => {
    if (predictionTruePositive !== undefined) {
      return predictionTruePositive.has(poly.objectHash)
        ? "#22c55e"
        : "#ef4444";
    } else {
      return poly.color;
    }
  };

  const renderPolygon = (
    poly: AnnotationObjectCommon & AnnotationObjectPolygon,
    select: boolean | undefined
  ) => (
    <polygon
      key={poly.objectHash}
      style={{
        fillOpacity: fillOpacity(select),
        fill: color(poly),
        strokeOpacity: strokeOpacity(select),
        stroke: color(poly),
        strokeWidth: strokeWidth(select),
      }}
      points={Object.values(poly.polygon)
        .map(({ x, y }) => `${x * width},${y * height}`)
        .join(" ")}
    />
  );

  const renderPolyline = (
    poly: AnnotationObjectCommon & AnnotationObjectPolyline,
    select: boolean | undefined
  ) => (
    <polyline
      key={poly.objectHash}
      style={{
        fillOpacity: 0,
        strokeOpacity: strokeOpacity(select),
        stroke: color(poly),
        strokeWidth: strokeWidth(select),
      }}
      points={Object.values(poly.polyline)
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
          r="1px"
          fill={color(poly)}
          fillOpacity={fillOpacity(select)}
        />
        <circle
          cx={x * width}
          cy={y * height}
          r="2px"
          fill="none"
          stroke={color(poly)}
          strokeWidth="1px"
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
        ? { ...poly.bounding_box, ...poly.boundingBox, theta: 0 }
        : { ...poly.rotatableBoundingBox, ...poly.rotatableBoundingBox };
    const x1 = bb.x * width;
    const y1 = bb.y * height;
    const x2 = (bb.x + bb.w) * width;
    const y2 = (bb.y + bb.h) * height;
    const xc = (x1 + x2) / 2.0;
    const yc = (y1 + y2) / 2.0;
    const DEG_TO_RAD = Math.PI / 180;
    const c = Math.cos(bb.theta * DEG_TO_RAD);
    const s = Math.sin(bb.theta * DEG_TO_RAD);

    const rotate = (x: number, y: number): string => {
      const xr = xc + ((x - xc) * c - (y - yc) * s);
      const yr = yc + ((x - xc) * s + (y - yc) * c);

      return `${xr},${yr}`;
    };

    return (
      <polygon
        key={poly.objectHash}
        style={{
          fillOpacity: fillOpacity(select),
          fill: color(poly),
          strokeOpacity: strokeOpacity(select),
          stroke: color(poly),
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
    } else if (object.shape === "polyline") {
      return renderPolyline(object, select);
    } else if (object.shape === "bitmask") {
      const { bitmask } = object;

      return (
        <CoCoBitmaskRaw
          key={object.objectHash}
          bitmask={bitmask.rleString}
          width={bitmask.width}
          height={bitmask.height}
          imgWidth={width}
          imgHeight={height}
          x={bitmask.left}
          y={bitmask.top}
          color={color(object)}
        />
      );
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

function CoCoBitmaskRaw(props: {
  bitmask: string;
  width: number;
  height: number;
  imgWidth: number;
  imgHeight: number;
  x: number;
  y: number;
  color: string;
}) {
  const { bitmask, x, y, width, height, imgWidth, imgHeight, color } = props;
  const { data: imageUrlRef, refetch } = useQuery(
    ["BITMASK:compileImage", bitmask, width, height, color],
    async () => {
      const { data: imageData, ...imageMetadata } = cocoBitmaskToImageBitmap(
        bitmask,
        width,
        height,
        color
      );
      // Would use ImageBitmap but the current structure is a svg which requires an object url
      const canvas = new OffscreenCanvas(imageData.width, imageData.height);
      const canvasContext = canvas.getContext("2d");
      if (canvasContext == null) {
        return undefined;
      }
      canvasContext.putImageData(imageData, 0, 0);
      const blob = await canvas.convertToBlob();

      return {
        url: URL.createObjectURL(blob),
        count: 0,
        ...imageMetadata,
      };
    },
    { staleTime: Infinity, cacheTime: 0 }
  );
  useEffect(() => {
    if (imageUrlRef !== undefined) {
      if (imageUrlRef.url === "") {
        return undefined;
      }
      imageUrlRef.count += 1;
      return () => {
        imageUrlRef.count -= 1;
        if (imageUrlRef.count <= 0) {
          URL.revokeObjectURL(imageUrlRef.url);
          imageUrlRef.url = "";
        }
      };
    }
    return undefined;
  }, [imageUrlRef]);
  useEffect(() => {
    const urlMaybe = imageUrlRef?.url;
    if (urlMaybe === "") {
      refetch();
    }
  }, [imageUrlRef?.url, refetch]);

  if (imageUrlRef === undefined) {
    return null;
  }

  return (
    <image
      x={((x + imageUrlRef.xOffset) / width) * imgWidth}
      y={((y + imageUrlRef.yOffset) / height) * imgHeight}
      width={(imageUrlRef.width / width) * imgWidth}
      height={(imageUrlRef.height / height) * imgHeight}
      href={imageUrlRef.url}
    />
  );
}

function cocoBitmaskToImageBitmap(
  bitmask: string,
  width: number,
  height: number,
  color: string
): {
  readonly data: ImageData;
  readonly xOffset: number;
  readonly yOffset: number;
  readonly width: number;
  readonly height: number;
} {
  // bitmask string => count list
  const encoder = new TextEncoder();
  const bytes = encoder.encode(bitmask);
  const counts: number[] = [];
  let p = 0;
  while (p < bytes.length) {
    let x = 0;
    let k = 0;
    let more = true;
    while (more && p < bytes.length) {
      const c = bytes[p] - 48;
      x |= (c & 0x1f) << (5 * k);
      more = (c & 0x20) !== 0;
      p += 1;
      k += 1;
      if (!more && (c & 0x10) !== 0) {
        x |= -1 << (5 * k);
      }
    }
    if (counts.length > 2) {
      x += counts[counts.length - 2];
    }
    counts.push(x);
  }

  // color decode
  const r = parseInt(color.substring(1, 3), 16);
  const g = parseInt(color.substring(3, 5), 16);
  const b = parseInt(color.substring(5, 7), 16);
  const rgba = [r, g, b, 127];

  // Find min image height
  const yOffset = Math.floor((counts[0] ?? 0) / height);
  counts[0] = (counts[0] ?? 0) - yOffset * height;
  const yEndOffset =
    counts.length <= 1
      ? 0
      : Math.floor((counts[counts.length - 1] ?? 0) / height);
  counts[counts.length - 1] =
    (counts[counts.length - 1] ?? 0) - yEndOffset * height;
  const resizedHeight = Math.max(1, height - yOffset - yEndOffset);

  // Find min image width
  let xOffset = width - 1;
  let xOffsetRangeEnd = 0;
  let xModuloOffset = 0;
  let fillModulo = false;
  counts.forEach((countRaw) => {
    const count = countRaw;
    if (fillModulo) {
      if (xModuloOffset + count >= width) {
        // No clamp possible
        xOffset = 0;
        xOffsetRangeEnd = width;
      } else {
        // Select minimum clamp
        xOffset = Math.min(xOffset, xModuloOffset);
        xOffsetRangeEnd = Math.max(xOffsetRangeEnd, xModuloOffset + count);
      }
    }
    fillModulo = !fillModulo;
    xModuloOffset = (xModuloOffset + count) % width;
  });
  const xEndOffset = Math.min(width - xOffsetRangeEnd, width - xOffset);
  if (xEndOffset !== 0 || xOffset !== 0) {
    xModuloOffset = 0;
    counts.forEach((countRaw, i) => {
      let count = countRaw;
      while (xModuloOffset + count >= width) {
        // Wrap-around (potentially many times)
        counts[i] -= xEndOffset + xOffset;
        count -= width;
      }
      if (xModuloOffset + count >= xOffsetRangeEnd) {
        counts[i] -= xOffsetRangeEnd - (xModuloOffset + count);
      }
      if (xModuloOffset < xOffset) {
        counts[i] -= xOffset - xModuloOffset;
      }
      xModuloOffset = (xModuloOffset + count) % width;
    });
  }
  const resizedWidth = Math.max(1, width - xOffset - xEndOffset);

  // count list
  const decoded = new Uint8Array(resizedWidth * resizedHeight * 4);
  let offset = 0;
  let fill = false;
  counts.forEach((count) => {
    if (fill) {
      decoded.fill(0xff, 4 * offset, 4 * (offset + count));
    }
    offset += count;
    fill = !fill;
  });
  // FIXME: slow -> try find a wa
  const decodedColor = decoded.map((value, index) => {
    if (value === 0) {
      return 0;
    } else {
      return rgba[index % 4];
    }
  });

  return {
    data: new ImageData(
      new Uint8ClampedArray(decodedColor),
      resizedWidth,
      resizedHeight
    ),
    yOffset,
    xOffset,
    width: resizedWidth,
    height: resizedHeight,
  };
}
