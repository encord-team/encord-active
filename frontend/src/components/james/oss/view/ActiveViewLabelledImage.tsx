import * as React from "react";
import useResizeObserver from "use-resize-observer";
import { useDebounce } from "usehooks-ts";
import ActiveViewRawImage from "./ActiveViewRawImage";
import { ActiveProjectPreviewItemResult } from "../ActiveTypes";

function objectToSVG(
  object: ActiveProjectPreviewItemResult["objects"][0],
  w: number,
  h: number
) {
  const { color, objectHash } = object;
  if (object.shape === "point") {
    return (
      <>
        <circle
          key={`${objectHash}_inner`}
          cx={object.point[0].x * w}
          cy={object.point[0].y * h}
          r="5px"
          fill={color}
        />
        <circle
          key={`${objectHash}_outer`}
          cx={object.point[0].x * w}
          cy={object.point[0].y * h}
          r="7px"
          fill="none"
          stroke={color}
          strokeWidth="1px"
        />
      </>
    );
  } else if (object.shape === "polygon" || object.shape === "polyline") {
    const pointsRaw =
      object.shape === "polygon" ? object.polygon : object.polyline;
    const points = Object.values(pointsRaw);
    return (
      <polygon
        key={`${objectHash}_poly`}
        style={{
          fill: object.shape === "polyline" ? "none" : color,
          fillOpacity: ".20",
          stroke: color,
          strokeWidth: "2px",
        }}
        points={(points || []).map(({ x, y }) => `${x * w},${y * h}`).join(" ")}
      />
    );
  } else if (object.shape === "bounding_box") {
    const { x, y, w: bbW, h: bbH } = object.boundingBox;
    const x2 = x + bbW;
    const y2 = y + bbH;
    return (
      <polygon
        key={`${objectHash}_poly`}
        style={{
          fill: color,
          fillOpacity: ".20",
          stroke: color,
          strokeWidth: "2px",
        }}
        points={`${x * w},${y * h} ${x2 * w},${y * h} ${x2 * w},${y2 * h} ${
          x * w
        },${y2 * h}`}
      />
    );
  } else if (object.shape === "rotatable_bounding_box") {
    const { x, y, w: bbW, h: bbH, theta } = object.rotatableBoundingBox;
    const s = Math.sin(theta);
    const c = Math.cos(theta);
    const midpoint: [number, number] = [x + 0.5 * bbW, y + 0.5 * bbH];
    const transform = (xv: number, yv: number): string => {
      const sx = xv * w - midpoint[0];
      const sy = yv * h - midpoint[1];
      const rx = sx * c - sy * s;
      const ry = sx * s + sy * c;
      return `${rx + midpoint[0]},${ry + midpoint[1]}`;
    };
    const x2 = x + bbH;
    const y2 = y + bbH;
    return (
      <polygon
        key={`${objectHash}_bb`}
        style={{
          fill: color,
          fillOpacity: ".20",
          stroke: color,
          strokeWidth: "2px",
        }}
        points={`${transform(x, y)} ${transform(x2, y)} ${transform(
          x2,
          y2
        )} ${transform(x, y2)}`}
      />
    );
  } else {
    throw Error(`Unknown shape: ${object}`);
  }
}

function ActiveViewLabelledImage(props: {
  visualization: ActiveProjectPreviewItemResult;
  width?: number | undefined | null;
  extraSVG?: React.ReactElement | undefined | null;
  imageOverlay?: React.ReactElement | undefined | null;
}) {
  const { visualization, width, extraSVG, imageOverlay } = props;
  const { ref, width: imgWidth, height: imgHeight } = useResizeObserver();
  const w = useDebounce(imgWidth ?? 0, 500);
  const h = useDebounce(imgHeight ?? 0, 500);
  const { url, objects, timestamp } = visualization;
  return (
    <figure style={{ width: "max-content", height: "max-content", margin: 0 }}>
      {imageOverlay != null ? (
        <div className="absolute">{imageOverlay}</div>
      ) : null}
      {objects.length > 0 && (
        <svg width={w} height={h} className="absolute">
          {objects.map((object) => objectToSVG(object, w, h))}
          {extraSVG}
        </svg>
      )}
      <div
        ref={ref}
        style={{ width: "max-content", height: "max-content", margin: "0" }}
      >
        <ActiveViewRawImage url={url} timestamp={timestamp} width={width} />
      </div>
    </figure>
  );
}

export default ActiveViewLabelledImage;
