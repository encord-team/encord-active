import * as React from "react";
import useResizeObserver from "use-resize-observer";
import { useRef, useState } from "react";
import { Spin } from "antd";
import { apiUrl } from "../../constants";
import { useImageSrc } from "../../hooks/useImageSrc";
import { classy } from "../../helpers/classy";
import { ProjectPreviewItemResult } from "../Types";
import { loadingIndicator } from "../Spin";

export function ImageWithPolygons(props: {
  preview: ProjectPreviewItemResult;
  className: string;
}) {
  const { preview, className } = props;
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
  const width = preview.timestamp != null ? videoWidth : imageWidth;
  const height = preview.timestamp != null ? videoHeight : imageHeight;
  const [polygons, setPolygons] = useState<
    Pick<ItemLabelObject, "points" | "boundingBoxPoints" | "shape" | "color">[]
  >([]);

  /* FIXME:
  useEffect(() => {
    if (width == null || height == null) return;
    const objects = getObjects(item);

    setPolygons(
      objects.map(({ points, color, shape, boundingBoxPoints }) => ({
        color,
        points,
        shape,
        boundingBoxPoints,
      })),
    );
  }, [width, height, item.id]);
  */

  const itemUrl = preview.url.startsWith("http")
    ? preview.url
    : `${apiUrl}${preview.url}`;

  const imgSrcUrl = useImageSrc(itemUrl);

  if (imgSrcUrl === undefined) {return <Spin indicator={loadingIndicator} />;}

  return (
    <figure className={classy("relative", className)}>
      {preview.timestamp != null ? (
        <video
          ref={video}
          className="rounded object-contain transition-opacity"
          src={imgSrcUrl}
          muted
          controls={false}
          onLoadedMetadata={() => {
            const videoRef = video.current;
            if (videoRef != null) {
              videoRef.currentTime = preview.timestamp || 0;
            }
          }}
        />
      ) : (
        <img
          ref={image}
          className="rounded object-contain transition-opacity"
          alt=""
          src={imgSrcUrl}
        />
      )}
      {width && height && polygons.length > 0 && (
        <svg className="absolute top-0 right-0 h-full w-full">
          {polygons.map(
            ({ points, boundingBoxPoints, color, shape }, index) => {
              if (shape === "point" && points)
                {return (
                  <g key={index}>
                    <circle
                      key={`${index  }_inner`}
                      cx={points[0].x}
                      cy={points[0].y}
                      r="5px"
                      fill={color}
                    />
                    <circle
                      key={`${index  }_outer`}
                      cx={points[0].x}
                      cy={points[0].y}
                      r="7px"
                      fill="none"
                      stroke={color}
                      strokeWidth="1px"
                    />
                  </g>
                );}
              return (
                <g key={index} fill={shape === "polyline" ? "none" : color}>
                  {points && (
                    <polygon
                      key={`${index  }_polygon`}
                      style={{
                        fillOpacity: ".20",
                        stroke: color,
                        strokeWidth: "2px",
                      }}
                      points={pointsRecordToPolygonPoints(
                        points,
                        width,
                        height
                      )}
                    />
                  )}
                  {boundingBoxPoints && (
                    <polygon
                      key={`${index  }_box`}
                      style={{
                        fillOpacity: ".40",
                        stroke: color,
                        strokeWidth: "4px",
                      }}
                      points={pointsRecordToPolygonPoints(
                        boundingBoxPoints,
                        width,
                        height
                      )}
                    />
                  )}
                </g>
              );
            }
          )}
        </svg>
      )}
    </figure>
  );
}
