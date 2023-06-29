import * as React from "react";
import { useRef } from "react";

function ActiveViewRawImage(props: {
  url: string;
  timestamp?: number | undefined | null;
  width?: number | undefined | null;
}) {
  const { url, timestamp, width } = props;
  const ref = useRef<HTMLVideoElement | null>(null);

  if (timestamp != null) {
    return (
      <video
        src={url}
        ref={ref}
        preload="metadata"
        width={width || undefined}
        muted
        controls={false}
        onLoadedMetadata={() => {
          const { current } = ref;
          if (current != null) {
            current.currentTime = timestamp;
          }
        }}
      />
    );
  } else {
    return <img src={url} alt="" width={width || undefined} />;
  }
}

export default ActiveViewRawImage;
