import { useEffect, useState } from "react";
import { FaInfo, FaWindowClose, FaExpand } from "react-icons/fa";
import { Streamlit } from "streamlit-component-lib";
import useResizeObserver from "use-resize-observer";

type Image = { url: string; metadata: Record<string, string | number> };

export type Props = { images: Image[] };

const METADATA = {
  Brightness: 12156456,
  Blur: 1215,
  "Blue Values": 98987945,
  Area: 45698,
} as const;

export const Explorer = ({ images }: Props) => {
  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();

  const [showMetadata, setShowMetadata] = useState<typeof METADATA | null>(
    null
  );
  const [selectedImage, setSelectedImage] = useState<Image | null>(null);

  useEffect(() => {
    console.log(height);
    Streamlit.setFrameHeight(height);
  }, [height]);

  return (
    <div className="flex">
      {selectedImage ? (
        <>
          <img
            className="w-full h-auto object-cover transition-opacity opacity-100 group-hover:opacity-30"
            src={document.referrer + selectedImage.url}
          />
          <button
            onClick={() => setSelectedImage(null)}
            className="btn btn-square absolute top-2 right-2"
          >
            <FaWindowClose />
          </button>
        </>
      ) : (
        <>
          <div ref={ref} className="flex-1 flex gap-3 flex-wrap justify-center">
            {images.map((image) => (
              <div
                key={image.url}
                className="group relative min-w-[13rem] w-[24%] align-middle"
              >
                <img
                  className="w-full h-full object-cover transition-opacity opacity-100 group-hover:opacity-30 rounded"
                  src={document.referrer + image.url}
                />
                <div className="absolute flex gap-2 top-1 right-1 opacity-0 group-hover:opacity-100">
                  <button
                    onClick={() => setShowMetadata(METADATA)}
                    className="btn btn-square"
                  >
                    <FaInfo />
                  </button>

                  <button
                    onClick={() => setSelectedImage(image)}
                    className="btn btn-square"
                  >
                    <FaExpand />
                  </button>
                </div>
              </div>
            ))}
          </div>
          {showMetadata ? (
            <>
              <div className="divider divider-horizontal"></div>
              <div className="w-72 flex flex-col">
                <div className="flex justify-between">
                  <h1 className="font-medium text-3xl">Metadata</h1>
                  <button
                    onClick={() => setShowMetadata(null)}
                    className="btn btn-square"
                  >
                    <FaWindowClose />
                  </button>
                </div>
                {Object.entries(METADATA).map(([key, value]) => (
                  <div>
                    <span>{key}: </span>
                    <span>{value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </>
      )}
    </div>
  );
};
