import { Select } from "antd";
import { useCallback, useEffect, useState } from "react";
import { FaExpand, FaEdit } from "react-icons/fa";
import { HiOutlineTag } from "react-icons/hi";
import { MdClose, MdImageSearch, MdOutlineImage } from "react-icons/md";
import { TbPolygon } from "react-icons/tb";
import { VscClearAll } from "react-icons/vsc";

import { Streamlit } from "streamlit-component-lib";

import useResizeObserver from "use-resize-observer";
import { classy } from "../../helpers/classy";

type Image = {
  url: string;
  metadata: Record<string, string | number>;
};

export type Props = { images: Image[] };

export const Explorer = ({ images }: Props) => {
  const [previewedImage, setPreviewedImage] = useState<Image | null>(null);
  const [similarityImage, setSimilarityImage] = useState<Image | null>(null);
  const [selectedImages, setSelectedImages] = useState(new Set<string>());

  const toggleImageSelection = (imageUrl: string) => {
    setSelectedImages((prev) => {
      const next = new Set(prev);
      if (next.has(imageUrl)) next.delete(imageUrl);
      else next.add(imageUrl);
      return next;
    });
  };

  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();

  const closePreview = () => setPreviewedImage(null);

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  return (
    <div ref={ref} className="flex">
      {previewedImage ? (
        <ImagePreview
          image={previewedImage}
          onClose={closePreview}
          onShowSimilar={() => (
            closePreview(), setSimilarityImage(previewedImage)
          )}
        />
      ) : (
        <div className="flex flex-col gap-5">
          {similarityImage && (
            <div className="flex gap-3">
              <figure>
                <img
                  className="w-48 h-auto object-cover rounded"
                  src={document.referrer + similarityImage.url}
                />
              </figure>
              <h1 className="text-lg">Similar images</h1>
            </div>
          )}
          <div className="flex justify-between">
            <div className="dropdown dropdown-bottom">
              <label
                tabIndex={0}
                className={classy("btn btn-ghost gap-2", {
                  "btn-disabled": !selectedImages.size,
                })}
              >
                <HiOutlineTag />
                Tag
              </label>
              <TaggingForm tabIndex={0} />
            </div>
            <button
              className={classy("btn btn-ghost gap-2", {
                "btn-disabled": !selectedImages.size,
              })}
              onClick={() => setSelectedImages(new Set())}
            >
              <VscClearAll />
              Clear selection
            </button>
          </div>
          <form
            onChange={({ target }) =>
              toggleImageSelection((target as HTMLInputElement).name)
            }
            className="flex-1 grid gap-1 grid-cols-4"
          >
            {images.map((image) => (
              <GalleryItem
                key={image.url}
                image={image}
                onExpand={() => setPreviewedImage(image)}
                selected={selectedImages.has(image.url)}
              />
            ))}
          </form>
        </div>
      )}
    </div>
  );
};

const TABS = [
  { value: "data", label: "Data", Icon: MdOutlineImage },
  { value: "label", label: "Label", Icon: TbPolygon },
] as const;

const TaggingForm = ({ className, ...rest }: JSX.IntrinsicElements["form"]) => {
  const [selectedTab, setTab] = useState<typeof TABS[number]>(TABS[0]);
  const [selectedTags, setSelectedTags] = useState({ data: [], label: [] });

  const tags = {
    data: [
      { value: "too bright" },
      { value: "too dark" },
      { value: "not labeled" },
    ],
    label: [{ value: "bad label" }, { value: "overlap" }],
  };

  const onChange = useCallback(
    (tags: string[]) =>
      setSelectedTags((prev) => ({ ...prev, [selectedTab.value]: tags })),
    [selectedTab.value]
  );

  return (
    <form
      {...rest}
      className={classy(
        "dropdown-content card card-compact w-64 p-2 shadow bg-base-100 text-primary-content",
        className
      )}
    >
      <div className="tabs flex justify-center bg-base-100">
        {TABS.map((tab) => (
          <a
            className={classy("tab tab-bordered gap-2", {
              "tab-active": selectedTab.label == tab.label,
            })}
            onClick={() => setTab(tab)}
          >
            <tab.Icon className="text-base" />
            {tab.label}
          </a>
        ))}
      </div>
      <div className="card-body">
        <Select
          mode="tags"
          placeholder="Tags"
          onChange={onChange}
          value={selectedTags[selectedTab.value]}
          options={tags[selectedTab.value]}
        />
      </div>
    </form>
  );
};

const ImagePreview = ({
  image,
  onClose,
  onShowSimilar,
}: {
  image: Image;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { url, tags, ...metadata } = image.metadata;

  return (
    <div className="w-full flex flex-col items-center gap-3 p-1">
      <div className="w-full flex justify-between">
        <div className="flex gap-3">
          <button className="btn btn-ghost gap-2" onClick={onShowSimilar}>
            <MdImageSearch className="text-base" />
            Similar
          </button>
          <button
            className="btn btn-ghost gap-2"
            onClick={() => window.open(url.toString(), "_blank")}
          >
            <FaEdit />
            Edit
          </button>
          <div className="dropdown dropdown-bottom">
            <label tabIndex={0} className="btn btn-ghost gap-2">
              <HiOutlineTag />
              Tag
            </label>
            <TaggingForm tabIndex={0} />
          </div>
        </div>
        <button onClick={onClose} className="btn btn-square btn-outline">
          <MdClose className="text-base" />
        </button>
      </div>
      <img
        className="w-full h-auto object-cover rounded"
        src={document.referrer + image.url}
      />
      <Metadata metadata={metadata} />
    </div>
  );
};

const GalleryItem = ({
  image,
  selected,
  onExpand,
}: {
  image: Image;
  selected: boolean;
  onExpand: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { url, tags, ...metadata } = image.metadata;

  return (
    <div className="card relative align-middle form-control">
      <label className="group label cursor-pointer p-0">
        <input
          name={image.url}
          type="checkbox"
          defaultChecked={selected}
          className={classy(
            "peer checkbox absolute left-1 top-1 opacity-0 group-hover:opacity-100 checked:opacity-100"
          )}
        />
        <img
          className={classy(
            "w-full h-full object-cover group-hover:opacity-30 rounded transition-opacity peer-checked:transition-none"
          )}
          src={document.referrer + image.url}
        />
        <div className="absolute flex gap-2 top-1 right-1 opacity-0 group-hover:opacity-100">
          <button onClick={onExpand} className="btn btn-square">
            <FaExpand />
          </button>
        </div>
      </label>
      <div className="card-body p-2">
        <button
          className="btn btn-ghost gap-2"
          onClick={() => window.open(url.toString(), "_blank")}
        >
          <FaEdit />
          Edit
        </button>
      </div>
    </div>
  );
};
const Metadata = ({ metadata }: { metadata: Image["metadata"] }) => (
  <div className="w-full flex flex-col">
    {Object.entries(metadata).map(([key, value]) => (
      <div key={key}>
        <span>{key}: </span>
        <span>{parseFloat(value.toString()).toFixed(4)}</span>
      </div>
    ))}
  </div>
);
