import { Select } from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FaExpand, FaEdit } from "react-icons/fa";
import { HiOutlineTag } from "react-icons/hi";
import { MdClose, MdImageSearch, MdOutlineImage } from "react-icons/md";
import { TbPolygon } from "react-icons/tb";
import { VscClearAll } from "react-icons/vsc";

import { Streamlit } from "streamlit-component-lib";

import useResizeObserver from "use-resize-observer";
import { classy } from "../../helpers/classy";

type ItemMetadata = {
  metrics: Record<string, string>;
  annotator?: string | null;
  labelClass?: string | null;
};

type Item = {
  id: string;
  editUrl: string;
  tags: GroupedTags;
  url: string;
  metadata: ItemMetadata;
};

type GroupedTags = {
  data: string[];
  label: string[];
};

export type Props = { items: Item[]; tags: GroupedTags };

export const Explorer = ({ items, tags }: Props) => {
  const [previewedItem, setPreviewedItem] = useState<Item | null>(null);
  const [similarityItem, setSimilarityItem] = useState<Item | null>(null);
  const [selectedItems, setSelectedItems] = useState(new Set<string>());

  const [itemMap, setItemMap] = useState(
    new Map(
      items.map((item) => [
        item.id,
        { ...item, url: document.referrer + item.url },
      ])
    )
  );

  const toggleImageSelection = (id: Item["id"]) => {
    setSelectedItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();

  const closePreview = () => setPreviewedItem(null);
  const showSimilarItems = (item: Item) => (
    closePreview(), setSimilarityItem(item)
  );

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  console.log(itemMap);

  return (
    <div ref={ref} className="flex">
      {previewedItem ? (
        <ItemPreview
          item={previewedItem}
          tags={tags}
          onClose={closePreview}
          onShowSimilar={() => showSimilarItems(previewedItem)}
        />
      ) : (
        <div className="flex flex-col gap-5">
          {similarityItem && (
            <div className="flex gap-3">
              <figure>
                <img
                  className="w-48 h-auto object-cover rounded"
                  src={similarityItem.url}
                />
              </figure>
              <h1 className="text-lg">Similar items</h1>
            </div>
          )}
          <div className="flex justify-between">
            <div className="dropdown dropdown-bottom">
              <label
                tabIndex={0}
                className={classy("btn btn-ghost gap-2", {
                  "btn-disabled": !selectedItems.size,
                })}
              >
                <HiOutlineTag />
                Tag
              </label>
              <TaggingForm
                tags={tags}
                tabIndex={0}
                onApply={(tags) => (
                  console.log(tags), setSelectedItems(new Set())
                )}
              />
            </div>
            <button
              className={classy("btn btn-ghost gap-2", {
                "btn-disabled": !selectedItems.size,
              })}
              onClick={() => setSelectedItems(new Set())}
            >
              <VscClearAll />
              Clear selection
            </button>
          </div>
          <form
            onChange={({ target }) =>
              toggleImageSelection((target as HTMLInputElement).name)
            }
            onSubmit={(e) => e.preventDefault()}
            className="flex-1 grid gap-1 grid-cols-4"
          >
            {[...itemMap.values()].map((item) => (
              <GalleryItem
                key={item.url}
                item={item}
                onExpand={() => setPreviewedItem(item)}
                onShowSimilar={() => showSimilarItems(item)}
                selected={selectedItems.has(item.url)}
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

const TaggingForm = ({
  tags,
  className,
  onApply,
  ...rest
}: { tags: GroupedTags; onApply?: (tags: GroupedTags) => void } & Omit<
  JSX.IntrinsicElements["form"],
  "onSubmit"
>) => {
  const [selectedTab, setTab] = useState<typeof TABS[number]>(TABS[0]);
  const [selectedTags, setSelectedTags] = useState<GroupedTags>({
    data: [],
    label: [],
  });

  const onChange = useCallback(
    (tags: string[]) =>
      setSelectedTags((prev) => ({ ...prev, [selectedTab.value]: tags })),
    [selectedTab.value]
  );

  return (
    <form
      {...rest}
      onSubmit={(event) => (event.preventDefault(), onApply?.(selectedTags))}
      className={classy(
        "dropdown-content card card-compact w-64 p-2 shadow bg-base-100 text-primary-content",
        className
      )}
    >
      <div className="tabs flex justify-center bg-base-100">
        {TABS.map((tab) => (
          <a
            key={tab.value}
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
          options={tags[selectedTab.value].map((tag) => ({ value: tag }))}
        />
      </div>
      {onApply && (
        <div className="card-actions flex justify-center">
          <button className="btn btn-ghost">Apply</button>
        </div>
      )}
    </form>
  );
};

const ItemPreview = ({
  item,
  onClose,
  onShowSimilar,
  tags,
}: {
  item: Item;
  tags: GroupedTags;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
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
            onClick={() => window.open(item.editUrl, "_blank")}
          >
            <FaEdit />
            Edit
          </button>
          <div className="dropdown dropdown-bottom">
            <label tabIndex={0} className="btn btn-ghost gap-2">
              <HiOutlineTag />
              Tag
            </label>
            <TaggingForm tags={tags} tabIndex={0} />
          </div>
        </div>
        <button onClick={onClose} className="btn btn-square btn-outline">
          <MdClose className="text-base" />
        </button>
      </div>
      <img className="w-full h-auto object-cover rounded" src={item.url} />
      <MetadataMetrics metrics={item.metadata.metrics} />
    </div>
  );
};

const GalleryItem = ({
  item,
  selected,
  onExpand,
  onShowSimilar,
}: {
  item: Item;
  selected: boolean;
  onExpand: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => (
  <div className="card relative align-middle form-control">
    <label className="group label cursor-pointer p-0">
      <input
        name={item.url}
        type="checkbox"
        checked={selected}
        readOnly
        className={classy(
          "peer checkbox absolute left-1 top-1 opacity-0 group-hover:opacity-100 checked:opacity-100"
        )}
      />
      <img
        className={classy(
          "w-full h-full object-cover group-hover:opacity-30 rounded transition-opacity peer-checked:transition-none"
        )}
        src={item.url}
      />
      <div className="absolute flex gap-2 top-1 right-1 opacity-0 group-hover:opacity-100">
        <button onClick={onExpand} className="btn btn-square">
          <FaExpand />
        </button>
      </div>
    </label>
    <div className="card-body p-2">
      <div className="card-actions flex justify-between">
        <div>
          <button className="btn btn-ghost gap-2" onClick={onShowSimilar}>
            <MdImageSearch className="text-base" />
            Similar
          </button>
        </div>
        <button
          className="btn btn-ghost gap-2"
          onClick={() => window.open(item.editUrl.toString(), "_blank")}
        >
          <FaEdit />
          Edit
        </button>
      </div>
    </div>
  </div>
);
const MetadataMetrics = ({
  metrics,
}: {
  metrics: Item["metadata"]["metrics"];
}) => (
  <div className="w-full flex flex-col">
    {Object.entries(metrics).map(([key, value]) => (
      <div key={key}>
        <span>{key}: </span>
        <span>{parseFloat(value.toString()).toFixed(4)}</span>
      </div>
    ))}
  </div>
);
