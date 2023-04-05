import { Select } from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FaExpand, FaEdit } from "react-icons/fa";
import { HiOutlineTag } from "react-icons/hi";
import {
  MdClose,
  MdImageSearch,
  MdOutlineImage,
  MdOutlineNavigateBefore,
  MdOutlineNavigateNext,
} from "react-icons/md";
import { TbPolygon } from "react-icons/tb";
import { RiUserLine } from "react-icons/ri";
import { VscClearAll, VscSymbolClass } from "react-icons/vsc";

import { Streamlit } from "streamlit-component-lib";

import useResizeObserver from "use-resize-observer";
import { classy } from "../../helpers/classy";

/* type ChangePage = ["CHANGE_PAGE", number]; */
/**/
/* type Output = ChangePage; */

/* const pushOutput = (output: Output) => Streamlit.setComponentValue(output); */

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

type PaginationInfo = {
  current: number;
  total: number;
};

export type Props = {
  items: Item[];
  tags: GroupedTags;
  pagination: PaginationInfo;
};

export const Explorer = ({ items, tags }: Props) => {
  const [previewedItem, setPreviewedItem] = useState<Item | null>(null);
  const [similarityItem, setSimilarityItem] = useState<Item | null>(null);
  const [selectedItems, setSelectedItems] = useState(new Set<string>());

  const [page, setPage] = useState(1);
  const [pageCount, setPageCount] = useState<PageCount>(PAGE_COUNTS[0]);

  const [itemMap, setItemMap] = useState(
    new Map(items.map((item) => [item.id, item]))
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

  /* console.log(itemMap); */

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
        <div className="flex flex-col gap-5 items-center pb-5">
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
          <div className="flex w-full justify-between">
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
            {[...itemMap.values()]
              .slice(page * pageCount, page * pageCount + pageCount)
              .map((item) => (
                <GalleryItem
                  key={item.id}
                  item={item}
                  onExpand={() => setPreviewedItem(item)}
                  onShowSimilar={() => showSimilarItems(item)}
                  selected={selectedItems.has(item.id)}
                />
              ))}
          </form>
          <Pagination
            current={page}
            pageCount={pageCount}
            totalItems={items.length}
            onChange={setPage}
            onChangePageCount={setPageCount}
          />
        </div>
      )}
    </div>
  );
};

const PAGE_COUNTS = [20, 40, 60, 80] as const;
type PageCount = typeof PAGE_COUNTS[number];

const Pagination = ({
  current,
  pageCount,
  totalItems,
  onChange,
  onChangePageCount,
}: {
  current: number;
  pageCount: number;
  totalItems: number;
  onChange: (to: number) => void;
  onChangePageCount: (count: PageCount) => void;
}) => {
  const prev = current - 1;
  const next = current + 1;

  let totalPages = (totalItems / pageCount) | 0;
  if (totalItems % pageCount == 0) totalPages--;

  return (
    <div className="inline-flex gap-5">
      <select
        className="select max-w-xs"
        onChange={(event) =>
          onChangePageCount(parseInt(event.target.value) as PageCount)
        }
        defaultValue={pageCount}
      >
        {PAGE_COUNTS.map((count) => (
          <option key={count}>{count}</option>
        ))}
      </select>
      <div className="btn-group">
        <button
          onClick={() => onChange(prev)}
          className={classy("btn", { "btn-disabled": current === 1 })}
        >
          <MdOutlineNavigateBefore />
        </button>
        {prev > 1 && (
          <>
            <button onClick={() => onChange(1)} className="btn">
              1
            </button>
            <button className="btn btn-disabled">...</button>
          </>
        )}

        {prev > 0 && (
          <button onClick={() => onChange(prev)} className="btn">
            {prev}
          </button>
        )}
        <button className="btn btn-active">{current}</button>
        {next < totalPages && (
          <button onClick={() => onChange(next)} className="btn">
            {next}
          </button>
        )}
        {next < totalPages - 1 && (
          <>
            <button className="btn btn-disabled">...</button>
            <button onClick={() => onChange(totalPages)} className="btn">
              {totalPages}
            </button>
          </>
        )}
        <button
          onClick={() => onChange(next)}
          className={classy("btn", { "btn-disabled": current === totalPages })}
        >
          <MdOutlineNavigateNext />
        </button>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const form = event.target as HTMLFormElement;
          const value = +(form[0] as HTMLInputElement).value;
          onChange(Math.min(1, Math.max(totalPages, value)));
          form.reset();
        }}
      >
        <input type="number" placeholder="Go to page" className="input w-36" />
      </form>
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
          allowClear
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
}) => (
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
        name={item.id}
        type="checkbox"
        checked={selected}
        readOnly
        className={classy(
          "peer checkbox absolute left-1 top-1 opacity-0 group-hover:opacity-100 checked:opacity-100"
        )}
      />
      {/* <figure className="group-hover:opacity-30 peer-checked:p-2 bg-base-300 rounded peer-checked:transition-none"> */}
      <figure className="group-hover:opacity-30 peer-checked:outline peer-checked:outline-offset-[-4px] peer-checked:outline-4 outline-base-300  rounded peer-checked:transition-none">
        <img
          className="object-cover rounded transition-opacity"
          src={item.url}
        />
      </figure>
      <div className="absolute flex gap-2 top-1 right-1 opacity-0 group-hover:opacity-100">
        <button onClick={onExpand} className="btn btn-square">
          <FaExpand />
        </button>
      </div>
    </label>
    <div className="card-body p-2">
      <div className="card-actions flex">
        <div className="btn-group">
          <button className="btn btn-ghost gap-2" onClick={onShowSimilar}>
            <MdImageSearch className="text-base" />
          </button>
          <button
            className="btn btn-ghost gap-2"
            onClick={() => window.open(item.editUrl.toString(), "_blank")}
          >
            <FaEdit />
          </button>
        </div>
        {item.metadata.labelClass || item.metadata.annotator ? (
          <div className="flex flex-col">
            <span className="inline-flex items-center gap-1">
              <VscSymbolClass />
              {item.metadata.labelClass}
            </span>
            <span className="inline-flex items-center gap-1">
              <RiUserLine />
              {item.metadata.annotator}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  </div>
);
const MetadataMetrics = ({
  metrics,
}: {
  metrics: Item["metadata"]["metrics"];
}) => {
  return (
    <div className="w-full flex flex-col">
      {Object.entries(metrics).map(([key, value]) => {
        const number = parseFloat(value.toString());

        return (
          <div key={key}>
            <span>{key}: </span>
            <span>{isNaN(number) ? value : number.toFixed(4)}</span>
          </div>
        );
      })}
    </div>
  );
};
