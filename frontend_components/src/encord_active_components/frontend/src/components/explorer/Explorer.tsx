import { Select } from "antd";
import { Column, Histogram } from "@ant-design/plots";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FaExpand, FaEdit } from "react-icons/fa";
import { HiOutlineTag } from "react-icons/hi";
import { BiSelectMultiple } from "react-icons/bi";
import {
  MdClose,
  MdImageSearch,
  MdOutlineImage,
  MdOutlineNavigateBefore,
  MdOutlineNavigateNext,
} from "react-icons/md";
import { TbPolygon, TbSortAscending, TbSortDescending } from "react-icons/tb";
import { RiUserLine } from "react-icons/ri";
import { VscClearAll, VscSymbolClass } from "react-icons/vsc";

import { Streamlit } from "streamlit-component-lib";

import useResizeObserver from "use-resize-observer";
import { classy } from "../../helpers/classy";
import { useQueries, useQuery } from "@tanstack/react-query";
import { IdParts, splitId } from "./id";
import {
  BASE_URL,
  EmbeddingType,
  fetchProjectItem,
  fetchProjectItemIds,
  fetchProjectMetrics,
  getSimilarItems,
  GroupedTags,
  IdValue,
  Item,
  Point,
  Scope,
} from "./api";
import { MetricDistribution } from "./Charts";

type Output = never;

const pushOutput = (output: Output) => Streamlit.setComponentValue(output);

export type Props = {
  projectName: string;
  items: string[];
  embeddingsType: EmbeddingType;
  tags: GroupedTags;
  scope: Scope;
};

export const Explorer = ({
  projectName,
  items,
  tags,
  scope,
  embeddingsType,
}: Props) => {
  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();
  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const itemSet = useMemo(() => new Set(items), []);

  const [previewedItem, setPreviewedItem] = useState<string | null>(null);
  const [similarityItem, setSimilarityItem] = useState<string | null>(null);

  const [selectedItems, setSelectedItems] = useState(new Set<string>());

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<PageSize>(PAGE_SIZE[0]);

  const [selectedMetric, setSelectedMetric] = useState<string>();

  const toggleImageSelection = (id: Item["id"]) => {
    setSelectedItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const closePreview = () => setPreviewedItem(null);
  const showSimilarItems = (itemId: string) => (
    closePreview(), setPage(1), setSimilarityItem(itemId)
  );

  const { data: metrics } = useQuery(["metrics"], () =>
    fetchProjectMetrics(projectName)(scope)
  );

  useEffect(() => {
    if (!selectedMetric && metrics && metrics?.length > 0)
      setSelectedMetric(metrics[0]);
  }, [metrics?.length]);

  const { data: sortedItems, refetch } = useQuery(
    ["item_ids"],
    () => fetchProjectItemIds(projectName)(selectedMetric!),
    { enabled: !!selectedMetric }
  );

  useEffect(() => {
    selectedMetric && refetch?.();
  }, [selectedMetric]);

  const { data: similarItems } = useQuery(
    ["similarities", similarityItem ?? ""],
    () => getSimilarItems(projectName)(similarityItem!, embeddingsType),
    { enabled: !!similarityItem }
  );

  const [sortedAndFiltered, setSortedAndFiltered] = useState<IdValue[]>([]);

  useEffect(() => {
    setSortedAndFiltered(
      sortedItems?.filter(({ id }) => itemSet.has(id)) || []
    );
  }, [itemSet, sortedItems]);

  const fetchItem = useCallback(fetchProjectItem(projectName), [
    fetchProjectItem,
    projectName,
  ]);

  const pageItems = (
    similarItems ?? sortedAndFiltered.map(({ id }) => id)
  ).slice(page * pageSize, page * pageSize + pageSize);

  return (
    <div ref={ref} className="w-full flex">
      {previewedItem ? (
        <ItemPreview
          itemId={previewedItem}
          fetchItem={fetchItem}
          tags={tags}
          onClose={closePreview}
          onShowSimilar={() => showSimilarItems(previewedItem)}
        />
      ) : (
        <div className="w-full flex flex-col gap-5 items-center pb-5">
          {sortedAndFiltered && (
            <div className="w-full h-52">
              <MetricDistribution
                values={sortedAndFiltered.map(({ value }) => value)}
              />
            </div>
          )}
          {similarityItem && (
            <SimilarityItem
              itemId={similarityItem}
              fetchItem={fetchItem}
              onClose={() => setSimilarityItem(null)}
            />
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
            <div className="form-control">
              {metrics && metrics?.length > 0 ? (
                <label className="input-group">
                  <span>Sort by</span>
                  <select
                    onChange={(event) => setSelectedMetric(event.target.value)}
                    className="select select-bordered focus:outline-none"
                  >
                    {metrics.map((metric) => (
                      <option key={metric}>{metric}</option>
                    ))}
                  </select>
                  <label className="btn swap swap-rotate">
                    <input
                      onChange={() =>
                        setSortedAndFiltered((prev) => [...prev].reverse())
                      }
                      type="checkbox"
                      defaultChecked={true}
                    />
                    <TbSortAscending className="swap-on text-base" />
                    <TbSortDescending className="swap-off text-base" />
                  </label>
                </label>
              ) : null}
            </div>
            <div className="flex gap-2">
              <button
                className="btn btn-ghost gap-2"
                onClick={() => setSelectedItems(new Set(items))}
              >
                <BiSelectMultiple />
                Select all
              </button>
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
          </div>
          <form
            onChange={({ target }) =>
              toggleImageSelection((target as HTMLInputElement).name)
            }
            onSubmit={(e) => e.preventDefault()}
            className="w-full flex-1 grid gap-1 grid-cols-2 lg:grid-cols-4 2xl:grid-cols-5"
          >
            {pageItems.map((id) => (
              <GalleryItem
                key={id}
                itemId={id}
                onExpand={() => setPreviewedItem(id)}
                onShowSimilar={() => showSimilarItems(id)}
                selected={selectedItems.has(id)}
                fetchItem={fetchItem}
              />
            ))}
          </form>
          <Pagination
            current={page}
            pageSize={pageSize}
            totalItems={items.length}
            onChange={setPage}
            onChangePageSize={setPageSize}
          />
        </div>
      )}
    </div>
  );
};

const PAGE_SIZE = [20, 40, 60, 80] as const;
type PageSize = typeof PAGE_SIZE[number];

const Pagination = ({
  current,
  pageSize,
  totalItems,
  onChange,
  onChangePageSize,
}: {
  current: number;
  pageSize: number;
  totalItems: number;
  onChange: (to: number) => void;
  onChangePageSize: (size: PageSize) => void;
}) => {
  const prev = current - 1;
  const next = current + 1;

  let totalPages = (totalItems / pageSize) | 0;
  if (totalItems % pageSize == 0) totalPages--;

  return (
    <div className="inline-flex gap-5">
      <select
        className="select max-w-xs"
        onChange={(event) =>
          onChangePageSize(parseInt(event.target.value) as PageSize)
        }
        defaultValue={pageSize}
      >
        {PAGE_SIZE.map((size) => (
          <option key={size}>{size}</option>
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
        onSubmit={async (event) => {
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

const SimilarityItem = ({
  itemId,
  onClose,
  fetchItem,
}: {
  itemId: string;
  fetchItem: ReturnType<typeof fetchProjectItem>;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { data, isLoading } = useQuery(["item", itemId], () =>
    fetchItem(itemId)
  );

  if (isLoading || !data) return null;

  return (
    <div className="flex flex-col gap-2">
      <h1 className="text-lg">Similar items</h1>
      <div className="group max-w-xs relative">
        <ImageWithPolygons className="group-hover:opacity-30" item={data} />
        <button
          onClick={onClose}
          className="btn btn-square absolute top-1 right-1 opacity-0 group-hover:opacity-100"
        >
          <MdClose className="text-base" />
        </button>
      </div>
    </div>
  );
};

const ItemPreview = ({
  itemId,
  onClose,
  onShowSimilar,
  tags,
  fetchItem,
}: {
  itemId: string;
  fetchItem: ReturnType<typeof fetchProjectItem>;
  tags: GroupedTags;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { data, isLoading } = useQuery(["item", itemId], () =>
    fetchItem(itemId)
  );

  if (isLoading || !data) return null;
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
            onClick={() => window.open(data.editUrl, "_blank")}
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
      <div className="inline-block relative">
        <ImageWithPolygons item={data} />
      </div>
      <MetadataMetrics metrics={data.metadata.metrics} />
    </div>
  );
};

const GalleryItem = ({
  itemId,
  selected,
  fetchItem,
  onExpand,
  onShowSimilar,
}: {
  itemId: string;
  selected: boolean;
  fetchItem: ReturnType<typeof fetchProjectItem>;
  onExpand: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { data, isLoading } = useQuery(["item", itemId], () =>
    fetchItem(itemId)
  );

  if (isLoading || !data) return null;

  return (
    <div className="card relative align-middle form-control">
      <label className="h-full group label cursor-pointer p-0">
        <input
          name={itemId}
          type="checkbox"
          checked={selected}
          readOnly
          className={classy(
            "peer checkbox absolute left-2 top-2 opacity-0 group-hover:opacity-100 checked:opacity-100 checked:z-10"
          )}
        />
        <div className="bg-gray-100 p-1 flex justify-center items-center w-full h-full peer-checked:opacity-100 peer-checked:outline peer-checked:outline-offset-[-4px] peer-checked:outline-4 outline-base-300  rounded checked:transition-none">
          <ImageWithPolygons className="group-hover:opacity-30" item={data} />
          <div className="absolute flex gap-2 top-1 right-1 opacity-0 group-hover:opacity-100">
            <button
              onClick={(e) => (console.log(data), onExpand?.(e))}
              className="btn btn-square"
            >
              <FaExpand />
            </button>
          </div>
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
              onClick={() => window.open(data.editUrl.toString(), "_blank")}
            >
              <FaEdit />
            </button>
          </div>
          {data.metadata.labelClass || data.metadata.annotator ? (
            <div className="flex flex-col">
              <span className="inline-flex items-center gap-1">
                <VscSymbolClass />
                {data.metadata.labelClass}
              </span>
              <span className="inline-flex items-center gap-1">
                <RiUserLine />
                {data.metadata.annotator}
              </span>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

const ImageWithPolygons = ({
  item,
  className,
  ...rest
}: { item: Item } & JSX.IntrinsicElements["figure"]) => {
  const image = useRef<HTMLImageElement>(null);
  const [polygons, setPolygons] = useState<
    { points: Point[]; color: string }[]
  >([]);

  useEffect(() => {
    const { current } = image;
    if (!current || !current.clientWidth || !current.clientHeight) return;

    const { objectHash } = splitId(item.id);
    const objects = objectHash
      ? item.labels.objects.filter((object) => object.objectHash === objectHash)
      : item.labels.objects;

    setPolygons(
      objects.map(({ polygon, color }) => ({
        color,
        points: Object.values(polygon).map(({ x, y }) => ({
          x: x * current.clientWidth,
          y: y * current.clientHeight,
        })),
      }))
    );
  }, [image.current?.clientWidth, image.current?.clientHeight]);

  return (
    <figure {...rest} className={classy("relative", className)}>
      <img
        ref={image}
        className="object-contain rounded transition-opacity"
        src={item.url}
      />
      {polygons.length > 0 ? (
        <svg className="absolute w-full h-full top-0 right-0">
          {polygons.map(({ points, color }, index) => (
            <polygon
              key={index}
              style={{
                fill: color,
                fillOpacity: ".20",
                stroke: color,
                strokeWidth: "2px",
              }}
              points={points.map(({ x, y }) => `${x},${y}`).join(" ")}
            />
          ))}
        </svg>
      ) : null}
    </figure>
  );
};

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
