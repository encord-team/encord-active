import { Select, Spin, RefSelectProps, SelectProps } from "antd";
import { useCallback, useEffect, useRef, useState } from "react";
import { FaExpand, FaEdit } from "react-icons/fa";
import { HiOutlineTag } from "react-icons/hi";
import { BiInfoCircle, BiSelectMultiple } from "react-icons/bi";
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
import { useMutation, useQuery } from "@tanstack/react-query";
import { splitId } from "./id";
import {
  fetchProjectItemIds,
  fetchProjectMetrics,
  fetchProjectTags,
  defaultTags,
  fetchSimilarItems,
  GroupedTags,
  IdValue,
  Item,
  Point,
  ProjectContext,
  Scope,
  useProjectQueries,
} from "./api";
import { MetricDistribution, ScatteredEmbeddings } from "./Charts";
import { capitalize } from "radash";
import { IconType } from "react-icons";

type Output = never;

const pushOutput = (output: Output) => Streamlit.setComponentValue(output);

export type Props = {
  projectName: string;
  items: string[];
  scope: Scope;
};

export const Explorer = ({ projectName, items, scope }: Props) => {
  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();
  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const [itemSet, setItemSet] = useState(new Set(items));

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
    () => fetchSimilarItems(projectName)(similarityItem!, selectedMetric!),
    { enabled: !!similarityItem && !!selectedMetric }
  );

  const [sortedAndFiltered, setSortedAndFiltered] = useState<IdValue[]>([]);

  useEffect(() => {
    setSortedAndFiltered(
      sortedItems?.filter(({ id }) => itemSet.has(id)) || []
    );
  }, [itemSet, sortedItems]);

  const itemsToRender = similarItems ?? sortedAndFiltered.map(({ id }) => id);
  const pageItems = itemsToRender.slice(
    (page - 1) * pageSize,
    (page + 1) * pageSize + pageSize
  );

  return (
    <ProjectContext.Provider value={projectName}>
      <div ref={ref} className="w-full">
        {previewedItem && (
          <ItemPreview
            id={previewedItem}
            onClose={closePreview}
            onShowSimilar={() => showSimilarItems(previewedItem)}
          />
        )}
        <div
          className={classy("w-full flex flex-col gap-5 items-center pb-5", {
            hidden: previewedItem,
          })}
        >
          {selectedMetric && (
            <Charts
              values={sortedAndFiltered.map(({ value }) => value)}
              selectedMetric={selectedMetric}
              onSelectionChange={(selection) => (
                setPage(1), setItemSet(new Set(selection.map(({ id }) => id)))
              )}
            />
          )}
          {similarityItem && (
            <SimilarityItem
              itemId={similarityItem}
              onClose={() => setSimilarityItem(null)}
            />
          )}
          <div className="flex w-full justify-between">
            <div className="inline-flex gap-2">
              <TaggingDropdown
                className={classy({ "btn-disabled": !selectedItems.size })}
              >
                <BulkTaggingForm items={[...selectedItems]} />
              </TaggingDropdown>
              {metrics && metrics?.length > 0 ? (
                <label className="input-group">
                  <span>Sort by</span>
                  <select
                    onChange={(event) => setSelectedMetric(event.target.value)}
                    className="select select-bordered focus:outline-none"
                    disabled={!!similarItems?.length}
                  >
                    {metrics.map((metric) => (
                      <option key={metric}>{metric}</option>
                    ))}
                  </select>
                  <label
                    className={classy("btn swap swap-rotate", {
                      "btn-disabled": !!similarItems?.length,
                    })}
                  >
                    <input
                      onChange={() =>
                        setSortedAndFiltered((prev) => [...prev].reverse())
                      }
                      type="checkbox"
                      disabled={!!similarItems?.length}
                      defaultChecked={true}
                    />
                    <TbSortAscending className="swap-on text-base" />
                    <TbSortDescending className="swap-off text-base" />
                  </label>
                </label>
              ) : null}
            </div>
            <div className="inline-flex gap-2">
              <button
                className={classy("btn btn-ghost gap-2", {
                  "btn-disabled": !selectedItems.size,
                })}
                onClick={() => setSelectedItems(new Set())}
              >
                <VscClearAll />
                Clear selection ({selectedItems.size})
              </button>
              <button
                className="btn btn-ghost gap-2"
                onClick={() =>
                  setSelectedItems(new Set(sortedItems?.map(({ id }) => id)))
                }
              >
                <BiSelectMultiple />
                Select all ({itemsToRender.length})
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
              />
            ))}
          </form>
          <Pagination
            current={page}
            pageSize={pageSize}
            totalItems={itemsToRender.length}
            onChange={setPage}
            onChangePageSize={setPageSize}
          />
        </div>
      </div>
    </ProjectContext.Provider>
  );
};

const Charts = ({
  values,
  selectedMetric,
  onSelectionChange,
}: {
  values: number[];
  selectedMetric: string;
  onSelectionChange: Parameters<
    typeof ScatteredEmbeddings
  >[0]["onSelectionChange"];
}) => {
  const { isLoading, data: scatteredEmbeddings } =
    useProjectQueries().fetch2DEmbeddings(selectedMetric);

  return (
    <div className="w-full flex gap-2 h-52 [&>*]:flex-1 items-center">
      {scatteredEmbeddings ? (
        <ScatteredEmbeddings
          embeddings={scatteredEmbeddings}
          onSelectionChange={onSelectionChange}
        />
      ) : isLoading ? (
        <Spin />
      ) : (
        <div className="alert shadow-lg h-fit">
          <div>
            <BiInfoCircle className="text-base" />
            <span>2D embedding are not available for this project. </span>
          </div>
        </div>
      )}
      {values && <MetricDistribution values={values} />}
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

const TAG_GROUPS = [
  { value: "data", label: "Data", Icon: MdOutlineImage },
  { value: "label", label: "Label", Icon: TbPolygon },
] as const;

const TaggingDropdown = ({
  disabled = false,
  children,
  className,
  ...rest
}: { disabled?: boolean } & JSX.IntrinsicElements["div"]) => (
  <div
    {...rest}
    className={classy("dropdown dropdown-bottom  min-w-fit", className)}
  >
    <label
      tabIndex={0}
      className={classy("btn btn-ghost gap-2", {
        "btn-disabled": disabled,
      })}
    >
      <HiOutlineTag />
      Tag
    </label>
    {children}
  </div>
);

const BulkTaggingForm = ({ items }: { items: string[] }) => {
  const { isLoading, data: taggedItems } =
    useProjectQueries().fetchTaggedItems();
  const { mutate, isLoading: isMutating } =
    useProjectQueries().itemTagsMutation;

  const itemSet = new Set(items);
  const { data: allDataTags, label: allLabelTags } = [...taggedItems]
    .filter(([id, _]) => itemSet.has(id))
    .map(([_, tags]) => tags)
    .reduce(
      (allTags, { data, label }) => (
        data.forEach(allTags.data.add, allTags.data),
        label.forEach(allTags.label.add, allTags.label),
        allTags
      ),
      { data: new Set<string>(), label: new Set<string>() }
    );

  return (
    <TaggingForm
      loading={isLoading || isMutating}
      controlled={true}
      allowClear={false}
      onSelect={(scope, selected) =>
        mutate(
          items.map((id) => {
            const itemTags = taggedItems.get(id);
            return {
              id,
              groupedTags: itemTags
                ? { ...itemTags, [scope]: [...itemTags[scope], selected] }
                : { ...defaultTags, [scope]: [selected] },
            };
          })
        )
      }
      onDeselect={(scope, deselected) =>
        mutate(
          items.reduce((payload, id) => {
            const itemPreviousTags = taggedItems.get(id);
            if (
              itemPreviousTags &&
              itemPreviousTags[scope].includes(deselected)
            ) {
              payload.push({
                id,
                groupedTags: {
                  ...itemPreviousTags,
                  [scope]: itemPreviousTags[scope].filter(
                    (tag) => tag !== deselected
                  ),
                },
              });
            }
            return payload;
          }, [] as Parameters<typeof mutate>[0])
        )
      }
      seletedTags={{ data: [...allDataTags], label: [...allLabelTags] }}
    />
  );
};

const TaggingForm = ({
  seletedTags: selectedTags,
  className,
  controlled = false,
  loading = false,
  onChange,
  onSelect,
  onDeselect,
  allowClear = true,
  ...rest
}: {
  seletedTags: GroupedTags;
  controlled?: boolean;
  loading?: boolean;
  onChange?: (tags: GroupedTags) => void;
  onDeselect?: (scope: keyof GroupedTags, tag: string) => void;
  onSelect?: (scope: keyof GroupedTags, tag: string) => void;
  allowClear?: SelectProps["allowClear"];
} & Omit<JSX.IntrinsicElements["div"], "onChange" | "onSelect">) => {
  const { data: allTags } = useProjectQueries().fetchProjectTags();

  const [selectedTab, setTab] = useState<typeof TAG_GROUPS[number]>(
    TAG_GROUPS[0]
  );

  // NOTE: hack to prevent loosing focus when loading
  const ref = useRef<HTMLDivElement>(null);
  if (loading) ref.current && ref.current.focus();

  return (
    <div
      {...rest}
      tabIndex={0}
      className={classy(
        "dropdown-content card card-compact w-64 p-2 shadow bg-base-100 text-primary-content",
        className
      )}
    >
      <div className="tabs flex justify-center bg-base-100">
        {TAG_GROUPS.map((group) => (
          <a
            key={group.value}
            className={classy("tab tab-bordered gap-2", {
              "tab-active": selectedTab.label == group.label,
            })}
            onClick={() => setTab(group)}
          >
            <group.Icon className="text-base" />
            {group.label}
          </a>
        ))}
      </div>
      <div ref={ref} tabIndex={-1} className="card-body">
        {loading ? (
          <Spin />
        ) : (
          TAG_GROUPS.map(({ value }) => (
            <Select
              key={value}
              className={classy({
                hidden: value !== selectedTab.value,
              })}
              loading={loading}
              mode="tags"
              placeholder="Tags"
              allowClear={allowClear}
              onChange={(tags) =>
                onChange?.({ ...selectedTags, [value]: tags })
              }
              onDeselect={(tag: string) => onDeselect?.(selectedTab.value, tag)}
              onSelect={(tag: string) => onSelect?.(selectedTab.value, tag)}
              options={allTags[value].map((tag) => ({ value: tag }))}
              {...(controlled
                ? { value: selectedTags[value] }
                : { defaultValue: selectedTags[value] })}
            />
          ))
        )}
      </div>
    </div>
  );
};

const SimilarityItem = ({
  itemId,
  onClose,
}: {
  itemId: string;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { data, isLoading } = useProjectQueries().fetchItem(itemId);

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
  id,
  onClose,
  onShowSimilar,
}: {
  id: string;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { data, isLoading } = useProjectQueries().fetchItem(id);
  const { mutate } = useProjectQueries().itemTagsMutation;

  if (isLoading || !data) return <Spin />;

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
          <TaggingDropdown>
            <TaggingForm
              onChange={(groupedTags) => mutate([{ id, groupedTags }])}
              seletedTags={data.tags}
              tabIndex={0}
            />
          </TaggingDropdown>
        </div>
        <button onClick={onClose} className="btn btn-square btn-outline">
          <MdClose className="text-base" />
        </button>
      </div>
      <div className="w-full flex justify-between">
        <div className="flex flex-col gap-5">
          <MetadataMetrics metrics={data.metadata.metrics} />
          <div className="flex flex-col gap-3">
            {TAG_GROUPS.map((group) => (
              <div key={group.value}>
                <div className="inline-flex items-center gap-1">
                  <group.Icon className="text-base" />
                  <span>{group.label} tags:</span>
                </div>
                <div className="flex-wrap">
                  {data.tags[group.value].length ? (
                    data.tags[group.value].map((tag, index) => (
                      <span key={index} className="badge">
                        {tag}
                      </span>
                    ))
                  ) : (
                    <span>None</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="w-fit inline-block relative">
          <ImageWithPolygons item={data} />
        </div>
      </div>
    </div>
  );
};

const GalleryItem = ({
  itemId,
  selected,
  onExpand,
  onShowSimilar,
}: {
  itemId: string;
  selected: boolean;
  onExpand: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  {
  }
  useProjectQueries();
  const { data, isLoading } = useProjectQueries().fetchItem(itemId);
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
            <button onClick={(e) => onExpand?.(e)} className="btn btn-square">
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
  }, [image.current?.clientWidth, image.current?.clientHeight, item.id]);

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
    <div className="flex flex-col">
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
