import { Spin } from "./Spinner";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FaExpand, FaEdit } from "react-icons/fa";
import { TbMoodSad2 } from "react-icons/tb";
import { BiInfoCircle, BiSelectMultiple } from "react-icons/bi";
import { MdClose, MdFilterAltOff, MdImageSearch } from "react-icons/md";
import { TbSortAscending, TbSortDescending } from "react-icons/tb";
import { BsCardText } from "react-icons/bs";
import { RiUserLine } from "react-icons/ri";
import { VscClearAll, VscSymbolClass } from "react-icons/vsc";

import { Streamlit } from "streamlit-component-lib";

import useResizeObserver from "use-resize-observer";
import { classy } from "../../helpers/classy";
import { useQuery } from "@tanstack/react-query";
import { splitId } from "./id";
import {
  DEFAULT_BASE_URL,
  getApi,
  IdValue,
  Item,
  Point,
  ApiContext,
  Scope,
  useApi,
  Metric,
} from "./api";
import { MetricDistributionTiny, ScatteredEmbeddings } from "./Charts";
import { Pagination, usePagination } from "./Pagination";
import {
  BulkTaggingForm,
  TaggingDropdown,
  TaggingForm,
  TagList,
} from "./Tagging";
import { Assistant } from "./Assistant";

export type Props = {
  authToken: string | null;
  projectName: string;
  items: string[];
  scope: Scope;
  baseUrl: string;
};

export const Explorer = ({
  authToken,
  projectName,
  items,
  scope,
  baseUrl = DEFAULT_BASE_URL,
}: Props) => {
  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();
  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const [itemSet, setItemSet] = useState(new Set(items));

  const [previewedItem, setPreviewedItem] = useState<string | null>(null);
  const [similarityItem, setSimilarityItem] = useState<string | null>(null);

  const [selectedItems, setSelectedItems] = useState(new Set<string>());
  const [selectedMetric, setSelectedMetric] = useState<Metric>();

  const [sortedAndFiltered, setSortedAndFiltered] = useState<IdValue[]>([]);

  const api = getApi(projectName, authToken, baseUrl);

  const { data: hasPremiumFeatures } = useQuery(
    ["hasPremiumFeatures"],
    api.fetchHasPremiumFeatures,
    { staleTime: Infinity }
  );
  const { data: hasSimilaritySearch } = useQuery(
    ["hasSimilaritySearch"],
    () => api.fetchHasSimilaritySearch(selectedMetric?.embeddingType!),
    { enabled: !!selectedMetric }
  );

  const { data: similarItems, isLoading: isLoadingSimilarItems } = useQuery(
    ["similarities", similarityItem ?? ""],
    () =>
      api.fetchSimilarItems(similarityItem!, selectedMetric?.embeddingType!),
    { enabled: !!similarityItem && !!selectedMetric }
  );
  const { data: sortedItems, isLoading: isLoadingSortedItems } = useQuery(
    ["item_ids", selectedMetric],
    () => api.fetchProjectItemIds(selectedMetric?.name!),
    { enabled: !!selectedMetric, staleTime: Infinity }
  );
  const { data: metrics, isLoading: isLoadingMetrics } = useQuery(
    ["metrics"],
    () => api.fetchProjectMetrics(scope),
    { staleTime: Infinity }
  );

  const itemsToRender = similarItems ?? sortedAndFiltered.map(({ id }) => id);
  const { pageSize, pageItems, page, setPage, setPageSize } =
    usePagination(itemsToRender);

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

  useEffect(() => {
    if (!selectedMetric && metrics && metrics?.length > 0)
      setSelectedMetric(metrics[0]);
  }, [metrics?.length]);

  useEffect(() => {
    if (isLoadingSortedItems) return;
    setSortedAndFiltered(
      sortedItems?.filter(({ id }) => itemSet.has(id)) || []
    );
  }, [isLoadingSortedItems, itemSet, sortedItems]);

  const loadingDescription = useMemo(() => {
    const descriptions = [
      {
        isLoading: isLoadingMetrics,
        description: "Loading available metrics",
      },
      {
        isLoading: isLoadingSortedItems,
        description: "Loading available data",
      },
      {
        isLoading: isLoadingSimilarItems,
        description: "Finding similar images",
      },
    ];
    return descriptions.reduce((res, item) => {
      return !res && item.isLoading ? item.description : res;
    }, "");
  }, [isLoadingMetrics, isLoadingSortedItems, isLoadingSimilarItems]);

  return (
    <ApiContext.Provider value={api}>
      <div ref={ref} className="w-full">
        {previewedItem && (
          <ItemPreview
            id={previewedItem}
            similaritySearchDisabled={!hasSimilaritySearch}
            onClose={closePreview}
            onShowSimilar={() => showSimilarItems(previewedItem)}
          />
        )}
        <div
          className={classy(
            "w-full flex flex-col gap-5 items-center pb-5 m-auto",
            {
              hidden: previewedItem,
            }
          )}
        >
          {selectedMetric && (
            <Embeddings
              isloadingItems={isLoadingSortedItems}
              idValues={sortedAndFiltered}
              embeddingType={selectedMetric.embeddingType}
              onSelectionChange={(selection) => (
                setPage(1), setItemSet(new Set(selection.map(({ id }) => id)))
              )}
              onReset={() => setItemSet(new Set(items))}
            />
          )}
          {similarityItem && (
            <SimilarityItem
              itemId={similarityItem}
              onClose={() => setSimilarityItem(null)}
            />
          )}
          <div className="flex w-full justify-between gap-5 flex-wrap">
            <div className="flex gap-2 max-w-[50%]">
              <TaggingDropdown
                className={classy({ "btn-disabled": !selectedItems.size })}
              >
                <BulkTaggingForm items={[...selectedItems]} />
              </TaggingDropdown>
              {metrics && metrics?.length && (
                <>
                  <label className="input-group">
                    <select
                      onChange={(event) =>
                        setSelectedMetric(
                          metrics.find(
                            (metric) => metric.name === event.target.value
                          )
                        )
                      }
                      className="select select-bordered focus:outline-none"
                      disabled={!!similarItems?.length}
                    >
                      {metrics.map(({ name }) => (
                        <option key={name}>{name}</option>
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
                  {!similarityItem && (
                    <MetricDistributionTiny
                      values={sortedAndFiltered || []}
                      setSeletedIds={(ids) => setItemSet(new Set(ids))}
                    />
                  )}
                </>
              )}
            </div>
            <div className="flex gap-2">
              <button
                className={classy("btn btn-ghost gap-2", {
                  "btn-disabled":
                    itemsToRender.length ===
                    (similarItems || sortedItems)?.length,
                })}
                onClick={() => setItemSet(new Set(items))}
              >
                <MdFilterAltOff />
                Reset filters
              </button>
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
                onClick={() => setSelectedItems(new Set(itemsToRender))}
              >
                <BiSelectMultiple />
                Select all ({itemsToRender.length})
              </button>
            </div>
          </div>
          <Assistant
            scope={scope}
            disabled={!hasPremiumFeatures}
            setResults={(ids) => {
              const idValues = new Map(
                sortedAndFiltered.map(({ id, value }) => [id, value])
              );
              setSortedAndFiltered(
                ids.map((id) => ({ id, value: idValues.get(id) || 0 }))
              );
            }}
          />
          {itemsToRender.length ? (
            <>
              <form
                onChange={({ target }) =>
                  toggleImageSelection((target as HTMLInputElement).name)
                }
                onSubmit={(e) => e.preventDefault()}
                className="w-full flex-1 grid gap-1 grid-cols-2 lg:grid-cols-4 2xl:grid-cols-5"
              >
                {pageItems.map((id) => (
                  <GalleryItem
                    selectedMetric={selectedMetric}
                    key={id}
                    itemId={id}
                    onExpand={() => setPreviewedItem(id)}
                    similaritySearchDisabled={!hasSimilaritySearch}
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
            </>
          ) : !!loadingDescription ? (
            <div className="h-32 flex items-center gap-2">
              <Spin />
              <span className="text-xl">{loadingDescription}</span>
            </div>
          ) : (
            <div className="h-32 flex items-center gap-2">
              <TbMoodSad2 className="text-3xl" />
              <span className="text-xl">No results</span>
            </div>
          )}
        </div>
      </div>
    </ApiContext.Provider>
  );
};

const Embeddings = ({
  isloadingItems,
  idValues,
  embeddingType,
  onSelectionChange,
  onReset,
}: {
  isloadingItems: boolean;
  idValues: IdValue[];
  embeddingType: Metric["embeddingType"];
  onSelectionChange: Parameters<
    typeof ScatteredEmbeddings
  >[0]["onSelectionChange"];
  onReset: () => void;
}) => {
  const { isLoading, data: scatteredEmbeddings } =
    useApi().fetch2DEmbeddings(embeddingType);

  const filtered = useMemo(() => {
    const ids = new Set(idValues.map(({ id }) => id));
    return scatteredEmbeddings?.filter(
      ({ id }) => ids.has(id) || ids.has(id.slice(0, id.lastIndexOf("_")))
    );
  }, [idValues, scatteredEmbeddings]);

  return !isLoading && !isloadingItems && !scatteredEmbeddings?.length ? (
    <div className="alert shadow-lg h-fit">
      <div>
        <BiInfoCircle className="text-base" />
        <span>2D embedding are not available for this project. </span>
      </div>
    </div>
  ) : (
    <div className="w-full flex  h-96 [&>*]:flex-1 items-center">
      {isLoading || isloadingItems ? (
        <Spin />
      ) : (
        <ScatteredEmbeddings
          embeddings={filtered || []}
          onSelectionChange={onSelectionChange}
          onReset={onReset}
        />
      )}
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
  const { data, isLoading } = useApi().fetchItem(itemId);

  if (isLoading || !data) return null;

  return (
    <div className="flex flex-col gap-2">
      <h1 className="text-lg">Similar items</h1>
      <div className="group max-w-xs relative">
        <ImageWithPolygons className="group-hover:opacity-20" item={data} />
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
  similaritySearchDisabled,
  onClose,
  onShowSimilar,
}: {
  id: string;
  similaritySearchDisabled: boolean;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { data, isLoading } = useApi().fetchItem(id);
  const { mutate } = useApi().itemTagsMutation;

  if (isLoading || !data) return <Spin />;

  const { description, ...metrics } = data.metadata.metrics;

  return (
    <div className="w-full flex flex-col items-center gap-3 p-1">
      <div className="w-full flex justify-between">
        <div className="flex gap-3">
          <button
            className="btn btn-ghost gap-2"
            disabled={similaritySearchDisabled}
            onClick={onShowSimilar}
          >
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
          <div className="flex flex-col">
            <div>
              <span>Title: </span>
              <span>{data.dataTitle || "unknown"}</span>
            </div>
            {description && (
              <div>
                <span>Description: </span>
                <span>{description}</span>
              </div>
            )}
          </div>
          <MetadataMetrics metrics={metrics} />
          <TagList tags={data.tags} />
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
  selectedMetric,
  similaritySearchDisabled,
  onExpand,
  onShowSimilar,
}: {
  itemId: string;
  selected: boolean;
  selectedMetric?: Metric;
  similaritySearchDisabled: boolean;
  onExpand: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const { data, isLoading } = useApi().fetchItem(itemId);

  if (isLoading || !data)
    return (
      <div className="w-full h-full flex justify-center items-center min-h-[230px]">
        <Spin />
      </div>
    );

  const [metricName, value] = Object.entries(data.metadata.metrics).find(
    ([metric, _]) => metric.toLowerCase() === selectedMetric?.name.toLowerCase()
  ) || [selectedMetric?.name, ""];
  const [intValue, floatValue] = [parseInt(value), parseFloat(value)];
  const displayValue =
    intValue === floatValue ? intValue : parseFloat(value).toFixed(4);
  const { description } = data.metadata.metrics;

  return (
    <div className="card relative align-middle bg-gray-100 form-control min-h-[230px]">
      <label className="relative h-full group label cursor-pointer p-0 not-last:z-10 not-last:opacity-0">
        <input
          name={itemId}
          type="checkbox"
          checked={selected}
          readOnly
          className="peer checkbox absolute left-2 top-2 checked:!opacity-100 group-hover:opacity-100"
        />
        {selectedMetric && (
          <div className="absolute top-2 group-hover:opacity-100 w-full flex justify-center gap-1">
            <span>{metricName}:</span>
            <span>{displayValue}</span>
          </div>
        )}
        <div className="absolute p-2 top-7 pb-8 group-hover:opacity-100 w-full h-5/6 flex flex-col gap-3 overflow-y-auto">
          <TagList tags={data.tags} />
          {description && (
            <div className="flex flex-col">
              <div className="inline-flex items-center gap-1">
                <BsCardText className="text-base" />
                <span>Description:</span>
              </div>
              <span>{description}</span>
            </div>
          )}
        </div>
        <div className="bg-gray-100 p-1 flex justify-center items-center w-full h-full peer-checked:opacity-100 peer-checked:outline peer-checked:outline-offset-[-4px] peer-checked:outline-4 outline-base-300  rounded checked:transition-none">
          <ImageWithPolygons className="group-hover:opacity-30" item={data} />
          <div className="absolute flex gap-2 top-1 right-1 opacity-0 group-hover:opacity-100">
            <button
              onClick={(e) => onExpand?.(e)}
              className="btn btn-square z-20"
            >
              <FaExpand />
            </button>
          </div>
        </div>
      </label>
      <div className="divider m-0"></div>
      <div className="card-body p-2">
        <div className="card-actions flex">
          <div className="btn-group">
            <button
              className="btn btn-ghost gap-2 tooltip tooltip-right"
              data-tip="Similar items"
              disabled={similaritySearchDisabled}
              onClick={onShowSimilar}
            >
              <MdImageSearch className="text-base" />
            </button>
            <button
              className="btn btn-ghost gap-2 tooltip tooltip-right"
              data-tip="Open in Encord Annotate"
              onClick={() => window.open(data.editUrl.toString(), "_blank")}
            >
              <FaEdit />
            </button>
          </div>
          {data.metadata.labelClass || data.metadata.annotator ? (
            <div className="flex flex-col">
              <span className="flex items-center gap-1">
                <VscSymbolClass />
                {data.metadata.labelClass}
              </span>
              <span className="flex items-center gap-1">
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
  const width = item.videoTimestamp != null ? videoWidth : imageWidth;
  const height = item.videoTimestamp != null ? videoHeight : imageHeight;
  const [polygons, setPolygons] = useState<
    { points: Point[]; color: string; shape: string }[]
  >([]);

  useEffect(() => {
    if (width == null || height == null) return;

    const { objectHash } = splitId(item.id);
    const objects = objectHash
      ? item.labels.objects.filter((object) => object.objectHash === objectHash)
      : item.labels.objects;

    setPolygons(
      objects.map(({ points, color, shape }) => ({
        color,
        points: Object.values(points).map(({ x, y }) => ({
          x: x * width,
          y: y * height,
        })),
        shape,
      }))
    );
  }, [width, height, item.id]);
  return (
    <figure {...rest} className={classy("relative", className)}>
      {item.videoTimestamp != null ? (
        <video
          ref={video}
          className="object-contain rounded transition-opacity"
          src={item.url}
          muted
          controls={false}
          onLoadedMetadata={() => {
            const videoRef = video.current;
            if (videoRef != null) {
              videoRef.currentTime = item.videoTimestamp || 0;
            }
          }}
        />
      ) : (
        <img
          ref={image}
          className="object-contain rounded transition-opacity"
          alt=""
          src={item.url}
        />
      )}
      {polygons.length > 0 && (
        <svg className="absolute w-full h-full top-0 right-0">
          {polygons.map(({ points, color, shape }, index) =>
            shape === "point" ? (
              <>
                <circle
                  key={index + "_inner"}
                  cx={points[0].x}
                  cy={points[0].y}
                  r="5px"
                  fill={color}
                />
                <circle
                  key={index + "_outer"}
                  cx={points[0].x}
                  cy={points[0].y}
                  r="7px"
                  fill="none"
                  stroke={color}
                  strokeWidth="1px"
                />
              </>
            ) : (
              <polygon
                key={index}
                style={{
                  fill: shape === "polyline" ? "none" : color,
                  fillOpacity: ".20",
                  stroke: color,
                  strokeWidth: "2px",
                }}
                points={points.map(({ x, y }) => `${x},${y}`).join(" ")}
              />
            )
          )}
        </svg>
      )}
    </figure>
  );
};

const MetadataMetrics = ({
  metrics,
}: {
  metrics: Item["metadata"]["metrics"];
}) => {
  const entries = Object.entries(metrics);
  entries.sort();

  return (
    <div className="flex flex-col">
      {entries.map(([key, value]) => {
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
