import { useEffect, useMemo, useRef, useState, useContext } from "react";
import {
  BiCloudUpload,
  BiInfoCircle,
  BiSelectMultiple,
  BiWindows,
} from "react-icons/bi";
import { BsCardText } from "react-icons/bs";
import { FaEdit, FaExpand } from "react-icons/fa";
import { MdClose, MdFilterAltOff, MdImageSearch } from "react-icons/md";
import { RiUserLine } from "react-icons/ri";
import { TbMoodSad2, TbSortAscending, TbSortDescending } from "react-icons/tb";
import { VscClearAll, VscSymbolClass } from "react-icons/vsc";
import { Spinner } from "./Spinner";
import { useQuery } from "@tanstack/react-query";
import useResizeObserver from "use-resize-observer";
import { classy } from "../../helpers/classy";
import { useDebounce } from "usehooks-ts";
import {
  ApiContext,
  classificationsPredictionOutcomes,
  getApi,
  IdValue,
  Item,
  Metric,
  objectPredictionOutcomes,
  PredictionOutcome,
  PredictionType,
  Scope,
  useApi,
} from "./api";
import { Assistant, useSearch } from "./Assistant";
import { MetricDistributionTiny, ScatteredEmbeddings } from "./Charts";
import { splitId } from "./id";
import { Pagination, usePagination } from "./Pagination";
import {
  BulkTaggingForm,
  TaggingDropdown,
  TaggingForm,
  TagList,
} from "./Tagging";
import { capitalize, isEmpty, sift } from "radash";
import {
  FilterState,
  MetricFilter,
  DefaultFilters,
} from "../util/MetricFilter";
import { Popover, Button } from "antd";
import {ProjectAnalysisDomain, ProjectMetricSummary, QueryAPI} from "../Types";
import { CreateSubsetModal } from "../tabs/modals/CreateSubsetModal";
import { UploadToEncordModal } from "../tabs/modals/UploadToEncordModal";
import { apiUrl, env, local } from "../../constants";
import { useImageSrc } from "../../hooks/useImageSrc";
import { useAuth } from "../../authContext";

export type InternalFilters = {
  analysisDomain: "data" | "annotation";
  filters: {
    data: {
      metrics: Readonly<Record<string, readonly [number, number]>>,
      enums:  Readonly<Record<string, readonly string[]>>,
      reduction: null,
      tags: null | string[],
    },
    annotation: {
      metrics: Readonly<Record<string, readonly [number, number]>>,
      enums:  Readonly<Record<string, readonly string[]>>,
      reduction: null,
      tags: null | string[],
    },
  },
  orderBy: null | string,
  desc: boolean,
  iou: number | undefined,
  predictionOutcome: PredictionOutcome | undefined,
  predictionType: "object" | "classification" | undefined,
};

export type Props = {
  projectHash: string;
  dataMetricsSummary: ProjectMetricSummary;
  annotationMetricsSummary: ProjectMetricSummary;
  scope: Scope;
  queryAPI: QueryAPI;
  featureHashMap: Parameters<typeof MetricFilter>[0]["featureHashMap"];
  setSelectedProjectHash: (projectHash: string | undefined) => void;
  remoteProject: boolean;
};

export const Explorer = ({
  projectHash,
  scope,
  queryAPI,
  featureHashMap,
  dataMetricsSummary,
  annotationMetricsSummary,
  setSelectedProjectHash,
  remoteProject,
}: Props) => {
  const [itemSet, setItemSet] = useState(new Set<string>());


  // Item selected for extra analysis operations
  const [previewedItem, setPreviewedItem] = useState<string | null>(null);
  const [similarityItem, setSimilarityItem] = useState<string | null>(null);

  // Selection
  const [selectedItems, setSelectedItems] = useState(new Set<string>());
  const [selectedMetric, setSelectedMetric] = useState<
  {
    domain: "data" | "annotation";
    metric_key: string;
  } | undefined>();

  // Prediction filters
  const [isAscending, setIsAscending] = useState(true);
  const [predictionType, setPredictionType] = useState<
    PredictionType | undefined
  >();
  const [predictionOutcome, setPredictionOutcome] = useState<
    PredictionOutcome | undefined
  >();
  const [iou, setIou] = useState<number | undefined>();
  const [dataFilters, setDataFilters] = useState<FilterState>(DefaultFilters);
  const [annotationFilters, setAnnotationFilters] = useState<FilterState>(DefaultFilters);

  const rawFilters: InternalFilters = useMemo(() => {
    const analysisDomain: ProjectAnalysisDomain = "data";
    return {
      analysisDomain,
      filters: {
        data: {
          metrics: dataFilters.metricFilters,
          enums: dataFilters.enumFilters,
          reduction: null,
          tags: null,
        },
        annotation: {
          metrics: dataFilters.metricFilters,
          enums: dataFilters.enumFilters,
          reduction: null,
          tags: null,
        },
      },
      orderBy: null,
      desc: false,
      iou,
      predictionOutcome,
      predictionType,
    }
  }, [dataFilters, annotationFilters, predictionType, predictionOutcome, iou]);

  const filters: InternalFilters = useDebounce(rawFilters, 500);

  useEffect(() => {
    setPage(1);
  }, [filters]);

  const apiContext = useContext(ApiContext);
  const { token } = useAuth();

  let api: ReturnType<typeof getApi>;
  if (apiContext == null) {
    api = getApi(projectHash, token);
  } else {
    api = apiContext;
  }

  const predictionTypeFound = scope !== "prediction" || predictionType != null;

  const { data: hasPremiumFeatures } = useQuery(
    ["hasPremiumFeatures"],
    api.fetchHasPremiumFeatures,
    { staleTime: Infinity },
  );
  /*
  const { data: hasSimilaritySearch } = useQuery(
    sift([projectHash, "hasSimilaritySearch", selectedMetric?.embeddingType]),
    () => api.fetchHasSimilaritySearch(selectedMetric?.embeddingType!),
    { enabled: !!selectedMetric, staleTime: Infinity },
  );
  */
  const hasSimilaritySearch = false;

  const similarItems: string[] | undefined = undefined;
  const isLoadingSimilarItems = false;
  /*
  FIXME:
  const { data: similarItems, isFetching: isLoadingSimilarItems } = useQuery(
    sift([projectHash, scope, "similarities", similarityItem]),
    () =>
      api.fetchSimilarItems(
        similarityItem!,
        "embedding_clip",
      ),
    { enabled: !!similarityItem && !!selectedMetric },
  );*/

  const {
    data: dataMetricRanges, isLoading: isLoadingDataMetrics
  } = queryAPI.useProjectAnalysisSummary(
    projectHash,
    "data",
  );
  const {
    data: annotationMetricRanges, isLoading: isLoadingAnnotationMetrics
  } = queryAPI.useProjectAnalysisSummary(
    projectHash,
    "annotation",
  );
  const isLoadingMetrics = isLoadingDataMetrics || isLoadingAnnotationMetrics;
  const { data: sortedItems, isLoading: isLoadingSortedItems } = queryAPI.useProjectAnalysisSearch(
    projectHash,
    filters.analysisDomain,
    filters.filters,
    filters.orderBy,
    filters.desc,
    {
      enabled: predictionTypeFound,
    },
  )

  const filterLabelClassMap = useMemo(() => {
    const res = Object.fromEntries(
      Object.values(featureHashMap).map((v) => [v.name, v]),
    );
    res["No class"] = {
      name: "No class",
      color: "",
    };
    return res;
  }, [featureHashMap]);

  const withSortOrder: readonly string[] = sortedItems?.results ?? [];

  /*
  FIXME: implement
  const {
    search,
    setSearch,
    result: searchResults,
    loading: searching,
  } = useSearch(scope, filters, api.searchInProject);
  */
  const search: any = (v: any) => {};
  const setSearch = () => {};
  const searchResults: undefined | { snippet: string | null; ids: string[] } = undefined;
  const searching = false;

  /*
  FIXME:
  const resetable =
    itemSet.size ||
    searchResults?.ids.length ||
    similarItems?.length ||
    !isEmpty(filters.range) ||
    !isEmpty([...filters.tags?.data, ...filters.tags?.label]);
   */
  const resetable = true;
  const reset = (clearFilters: boolean = true) => {
    setItemSet(new Set());
    //FIXME: setSearch(undefined);
    setSimilarityItem(null);
    if (clearFilters) setDataFilters(DefaultFilters);
    if (clearFilters) setAnnotationFilters(DefaultFilters);
    setPage(1);
  };

  const itemsToRender: string[] =
    similarItems ?? searchResults?.ids ?? withSortOrder.map(({ id }) => id);

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
  const showSimilarItems = (itemId: string) => {
    closePreview();
    setPage(1);
    setSimilarityItem(itemId);
  }

  const onMetricSelected = (newMetric: string) => {
    const [domain, metricKey] = newMetric.split("-", 1);
    let analysisDomain: "data" | "annotation";
    if (domain === "data-") {
      analysisDomain = "data"
    } else if (domain === "annotation-") {
      analysisDomain = "annotation"
    } else {
      throw Error(domain)
    }
    /*
    FIXME:
    const usedScopes = Object.entries(metrics)
      .map(([_, scopedMetrics]) => scopedMetrics.map((metric) => metric.name))
      .filter(
        (scopedMetricNames) =>
          scopedMetricNames.includes(selectedMetric.name) ||
          scopedMetricNames.includes(newMetric.name),
      )!;
    if (usedScopes.length !== 1) {
      reset(false);
    }
    */
    setSelectedMetric({domain: analysisDomain, metric_key: metricKey});
  };
  const allowTaggingAnnotations = useMemo(() => {
    if (!selectedMetric) {
      return false;
    } else {
      return selectedMetric.domain === "annotation";
    }
  }, [selectedMetric]);

  useEffect(() => {
    if (!selectedMetric) {
      setSelectedMetric({domain: "data", metric_key: "metric_random"});
    }
  }, [selectedMetric]);

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
      {
        isLoading: searching,
        description: "Searching",
      },
    ];
    return descriptions.reduce((res, item) => {
      return !res && item.isLoading ? item.description : res;
    }, "");
  }, [
    isLoadingMetrics,
    isLoadingSortedItems,
    isLoadingSimilarItems,
    searching,
  ]);
  const [open, setOpen] = useState<undefined | "subset" | "upload">();
  const close = () => setOpen(undefined);
  return (
    <>
      {/* FIXME: uncomment
      <CreateSubsetModal
        open={open == "subset"}
        close={close}
        projectHash={projectHash}
        queryAPI={queryAPI}
        filters={filters}
        ids={[...itemSet]}
      />
      */}
      <UploadToEncordModal
        open={open === "upload"}
        close={close}
        projectHash={projectHash}
        queryAPI={queryAPI}
        setSelectedProjectHash={setSelectedProjectHash}
      />
      <div className="w-full">
        {previewedItem && (
          <ItemPreview
            id={previewedItem}
            similaritySearchDisabled={!hasSimilaritySearch}
            scope={scope}
            iou={iou}
            onClose={closePreview}
            onShowSimilar={() => showSimilarItems(previewedItem)}
            allowTaggingAnnotations={allowTaggingAnnotations}
          />
        )}
        <div
          className={classy(
            "w-full flex flex-col gap-5 items-center pb-5 m-auto",
            {
              hidden: previewedItem,
            },
          )}
        >
          {/* TODO: move model predictions embeddings plot to FE */}
          {selectedMetric && false && (
            <Embeddings
              queryApi={queryAPI}
              projectHash={projectHash}
              isloadingItems={isLoadingSortedItems}
              idValues={
                (scope === "prediction"
                  ? sortedItems?.results?.map(({ id, ...item }) => ({
                      ...item,
                      id: id.slice(0, id.lastIndexOf("_")),
                    }))
                  : sortedItems) || []
              }
              filters={filters}
              embeddingType="embedding_clip"
              onSelectionChange={(selection) => {
                setPage(1);
                setItemSet(new Set(selection.map(({ id }) => id)));
              }}
              onReset={() => setItemSet(new Set())}
            />
          )}
          {similarityItem && (
            <SimilarityItem
              itemId={similarityItem}
              onClose={() => setSimilarityItem(null)}
            />
          )}
          <div className="flex w-full gap-2 flex-col flex-wrap">
            <div className="flex gap-2 flex-wrap">
              <TaggingDropdown
                disabledReason={
                  scope === "prediction"
                    ? scope
                    : !selectedItems.size
                    ? "missing-target"
                    : undefined
                }
              >
                <BulkTaggingForm
                  items={[...selectedItems]}
                  allowTaggingAnnotations={allowTaggingAnnotations}
                />
              </TaggingDropdown>
              {(
                <label className="input-group  w-auto">
                  <select
                    onChange={(event) =>
                      onMetricSelected(event.target.value)
                    }
                    className="select select-bordered focus:outline-none"
                    disabled={!!similarItems?.length}
                  >
                    <optgroup
                      label="Data Metrics"
                    >
                      {Object.entries(annotationMetricsSummary.metrics).map(
                        ([metricKey, metric]) => (
                          <option
                            key={`data-${metricKey}`}
                            value={`data-${metricKey}`}
                          >
                            {metric.title}
                          </option>
                        ),
                      )}
                    </optgroup>
                    <optgroup
                      label="Annotation Metrics"
                    >
                      {Object.entries(dataMetricsSummary.metrics).map(
                        ([metricKey, metric]) => (
                          <option
                            key={`annotation-${metricKey}`}
                            value={`annotation-${metricKey}`}
                          >
                            {metric.title}
                          </option>
                        ),
                      )}
                    </optgroup>
                  </select>
                  <label
                    className={classy("btn swap swap-rotate", {
                      "btn-disabled": !!similarItems?.length,
                    })}
                  >
                    <input
                      onChange={() => setIsAscending((prev) => !prev)}
                      type="checkbox"
                      disabled={!!similarItems?.length}
                      defaultChecked={true}
                    />
                    <TbSortAscending className="swap-on text-base" />
                    <TbSortDescending className="swap-off text-base" />
                  </label>
                </label>
              )}
              {/* FIXME: !similarityItem && scope !== "prediction" && (
                <MetricDistributionTiny
                  values={sortedItems || []}
                  setSeletedIds={(ids) => setItemSet(new Set(ids))}
                />
              ) */}
              {scope === "prediction" && (
                <PredictionFilters
                  predictionType={predictionType}
                  setPredictionType={setPredictionType}
                  onOutcomeChange={setPredictionOutcome}
                  onIouChange={setIou}
                  disabled={!!similarityItem}
                />
              )}
            </div>
            <div className="flex justify-between gap-2 flex-wrap">
              <Assistant
                defaultSearch={search}
                isFetching={searching}
                setSearch={setSearch}
                snippet={searchResults?.snippet}
                disabled={!hasPremiumFeatures}
              />
              <div className="flex gap-2 flex-wrap">
                <Popover
                  placement="bottomLeft"
                  content={
                    <MetricFilter
                      filters={dataFilters}
                      setFilters={setDataFilters}
                      metricsSummary={dataMetricsSummary}
                      metricRanges={dataMetricRanges?.metrics}
                      featureHashMap={filterLabelClassMap}
                    />
                  }
                  trigger="click"
                >
                  <button className="btn btn-ghost">Data Filters</button>
                </Popover>
                <Popover
                  placement="bottomLeft"
                  content={
                    <MetricFilter
                      filters={annotationFilters}
                      setFilters={setAnnotationFilters}
                      metricsSummary={annotationMetricsSummary}
                      metricRanges={annotationMetricRanges?.metrics}
                      featureHashMap={filterLabelClassMap}
                    />
                  }
                  trigger="click"
                >
                  <button className="btn btn-ghost">Annotation Filters</button>
                </Popover>
                <button
                  className={classy("btn btn-ghost gap-2", {
                    "btn-disabled": !resetable,
                  })}
                  onClick={() => reset()}
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
                {env !== "sandbox" && (
                  <button
                    className="btn btn-ghost gap-2"
                    onClick={() => setOpen("subset")}
                    disabled={!resetable}
                  >
                    <BiWindows />
                    Create Project subset
                  </button>
                )}
                {local && !remoteProject && (
                  <button
                    className="btn btn-ghost gap-2"
                    onClick={() => setOpen("upload")}
                    disabled={!!resetable}
                  >
                    <BiCloudUpload />
                    Upload project
                  </button>
                )}
              </div>
            </div>
          </div>
          {!!loadingDescription ? (
            <div className="h-32 flex items-center gap-2">
              <Spinner />
              <span className="text-xl">{loadingDescription}</span>
            </div>
          ) : itemsToRender.length ? (
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
                    iou={iou}
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
          ) : (
            <div className="h-32 flex items-center gap-2">
              <TbMoodSad2 className="text-3xl" />
              <span className="text-xl">No results</span>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

const ALL_PREDICTION_OUTCOMES = "All Prediction Outcomes";

const PredictionFilters = ({
  predictionType,
  setPredictionType,
  onOutcomeChange,
  onIouChange,
  disabled = false,
}: {
  predictionType?: PredictionType;
  setPredictionType: (type: PredictionType) => void;
  onOutcomeChange: (predictionOutcome?: PredictionOutcome) => void;
  onIouChange?: (iou: number) => void;
  disabled?: boolean;
}) => {
  const { data: predictionTypes, isLoading } =
    useApi().fetchAvailablePredictionTypes();

  const [iou, setIou] = useState(0.5);
  const [drag, setDrag] = useState(false);

  useEffect(() => {
    if (!predictionType && predictionTypes?.length)
      setPredictionType(predictionTypes[0]);
  }, [predictionTypes]);

  useEffect(() => {
    if (iou != null && !drag) onIouChange?.(iou);
  }, [iou, drag]);

  if (isLoading) return <Spinner />;
  if (!predictionTypes?.length) return null;

  const outcomes =
    predictionType === "object"
      ? objectPredictionOutcomes
      : classificationsPredictionOutcomes;

  return (
    <div className="flex gap-2">
      {predictionTypes?.length > 1 && (
        <select
          disabled={disabled}
          className="select select-bordered w-full max-w-xs"
          onChange={({ target: { value } }) =>
            setPredictionType(value as PredictionType)
          }
        >
          {predictionTypes.map((type) => (
            <option key={type} value={type}>
              {`${capitalize(type)} Predictions`}
            </option>
          ))}
        </select>
      )}
      <select
        disabled={disabled}
        className="select select-bordered w-full max-w-xs"
        onChange={({ target: { value } }) =>
          onOutcomeChange(
            value === ALL_PREDICTION_OUTCOMES
              ? undefined
              : (value as PredictionOutcome),
          )
        }
      >
        {[ALL_PREDICTION_OUTCOMES, ...outcomes].map((outcome) => (
          <option key={outcome}>{outcome}</option>
        ))}
      </select>
      {predictionType === "object" && (
        <div className="form-control w-full max-w-xs min-w-[256px]">
          <label>
            <span>IOU: {iou}</span>
          </label>
          <input
            disabled={disabled}
            type="range"
            min={0.01}
            max={1}
            step={0.01}
            defaultValue={iou}
            onMouseUp={() => setDrag(false)}
            onMouseDown={() => setDrag(true)}
            onChange={({ target: { value } }) => setIou(parseFloat(value))}
            className="range range-xs"
          />
        </div>
      )}
    </div>
  );
};

const Embeddings = ({
  queryApi,
  projectHash,
  isloadingItems,
  idValues,
  filters,
  embeddingType,
  onSelectionChange,
  onReset,
}: {
  queryApi: QueryAPI;
  projectHash: string;
  isloadingItems: boolean;
  idValues: IdValue[];
  filters: InternalFilters;
  embeddingType: Metric["embeddingType"];
  onSelectionChange: Parameters<
    typeof ScatteredEmbeddings
  >[0]["onSelectionChange"];
  onReset: () => void;
}) => {
  const {
    data: reductionHashes,
  } = queryApi.useProjectListEmbeddingReductions(projectHash);
  const reductionHash: string | undefined = useMemo(
    () => Object.keys(reductionHashes?.results ?? {})[0],
    [reductionHashes]
  );
  const {
    isLoading,
    data: scatteredEmbeddings
  } = queryApi.useProjectAnalysisReducedEmbeddings(
    projectHash,
    filters.analysisDomain,
    reductionHash ?? "",
    filters.filters,
    { enabled: reductionHash != null}
  );

  const filtered = useMemo(() => {
    const ids = new Set(idValues.map(({ id }) => id));
    return scatteredEmbeddings?.filter(
      ({ id }) => ids.has(id) || ids.has(id.slice(0, id.lastIndexOf("_"))),
    );
  }, [JSON.stringify(idValues), JSON.stringify(scatteredEmbeddings)]);

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
        <div className="absolute" style={{ left: "50%" }}>
          <Spinner />
        </div>
      ) : (
        <ScatteredEmbeddings
          embeddings={filtered || []}
          onSelectionChange={onSelectionChange}
          onReset={onReset}
          predictionType={filters.prediction_filters?.type}
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
  scope,
  iou,
  onClose,
  onShowSimilar,
  allowTaggingAnnotations = false,
}: {
  id: string;
  similaritySearchDisabled: boolean;
  scope: Scope;
  onClose: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
  iou?: number;
  allowTaggingAnnotations: boolean;
}) => {
  const { data, isLoading } = useApi().fetchItem(id, iou);
  const { mutate } = useApi().itemTagsMutation;

  if (isLoading || !data) return <Spinner />;

  const { description, ...metrics } = data.metadata.metrics;
  const { editUrl } = data;
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
            onClick={() =>
              editUrl ? window.open(editUrl, "_blank") : undefined
            }
            disabled={editUrl == null}
          >
            <FaEdit />
            Edit
          </button>
          <TaggingDropdown
            disabledReason={scope === "prediction" ? scope : undefined}
          >
            <TaggingForm
              onChange={(groupedTags) => mutate([{ id, groupedTags }])}
              selectedTags={data.tags}
              tabIndex={0}
              allowTaggingAnnotations={allowTaggingAnnotations}
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
  iou,
}: {
  itemId: string;
  selected: boolean;
  selectedMetric?: Metric;
  similaritySearchDisabled: boolean;
  onExpand: JSX.IntrinsicElements["button"]["onClick"];
  onShowSimilar: JSX.IntrinsicElements["button"]["onClick"];
  iou?: number;
}) => {
  const { data, isLoading } = useApi().fetchItem(itemId, iou);

  if (isLoading || !data)
    return (
      <div className="w-full h-full flex justify-center items-center min-h-[230px]">
        <Spinner />
      </div>
    );

  const [metricName, value] = Object.entries(data.metadata.metrics).find(
    ([metric, _]) =>
      metric.toLowerCase() === selectedMetric?.name.toLowerCase(),
  ) || [selectedMetric?.name, ""];
  const [intValue, floatValue] = [parseInt(value), parseFloat(value)];
  const displayValue =
    intValue === floatValue ? intValue : parseFloat(value).toFixed(4);
  const { description } = data.metadata.metrics;
  const { editUrl } = data;

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
              data-tip={
                editUrl
                  ? "Open in Encord Annotate"
                  : "Upload to Encord to edit annotations"
              }
              onClick={() =>
                editUrl ? window.open(editUrl.toString(), "_blank") : null
              }
              disabled={editUrl == null}
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

const getObjects = (item: Item) => {
  const { objectHash } = splitId(item.id);
  const object = item.labels.objects.find(
    (object) => object.objectHash === objectHash,
  );
  const prediction = item.predictions.objects.find(
    (object) => object.objectHash === objectHash,
  );

  if (object) return [object];

  return prediction
    ? [...item.labels.objects, prediction]
    : item.labels.objects;
};

type ItemLabelObject = Item["labels"]["objects"][0];

const pointsRecordToPolygonPoints = (
  points: NonNullable<ItemLabelObject["points"]>,
  width: number,
  height: number,
) =>
  Object.values(points)
    .map(({ x, y }) => `${x * width},${y * height}`)
    .join(" ");

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
    Pick<ItemLabelObject, "points" | "boundingBoxPoints" | "shape" | "color">[]
  >([]);

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

  const itemUrl = item.url.startsWith("http")
    ? item.url
    : `${apiUrl}${item.url}`;

  const { data: imgSrcUrl, isLoading } = useImageSrc(itemUrl);

  if (isLoading) return <Spinner />;

  return (
    <figure {...rest} className={classy("relative", className)}>
      {item.videoTimestamp != null ? (
        <video
          ref={video}
          className="object-contain rounded transition-opacity"
          src={imgSrcUrl}
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
          src={imgSrcUrl}
        />
      )}
      {width && height && polygons.length > 0 && (
        <svg className="absolute w-full h-full top-0 right-0">
          {polygons.map(
            ({ points, boundingBoxPoints, color, shape }, index) => {
              if (shape === "point" && points)
                return (
                  <g key={index}>
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
                  </g>
                );
              return (
                <g key={index} fill={shape === "polyline" ? "none" : color}>
                  {points && (
                    <polygon
                      key={index + "_polygon"}
                      style={{
                        fillOpacity: ".20",
                        stroke: color,
                        strokeWidth: "2px",
                      }}
                      points={pointsRecordToPolygonPoints(
                        points,
                        width,
                        height,
                      )}
                    />
                  )}
                  {boundingBoxPoints && (
                    <polygon
                      key={index + "_box"}
                      style={{
                        fillOpacity: ".40",
                        stroke: color,
                        strokeWidth: "4px",
                      }}
                      points={pointsRecordToPolygonPoints(
                        boundingBoxPoints,
                        width,
                        height,
                      )}
                    />
                  )}
                </g>
              );
            },
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
