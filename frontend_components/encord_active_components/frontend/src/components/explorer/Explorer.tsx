import { useMemo, useState, useContext, useEffect } from "react";
import { BiCloudUpload, BiSelectMultiple, BiWindows } from "react-icons/bi";
import { MdClose, MdFilterAltOff } from "react-icons/md";
import { TbSortAscending, TbSortDescending } from "react-icons/tb";
import { VscClearAll } from "react-icons/vsc";
import { useQuery } from "@tanstack/react-query";
import { useDebounce } from "usehooks-ts";
import { Button, List, Popover, Select, Slider, Space } from "antd";
import { HiOutlineTag } from "react-icons/hi";
import { ApiContext, getApi, Item, useApi } from "./api";
import { Assistant } from "./Assistant";
import { splitId } from "./id";
import { BulkTaggingForm } from "./Tagging";
import {
  FilterState,
  MetricFilter,
  DefaultFilters,
} from "../util/MetricFilter";
import { UploadToEncordModal } from "../tabs/modals/UploadToEncordModal";
import { env, local } from "../../constants";
import { useAuth } from "../../authContext";
import { ExplorerEmbeddings } from "./ExplorerEmbeddings";
import { CreateSubsetModal } from "../tabs/modals/CreateSubsetModal";
import { MetricDistributionTiny } from "./ExplorerCharts";
import { GalleryCard } from "../preview/GalleryCard";
import { loadingIndicator } from "../Spin";
import {
  DomainSearchFilters,
  Embedding2DFilter,
  PredictionDomain,
  ProjectDomainSummary,
} from "../../openapi/api";
import { QueryContext } from "../../hooks/Context";
import { useProjectListReductions } from "../../hooks/queries/useProjectListReductions";
import { useProjectAnalysisSummary } from "../../hooks/queries/useProjectAnalysisSummary";
import { useProjectAnalysisSearch } from "../../hooks/queries/useProjectAnalysisSearch";
import { useProjectAnalysisSimilaritySearch } from "../../hooks/queries/useProjectAnalysisSimilaritySearch";
import { ExplorerFilterState } from "./ExplorerTypes";
import { AnnotatedImage } from "../preview/AnnotatedImage";
import { ItemPreviewModal } from "../preview/ItemPreviewModal";
import { usePredictionAnalysisSearch } from "../../hooks/queries/usePredictionAnalysisSearch";

export type Props = {
  projectHash: string;
  predictionHash: string | undefined;
  dataMetricsSummary: ProjectDomainSummary;
  annotationMetricsSummary: ProjectDomainSummary;
  queryContext: QueryContext;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  featureHashMap: Parameters<typeof MetricFilter>[0]["featureHashMap"];
  setSelectedProjectHash: (projectHash: string | undefined) => void;
  remoteProject: boolean;
};

export function Explorer({
  projectHash,
  predictionHash,
  queryContext,
  editUrl,
  featureHashMap,
  dataMetricsSummary,
  annotationMetricsSummary,
  setSelectedProjectHash,
  remoteProject,
}: Props) {
  // Item selected for extra analysis operations
  const [previewedItem, setPreviewedItem] = useState<string | null>(null);
  const [similarityItem, setSimilarityItem] = useState<string | null>(null);

  // Select reduction hash
  const { data: reductionHashes } = useProjectListReductions(
    queryContext,
    projectHash
  );
  const reductionHash: string | undefined = useMemo(
    () =>
      reductionHashes === undefined || reductionHashes.results.length === 0
        ? undefined
        : reductionHashes.results[0].hash,
    [reductionHashes]
  );

  // Selection
  const [selectedItems, setSelectedItems] = useState(new Set<string>());
  const [selectedMetric, setSelectedMetric] = useState<{
    domain: "data" | "annotation";
    metric_key: string;
  }>({
    domain: "data",
    metric_key: "metric_random",
  });
  const [hideExtraAnnotations, setHideExtraAnnotations] = useState(false);
  useEffect(() => {
    setHideExtraAnnotations(selectedMetric.domain === "annotation");
  }, [selectedMetric.domain]);

  // Filter State
  const [isAscending, setIsAscending] = useState(true);
  const [predictionOutcome, setPredictionOutcome] =
    useState<PredictionDomain>("tp");
  const [iou, setIou] = useState<number>(0.5);
  const [dataFilters, setDataFilters] = useState<FilterState>(DefaultFilters);
  const [annotationFilters, setAnnotationFilters] =
    useState<FilterState>(DefaultFilters);
  const [embeddingFilter, setEmbeddingFilter] = useState<
    Embedding2DFilter | undefined
  >();
  const canResetFilters =
    predictionOutcome !== "tp" ||
    iou !== 0.5 ||
    dataFilters.ordering.length !== 0 ||
    annotationFilters.ordering.length !== 0 ||
    embeddingFilter !== undefined;
  const resetAllFilters = () => {
    setIsAscending(true);
    setPredictionOutcome("tp");
    setIou(0.5);
    setDataFilters(DefaultFilters);
    setAnnotationFilters(DefaultFilters);
    setEmbeddingFilter(undefined);
  };

  const rawFilters: ExplorerFilterState = useMemo(
    () => ({
      analysisDomain: selectedMetric.domain,
      filters: {
        data: {
          // FIXME: the 'as' casts should NOT! be needed
          metrics: dataFilters.metricFilters as DomainSearchFilters["metrics"],
          enums: dataFilters.enumFilters as DomainSearchFilters["enums"],
          reduction:
            selectedMetric.domain === "data" ? embeddingFilter : undefined,
          tags: undefined,
        },
        annotation: {
          metrics:
            annotationFilters.metricFilters as DomainSearchFilters["metrics"],
          enums: annotationFilters.enumFilters as DomainSearchFilters["enums"],
          reduction:
            selectedMetric.domain === "annotation"
              ? embeddingFilter
              : undefined,
          tags: undefined,
        },
      },
      orderBy: selectedMetric.metric_key,
      desc: !isAscending,
      iou,
      predictionOutcome,
      predictionHash,
    }),
    [
      selectedMetric,
      dataFilters,
      isAscending,
      annotationFilters,
      predictionHash,
      predictionOutcome,
      iou,
    ]
  );

  const filters: ExplorerFilterState = useDebounce(rawFilters, 500);

  const apiContext = useContext(ApiContext);
  const { token } = useAuth();

  let apiLegacy: ReturnType<typeof getApi>;
  if (apiContext == null) {
    apiLegacy = getApi(projectHash, token);
  } else {
    apiLegacy = apiContext;
  }

  const { data: hasPremiumFeatures } = useQuery(
    ["hasPremiumFeatures"],
    apiLegacy.fetchHasPremiumFeatures,
    { staleTime: Infinity }
  );

  /// / START OF SIMILARITY SEARCH.
  /*
  const { data: hasSimilaritySearch } = useQuery(
    sift([projectHash, "hasSimilaritySearch", selectedMetric?.embeddingType]),
    () => api.fetchHasSimilaritySearch(selectedMetric?.embeddingType!),
    { enabled: !!selectedMetric, staleTime: Infinity },
  );
  */
  const hasSimilaritySearch = false;
  const { data: similarItems, isLoading: isLoadingSimilarItemsRaw } =
    useProjectAnalysisSimilaritySearch(
      queryContext,
      projectHash,
      filters.analysisDomain,
      similarityItem ?? "",
      { enabled: hasSimilaritySearch && similarityItem !== undefined }
    );
  const isLoadingSimilarItems =
    isLoadingSimilarItemsRaw &&
    hasSimilaritySearch &&
    similarityItem !== undefined;

  // Load metric ranges
  const { data: dataMetricRanges, isLoading: isLoadingDataMetrics } =
    useProjectAnalysisSummary(queryContext, projectHash, "data");
  const {
    data: annotationMetricRanges,
    isLoading: isLoadingAnnotationMetrics,
  } = useProjectAnalysisSummary(queryContext, projectHash, "annotation");
  const isLoadingMetrics = isLoadingDataMetrics || isLoadingAnnotationMetrics;
  const { data: sortedItemsProject, isLoading: isLoadingSortedItemsProject } =
    useProjectAnalysisSearch(
      queryContext,
      projectHash,
      filters.analysisDomain,
      filters.orderBy,
      filters.desc,
      0,
      1000,
      filters.filters,
      {
        enabled: predictionHash === undefined,
      }
    );
  const {
    data: sortedItemsPrediction,
    isLoading: isLoadingSortedItemsPrediction,
  } = usePredictionAnalysisSearch(
    queryContext,
    projectHash,
    predictionHash ?? "",
    filters.predictionOutcome,
    filters.iou,
    filters.orderBy,
    filters.desc,
    0,
    1000,
    filters.filters,
    {
      enabled: predictionHash !== undefined,
    }
  );
  const sortedItems =
    predictionHash === undefined ? sortedItemsProject : sortedItemsPrediction;
  const isLoadingSortedItems =
    predictionHash === undefined
      ? isLoadingSortedItemsProject
      : isLoadingSortedItemsPrediction;

  /*
  FIXME: implement
  const {
    search,
    setSearch,
    result: searchResults,
    loading: searching,
  } = useSearch(scope, filters, api.searchInProject);
  */
  const [premiumSearch, setPremiumSearch] = useState("");
  const searchResults: undefined | { snippet: string | null; ids: string[] } =
    1 === 1 ? undefined : { snippet: "", ids: [] };
  const searching = false;

  const reset = (clearFilters: boolean = true) => {
    setSimilarityItem(null);
    if (clearFilters) {
      resetAllFilters();
    }
  };

  const itemsToRender: readonly string[] =
    similarItems?.map((v) => v.item) ??
    searchResults?.ids ??
    sortedItems?.results ??
    [];

  const toggleImageSelection = (id: string) => {
    setSelectedItems((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const closePreview = () => setPreviewedItem(null);
  const showSimilarItems = (itemId: string) => {
    closePreview();
    setSimilarityItem(itemId);
  };

  const allowTaggingAnnotations = selectedMetric?.domain === "annotation";

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

    return descriptions.reduce(
      (res, item) => (!res && item.isLoading ? item.description : res),
      ""
    );
  }, [
    isLoadingMetrics,
    isLoadingSortedItems,
    isLoadingSimilarItems,
    searching,
  ]);

  // Modal state
  const [open, setOpen] = useState<undefined | "subset" | "upload">();
  const close = () => setOpen(undefined);

  return (
    <div>
      <CreateSubsetModal
        open={open == "subset"}
        close={close}
        projectHash={projectHash}
        queryContext={queryContext}
        filters={filters.filters}
      />
      <UploadToEncordModal
        open={open === "upload"}
        close={close}
        projectHash={projectHash}
        queryContext={queryContext}
        setSelectedProjectHash={setSelectedProjectHash}
      />
      <ExplorerEmbeddings
        queryContext={queryContext}
        projectHash={projectHash}
        predictionHash={predictionHash}
        reductionHash={reductionHash}
        filters={filters}
        setEmbeddingSelection={() => {
          /* FIXME */
        }}
      />
      {previewedItem && (
        <ItemPreviewModal
          queryContext={queryContext}
          projectHash={projectHash}
          previewItem={previewedItem}
          domain={selectedMetric.domain}
          similaritySearchDisabled={!hasSimilaritySearch}
          iou={iou}
          onClose={closePreview}
          onShowSimilar={() => showSimilarItems(previewedItem)}
          allowTaggingAnnotations={allowTaggingAnnotations}
        />
      )}
      {similarityItem && (
        <SimilarityItem
          itemId={similarityItem}
          onClose={() => setSimilarityItem(null)}
        />
      )}
      {!similarityItem && predictionHash === undefined && (
        <MetricDistributionTiny
          projectHash={projectHash}
          queryContext={queryContext}
          filters={filters}
        />
      )}
      {predictionHash !== undefined && (
        <PredictionFilters
          disabled={!!similarityItem}
          iou={iou}
          setIou={setIou}
          predictionOutcome={predictionOutcome}
          isClassificationOnly={false}
          setPredictionOutcome={setPredictionOutcome}
        />
      )}
      <Assistant
        defaultSearch={premiumSearch}
        isFetching={searching}
        setSearch={setPremiumSearch}
        snippet={searchResults?.snippet}
        disabled={!hasPremiumFeatures}
      />
      <Space wrap>
        <Space.Compact size="large">
          <Select
            value={`${selectedMetric.domain}-${selectedMetric.metric_key}`}
            onChange={(strKey: string) => {
              const [domain, metric_key] = strKey.split("-");
              setSelectedMetric({
                domain: domain as "data" | "annotation",
                metric_key,
              });
            }}
            style={{ width: 300 }}
            options={[
              {
                label: "Data Metrics",
                options: Object.entries(dataMetricsSummary.metrics).map(
                  ([metricKey, metric]) => ({
                    label: `D: ${metric?.title ?? metricKey}`,
                    value: `data-${metricKey}`,
                  })
                ),
              },
              {
                label: "Annotation Metrics",
                options: Object.entries(annotationMetricsSummary.metrics).map(
                  ([metricKey, metric]) => ({
                    label: `A: ${metric?.title ?? metricKey}`,
                    value: `annotation-${metricKey}`,
                  })
                ),
              },
            ]}
          />
          <Button
            onClick={() => setIsAscending(!isAscending)}
            icon={isAscending ? <TbSortAscending /> : <TbSortDescending />}
          />
          <Button onClick={() => setHideExtraAnnotations((v) => !v)}>
            {`${hideExtraAnnotations ? "Show" : "hide"} all annotations`}
          </Button>
          <Popover
            placement="bottomLeft"
            content={
              <MetricFilter
                filters={dataFilters}
                setFilters={setDataFilters}
                metricsSummary={dataMetricsSummary}
                metricRanges={dataMetricRanges?.metrics}
                featureHashMap={featureHashMap}
              />
            }
            trigger="click"
          >
            <Button>Data Filters</Button>
          </Popover>
          <Popover
            placement="bottomLeft"
            content={
              <MetricFilter
                filters={annotationFilters}
                setFilters={setAnnotationFilters}
                metricsSummary={annotationMetricsSummary}
                metricRanges={annotationMetricRanges?.metrics}
                featureHashMap={featureHashMap}
              />
            }
            trigger="click"
          >
            <Button>Annotation Filters</Button>
          </Popover>
          <Button
            disabled={!canResetFilters}
            onClick={() => reset()}
            icon={<MdFilterAltOff />}
          >
            Reset filters
          </Button>
        </Space.Compact>
        <Space.Compact size="large">
          <Popover
            placement="bottomRight"
            content={
              <BulkTaggingForm
                items={[...selectedItems]}
                allowTaggingAnnotations={allowTaggingAnnotations}
              />
            }
            trigger="click"
          >
            <Button icon={<HiOutlineTag />} disabled={!selectedItems.size}>
              Tag
            </Button>
          </Popover>
          <Button
            disabled={!selectedItems.size}
            onClick={() => setSelectedItems(new Set())}
            icon={<VscClearAll />}
          >
            Clear selection ({selectedItems.size})
          </Button>
          <Button
            onClick={() => setSelectedItems(new Set(itemsToRender))}
            disabled={itemsToRender.length === 0}
            icon={<BiSelectMultiple />}
          >
            Select all ({itemsToRender.length})
          </Button>
        </Space.Compact>
        <Space.Compact size="large">
          <Button
            onClick={() => setOpen("subset")}
            disabled={!canResetFilters}
            hidden={env === "sandbox"}
            icon={<BiWindows />}
          >
            Create Project subset
          </Button>
          <Button
            onClick={() => setOpen("upload")}
            disabled={canResetFilters}
            hidden={remoteProject || !local}
            icon={<BiCloudUpload />}
          >
            Upload project
          </Button>
        </Space.Compact>
      </Space>
      <List
        style={{ marginTop: 10 }}
        dataSource={itemsToRender as string[]}
        grid={{}}
        loading={{
          spinning: loadingDescription != "",
          tip: loadingDescription,
          indicator: loadingIndicator,
        }}
        locale={{
          emptyText: "No Results",
        }}
        pagination={{
          defaultPageSize: 20,
        }}
        renderItem={(item: string) => (
          <GalleryCard
            projectHash={projectHash}
            predictionHash={predictionHash}
            queryContext={queryContext}
            selectedMetric={selectedMetric}
            key={item}
            itemId={item}
            onClick={() => {}}
            onExpand={() => setPreviewedItem(item)}
            similaritySearchDisabled={!hasSimilaritySearch}
            onShowSimilar={() => showSimilarItems(item)}
            selected={selectedItems.has(item)}
            hideExtraAnnotations={hideExtraAnnotations}
            editUrl={editUrl}
          />
        )}
      />
    </div>
  );
}
function PredictionFilters({
  iou,
  setIou,
  predictionOutcome,
  setPredictionOutcome,
  isClassificationOnly,
  disabled,
}: {
  iou: number;
  setIou: (iou: number) => void;
  predictionOutcome: PredictionDomain;
  setPredictionOutcome: (outcome: PredictionDomain) => void;
  isClassificationOnly: boolean;
  disabled: boolean;
}) {
  return (
    <Space.Compact>
      <Select
        disabled={disabled}
        onChange={setPredictionOutcome}
        value={predictionOutcome}
        options={[
          {
            value: "fp",
            label: "False Positive",
          },
          {
            value: "tp",
            label: "True Positive",
          },
          {
            value: "fn",
            label: "False Negative",
          },
          {
            value: "p",
            label: "All Positive",
          },
          {
            value: "a",
            label: "All Outcomes",
          },
        ]}
      />
      {!isClassificationOnly && (
        <Slider
          style={{ width: 200, paddingLeft: 10 }}
          value={iou}
          onChange={setIou}
          min={0.0}
          max={0.0}
          step={0.01}
        />
      )}
    </Space.Compact>
  );
}

function SimilarityItem({
  itemId,
  onClose,
}: {
  itemId: string;
  onClose: () => void;
}) {
  const { data, isLoading } = useApi().fetchItem(itemId);

  if (isLoading || !data) {
    return null;
  }

  return (
    <div className="flex flex-col gap-2">
      <h1 className="text-lg">Similar items</h1>
      <div className="group relative max-w-xs">
        <AnnotatedImage className="group-hover:opacity-20" item={data} />
        <button
          onClick={onClose}
          className="btn btn-square absolute top-1 right-1 opacity-0 group-hover:opacity-100"
        >
          <MdClose className="text-base" />
        </button>
      </div>
    </div>
  );
}

const getObjects = (item: Item) => {
  const { annotation_hash } = splitId(item.id);
  const object = item.labels.objects.find(
    (object) => object.objectHash === annotation_hash
  );
  const prediction = item.predictions.objects.find(
    (object) => object.objectHash === annotation_hash
  );

  if (object) {
    return [object];
  }

  return prediction
    ? [...item.labels.objects, prediction]
    : item.labels.objects;
};

type ItemLabelObject = Item["labels"]["objects"][0];

const pointsRecordToPolygonPoints = (
  points: NonNullable<ItemLabelObject["points"]>,
  width: number,
  height: number
) =>
  Object.values(points)
    .map(({ x, y }) => `${x * width},${y * height}`)
    .join(" ");

function MetadataMetrics({
  metrics,
}: {
  metrics: Item["metadata"]["metrics"];
}) {
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
}
