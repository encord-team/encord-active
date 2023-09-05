import { useMemo, useState, useContext } from "react";
import {
  BiCloudUpload,
  BiSelectMultiple,
  BiWindows,
} from "react-icons/bi";
import { FaEdit } from "react-icons/fa";
import { MdClose, MdFilterAltOff, MdImageSearch } from "react-icons/md";
import { TbMoodSad2, TbSortAscending, TbSortDescending } from "react-icons/tb";
import { VscClearAll } from "react-icons/vsc";
import { useQuery } from "@tanstack/react-query";
import { useDebounce } from "usehooks-ts";
import {
  ApiContext,
  getApi,
  Item,
  useApi,
} from "./api";
import { Assistant } from "./Assistant";
import { splitId } from "./id";
import {
  BulkTaggingForm,
  TaggingDropdown,
  TaggingForm,
  TagList,
} from "./Tagging";
import {
  FilterState,
  MetricFilter,
  DefaultFilters,
} from "../util/MetricFilter";
import {Button, List, Popover, Select, Slider, Space, Spin} from "antd";
import {ProjectAnalysisDomain, ProjectMetricSummary, QueryAPI} from "../Types";
import { UploadToEncordModal } from "../tabs/modals/UploadToEncordModal";
import { env, local } from "../../constants";
import { useAuth } from "../../authContext";
import {ExplorerEmbeddings} from "./ExplorerEmbeddings";
import {CreateSubsetModal} from "../tabs/modals/CreateSubsetModal";
import {MetricDistributionTiny} from "./ExplorerCharts";
import {ExplorerGalleryItem} from "./ExplorerGalleryItem";
import {HiOutlineTag} from "react-icons/hi";
import {loadingIndicator} from "../Spin";
import {ImageWithPolygons} from "./ImageWithPolygons";

export type InternalFilters = {
  readonly analysisDomain: "data" | "annotation";
  readonly filters: {
    readonly data: {
      readonly metrics: Readonly<Record<string, readonly [number, number]>>,
      readonly enums: Readonly<Record<string, readonly string[]>>,
      readonly reduction: null,
      readonly tags: null | readonly string[],
    },
    readonly annotation: {
      readonly metrics: Readonly<Record<string, readonly [number, number]>>,
      readonly enums: Readonly<Record<string, readonly string[]>>,
      readonly reduction: null,
      readonly tags: null | readonly string[],
    },
  },
  readonly orderBy: string,
  readonly desc: boolean,
  readonly iou: number | undefined,
  readonly predictionOutcome: "tp" | "fp" | "fn" | undefined,
  readonly predictionHash: string | undefined,
};

export type Props = {
  projectHash: string;
  predictionHash: string | undefined;
  dataMetricsSummary: ProjectMetricSummary;
  annotationMetricsSummary: ProjectMetricSummary;
  scope: "analysis" | "prediction";
  queryAPI: QueryAPI;
  featureHashMap: Parameters<typeof MetricFilter>[0]["featureHashMap"];
  setSelectedProjectHash: (projectHash: string | undefined) => void;
  remoteProject: boolean;
};

export const Explorer = ({
  projectHash,
  predictionHash,
  scope,
  queryAPI,
  featureHashMap,
  dataMetricsSummary,
  annotationMetricsSummary,
  setSelectedProjectHash,
  remoteProject,
}: Props) => {
  // Item selected for extra analysis operations
  const [previewedItem, setPreviewedItem] = useState<string | null>(null);
  const [similarityItem, setSimilarityItem] = useState<string | null>(null);

  // Select reduction hash
  const {
    data: reductionHashes,
  } = queryAPI.useProjectListEmbeddingReductions(projectHash);
  const reductionHash: string | undefined = useMemo(
    () => reductionHashes === undefined
        || reductionHashes.results.length === 0
        ? undefined : reductionHashes.results[0].hash,
    [reductionHashes]
  );


  // Selection
  const [selectedItems, setSelectedItems] = useState(new Set<string>());
  const [selectedMetric, setSelectedMetric] = useState<
  {
    domain: "data" | "annotation";
    metric_key: string;
  }>({
    domain: "data", metric_key: "metric_random"
  });

  // Filter State
  const [isAscending, setIsAscending] = useState(true);
  const [predictionOutcome, setPredictionOutcome] = useState<
    "tp" | "fp" | "fn"
  >("tp");
  const [iou, setIou] = useState<number>(0.5);
  const [dataFilters, setDataFilters] = useState<FilterState>(DefaultFilters);
  const [annotationFilters, setAnnotationFilters] = useState<FilterState>(DefaultFilters);
  const canResetFilters = predictionOutcome !== "tp" || iou !== 0.5
    || dataFilters.ordering.length !== 0 || annotationFilters.ordering.length !== 0;
  const resetAllFilters = () => {
    setIsAscending(true);
    setPredictionOutcome("tp");
    setIou(0.5);
    setDataFilters(DefaultFilters);
    setAnnotationFilters(DefaultFilters);
  }

  const rawFilters: InternalFilters = useMemo(() => {
    return {
      analysisDomain: selectedMetric.domain,
      filters: {
        data: {
          metrics: dataFilters.metricFilters,
          enums: dataFilters.enumFilters,
          reduction: null,
          tags: null,
        },
        annotation: {
          metrics: annotationFilters.metricFilters,
          enums: annotationFilters.enumFilters,
          reduction: null,
          tags: null,
        },
      },
      orderBy: selectedMetric.metric_key,
      desc: !isAscending,
      iou,
      predictionOutcome,
      predictionHash,
    }
  }, [selectedMetric, dataFilters, isAscending, annotationFilters, predictionHash, predictionOutcome, iou]);

  const filters: InternalFilters = useDebounce(rawFilters, 500);

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
    { staleTime: Infinity },
  );


  //// START OF SIMILARITY SEARCH.
  /*
  const { data: hasSimilaritySearch } = useQuery(
    sift([projectHash, "hasSimilaritySearch", selectedMetric?.embeddingType]),
    () => api.fetchHasSimilaritySearch(selectedMetric?.embeddingType!),
    { enabled: !!selectedMetric, staleTime: Infinity },
  );
  */
  const hasSimilaritySearch = false;
  const { data: similarItems, isLoading: isLoadingSimilarItemsRaw} = useQuery(
    ["FIXME: IMPLEMENT PROPERLY"],
    () => [] as string[],
    { enabled: hasSimilaritySearch && similarityItem !== undefined }
  );
  const isLoadingSimilarItems = isLoadingSimilarItemsRaw && hasSimilaritySearch && similarityItem !== undefined;

  // Load metric ranges
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
      enabled: scope !== "prediction",
    },
  )

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
  const searchResults: undefined | { snippet: string | null; ids: string[] } = 1 == 1 ? undefined : { snippet: "", ids: []};
  const searching = false;

  const reset = (clearFilters: boolean = true) => {
    setSimilarityItem(null);
    if (clearFilters) resetAllFilters();
  };

  const itemsToRender: readonly string[] =
    similarItems ?? searchResults?.ids ?? sortedItems?.results ?? [];

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
    setSimilarityItem(itemId);
  }

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
    return descriptions.reduce((res, item) => {
      return !res && item.isLoading ? item.description : res;
    }, "");
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
        queryAPI={queryAPI}
        filters={filters}
      />
      <UploadToEncordModal
        open={open === "upload"}
        close={close}
        projectHash={projectHash}
        queryAPI={queryAPI}
        setSelectedProjectHash={setSelectedProjectHash}
      />
      <ExplorerEmbeddings
        queryApi={queryAPI}
        projectHash={projectHash}
        reductionHash={reductionHash}
        filters={filters}
        setEmbeddingSelection={() => {/*FIXME*/}}
      />
      {previewedItem && (
        <ItemPreview
          queryAPI={queryAPI}
          projectHash={projectHash}
          id={previewedItem}
          similaritySearchDisabled={!hasSimilaritySearch}
          scope={scope}
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
      {!similarityItem && scope !== "prediction" && (
        <MetricDistributionTiny
          projectHash={projectHash}
          queryAPI={queryAPI}
          filters={filters}
        />
      )}
      {scope === "prediction" && (
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
              setSelectedMetric({ domain: domain as "data" | "annotation", metric_key});
            }}
            style={{width: 300}}
            options={[
              {
                label: 'Data Metrics',
                options: Object.entries(dataMetricsSummary.metrics).map(([metricKey, metric]) => ({
                  label: `D: ${metric.title}`,
                  value: `data-${metricKey}`
                })),
              },
              {
                label: 'Annotation Metrics',
                options: Object.entries(annotationMetricsSummary.metrics).map(([metricKey, metric]) => ({
                  label: `A: ${metric.title}`,
                  value: `annotation-${metricKey}`
                })),
              },
            ]}
          />
          <Button
            onClick={() => setIsAscending(!isAscending)}
            icon={isAscending ? <TbSortAscending/> : <TbSortDescending/>}
          />
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
            <Button icon={<HiOutlineTag />} disabled={!selectedItems.size}>Tag</Button>
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
        style={{marginTop: 10}}
        dataSource={itemsToRender as string[]}
        grid={{}}
        loading={{
          spinning: loadingDescription != "",
          tip: loadingDescription,
          indicator: loadingIndicator
        }}
        locale={{
          emptyText: "No Results"
        }}
        pagination={{
          defaultPageSize: 10,
        }}
        renderItem={(item: string) => {
          return <ExplorerGalleryItem
            projectHash={projectHash}
            queryAPI={queryAPI}
            selectedMetric={selectedMetric}
            key={item}
            itemId={item}
            onExpand={() => setPreviewedItem(item)}
            similaritySearchDisabled={!hasSimilaritySearch}
            onShowSimilar={() => showSimilarItems(item)}
            selected={selectedItems.has(item)}
            iou={iou}
          />
        }}
      />
    </div>
  );
};
const PredictionFilters = ({
  iou,
  setIou,
  predictionOutcome,
  setPredictionOutcome,
  isClassificationOnly,
  disabled,
}: {
  iou: number;
  setIou: (iou: number) => void;
  predictionOutcome: "fp" | "fn" | "tp";
  setPredictionOutcome: (outcome: "fp" | "fn" | "tp") => void;
  isClassificationOnly: boolean;
  disabled: boolean;
}) => {
  return (
    <div className="flex gap-2">
      <Select
        disabled={disabled}
        onChange={setPredictionOutcome}
        value={predictionOutcome}
        options={[{
          key: "fp",
          label: "False Positive"
        },{
          key: "tp",
          label: "True Positive"
        },{
          key: "fn",
          label: "False Negative"
        }]}
      />
      {!isClassificationOnly && (
        <Slider
          value={iou}
          onChange={setIou}
          min={0.0}
          max={0.0}
          step={0.01}
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
  queryAPI,
  projectHash,
  id,
  similaritySearchDisabled,
  scope,
  iou,
  onClose,
  onShowSimilar,
  allowTaggingAnnotations = false,
}: {
  queryAPI: QueryAPI,
  projectHash: string,
  id: string;
  similaritySearchDisabled: boolean;
  scope: "prediction" | "analytics";
  onClose: () => void;
  onShowSimilar: () => void;
  iou?: number;
  allowTaggingAnnotations: boolean;
}) => {
  const { du_hash, frame, annotation_hash} = splitId(id);
  const { data: preview, isLoading } = queryAPI.useProjectItemPreview(
    projectHash,
    du_hash,
    frame,
    annotation_hash,
  );
  const { data: info } = queryAPI.useProjectItemDetails(
    projectHash,
    du_hash,
    frame,
  )
  const mutate = () => console.log('fixme');

  if (isLoading || !preview) return <Spin indicator={loadingIndicator} />;

  /*const { description, ...metrics } = preview.metadata.metrics;
  const { editUrl } = data;*/
  const editUrl = "FIXME";
  const description = "";
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
              selectedTags={{ data: [], label: []}} // FIXME:
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
              <span>{info?.data_title ?? "unknown"}</span>
            </div>
            {description && (
              <div>
                <span>Description: </span>
                <span>{description}</span>
              </div>
            )}
          </div>
          {/*<MetadataMetrics metrics={metrics} />
          <TagList tags={data.tags} />*/}
        </div>
        <div className="w-fit inline-block relative">
          <ImageWithPolygons className="" preview={preview} />
        </div>
      </div>
    </div>
  );
};



const getObjects = (item: Item) => {
  const { annotation_hash } = splitId(item.id);
  const object = item.labels.objects.find(
    (object) => object.objectHash === annotation_hash,
  );
  const prediction = item.predictions.objects.find(
    (object) => object.objectHash === annotation_hash,
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
