import { useMemo, useState, useEffect, useCallback } from "react";
import { BiCloudUpload, BiSelectMultiple, BiWindows } from "react-icons/bi";
import { MdFilterAltOff } from "react-icons/md";
import { TbSortAscending, TbSortDescending } from "react-icons/tb";
import { VscClearAll } from "react-icons/vsc";
import { useDebounce, useToggle } from "usehooks-ts";
import TableIcon from "../../../assets/table.svg";
import EmbeddingsIcon from "../../../assets/dot-chart.svg";
import {
  Button,
  Col,
  Modal,
  Popover,
  Row,
  Segmented,
  Select,
  Slider,
  Space,
  Tabs,
  Tooltip,
  Typography,
} from "antd";
import { HiOutlineTag } from "react-icons/hi";
import { useNavigate, useParams } from "react-router";
import { BulkTaggingForm } from "./Tagging";
import {
  FilterState,
  MetricFilter,
  DefaultFilters,
} from "../util/MetricFilter";
import { UploadToEncordModal } from "../tabs/modals/UploadToEncordModal";
import { env, local } from "../../constants";
import { ExplorerEmbeddings } from "./ExplorerEmbeddings";
import { CreateSubsetModal } from "../tabs/modals/CreateSubsetModal";
import { ExplorerDistribution } from "./ExplorerDistribution";
import {
  DomainSearchFilters,
  Embedding2DFilter,
  PredictionDomain,
  ProjectDomainSummary,
} from "../../openapi/api";
import { useProjectListReductions } from "../../hooks/queries/useProjectListReductions";
import { useProjectAnalysisSummary } from "../../hooks/queries/useProjectAnalysisSummary";
import { useProjectAnalysisSearch } from "../../hooks/queries/useProjectAnalysisSearch";
import { ExplorerFilterState } from "./ExplorerTypes";
import { ItemPreviewModal } from "../preview/ItemPreviewModal";
import { usePredictionAnalysisSearch } from "../../hooks/queries/usePredictionAnalysisSearch";
import {
  ExplorerPremiumSearch,
  useExplorerPremiumSearch,
} from "./ExplorerPremiumSearch";
import { ExplorerSearchResults } from "./ExplorerSearchResults";
import { useProjectListCollaborators } from "../../hooks/queries/useProjectListCollaborators";
import { useProjectListTags } from "../../hooks/queries/useProjectListTags";
import { FeatureHashMap } from "../Types";
import { classy } from "../../helpers/classy";
import { Content } from "antd/es/layout/layout";
import Icon from "@ant-design/icons/lib/components/Icon";
import { InfoCircleOutlined } from "@ant-design/icons";

export type Props = {
  projectHash: string;
  predictionHash: string | undefined;
  dataMetricsSummary: ProjectDomainSummary;
  annotationMetricsSummary: ProjectDomainSummary;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  featureHashMap: FeatureHashMap;
  setSelectedProjectHash: (projectHash: string | undefined) => void;
  remoteProject: boolean;
};

export function Explorer({
  projectHash,
  predictionHash,
  editUrl,
  featureHashMap,
  dataMetricsSummary,
  annotationMetricsSummary,
  setSelectedProjectHash,
  remoteProject,
}: Props) {
  // Item selected for extra analysis operations
  const [similarityItem, setSimilarityItem] = useState<string | undefined>();

  const navigate = useNavigate();
  const { previewItem } = useParams<{ previewItem?: string }>();
  const navigateBase =
    predictionHash === undefined
      ? `/projects/${projectHash}/explorer`
      : `/projects/${projectHash}/predictions`;
  const setPreviewedItem = useCallback(
    (id?: string | undefined) =>
      id === undefined
        ? navigate(navigateBase)
        : navigate(`${navigateBase}/${encodeURIComponent(id)}`),
    [navigate, navigateBase]
  );

  // Select reduction hash
  const { data: reductionHashes, isLoading: reductionHashLoading } =
    useProjectListReductions(projectHash);
  const reductionHash: string | undefined = useMemo(
    () =>
      reductionHashes === undefined || reductionHashes.results.length === 0
        ? undefined
        : reductionHashes.results[0].hash,
    [reductionHashes]
  );

  // Selection
  const [selectedItems, setSelectedItems] = useState<
    ReadonlySet<string> | "ALL"
  >(new Set<string>());
  const [selectedMetric, setSelectedMetric] = useState<{
    domain: "data" | "annotation";
    metric_key: string;
  }>({
    domain: "data",
    metric_key: "metric_random",
  });
  const hasSelectedItems = selectedItems === "ALL" || selectedItems.size > 0;
  const [clearSelectionModalVisible, setClearSelectionModalVisible] =
    useState(false);
  useEffect(() => {
    // Similarity item does not work between similarity domains.
    setSimilarityItem(undefined);
    // Selection clear modal.
    if (hasSelectedItems) {
      setClearSelectionModalVisible(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMetric.domain]);

  // Set show animations view state.
  const [showAnnotations, toggleShowAnnotations, setShowAnnotations] =
    useToggle(true);
  useEffect(() => {
    setShowAnnotations(selectedMetric.domain === "annotation");
  }, [selectedMetric.domain, setShowAnnotations]);

  // Filter State
  const [isAscending, setIsAscending] = useState(true);
  const [predictionOutcome, setPredictionOutcome] =
    useState<PredictionDomain>("p");
  const [iou, setIou] = useState<number>(0.5);
  const [dataFilters, setDataFilters] = useState<FilterState>(DefaultFilters);
  const [annotationFilters, setAnnotationFilters] =
    useState<FilterState>(DefaultFilters);
  const [embeddingFilter, setEmbeddingFilter] = useState<
    Embedding2DFilter | undefined
  >();

  const rawFilters: ExplorerFilterState = useMemo(() => {
    const removeTagFilter = (
      enumFilters: Readonly<Record<string, readonly string[]>>
    ): DomainSearchFilters["enums"] =>
      Object.fromEntries(
        Object.entries(enumFilters)
          .filter(([k]) => k !== "tags")
          .map(([k, v]) => [k, [...v]])
      );

    return {
      analysisDomain: selectedMetric.domain,
      filters: {
        data: {
          // FIXME: the 'as' casts should NOT! be needed
          metrics: dataFilters.metricFilters as DomainSearchFilters["metrics"],
          enums: removeTagFilter(dataFilters.enumFilters),
          reduction:
            selectedMetric.domain === "data" ? embeddingFilter : undefined,
          tags: dataFilters.enumFilters.tags as DomainSearchFilters["tags"],
        },
        annotation: {
          metrics:
            annotationFilters.metricFilters as DomainSearchFilters["metrics"],
          enums: removeTagFilter(annotationFilters.enumFilters),
          reduction:
            selectedMetric.domain === "annotation"
              ? embeddingFilter
              : undefined,
          tags: annotationFilters.enumFilters
            .tags as DomainSearchFilters["tags"],
        },
      },
      orderBy: selectedMetric.metric_key,
      desc: !isAscending,
      iou,
      predictionOutcome,
      predictionHash,
    };
  }, [
    selectedMetric,
    dataFilters,
    isAscending,
    annotationFilters,
    predictionHash,
    predictionOutcome,
    embeddingFilter,
    iou,
  ]);

  const filters: ExplorerFilterState = useDebounce(rawFilters, 500);

  // Load all collaborators & tags -> needed to support filters
  const { data: collaborators } = useProjectListCollaborators(projectHash);
  const { data: tags } = useProjectListTags(projectHash);

  // Load metric ranges
  const { data: dataMetricRanges, isLoading: isLoadingDataMetrics } =
    useProjectAnalysisSummary(projectHash, "data");
  const {
    data: annotationMetricRanges,
    isLoading: isLoadingAnnotationMetrics,
  } = useProjectAnalysisSummary(projectHash, "annotation");
  const isLoadingMetrics = isLoadingDataMetrics || isLoadingAnnotationMetrics;

  // Premium search hooks:
  const { premiumSearchState } = useExplorerPremiumSearch();
  const { search, setSearch } = premiumSearchState;

  const { data: sortedItemsProject, isLoading: isLoadingSortedItemsProject } =
    useProjectAnalysisSearch(
      projectHash,
      filters.analysisDomain,
      similarityItem != null || search != null ? undefined : filters.orderBy,
      filters.desc,
      0,
      1000,
      filters.filters,
      similarityItem,
      typeof search === "string" ? search : undefined,
      typeof search !== "string" ? search : undefined,
      {
        enabled: predictionHash === undefined,
      }
    );
  const {
    data: sortedItemsPrediction,
    isLoading: isLoadingSortedItemsPrediction,
  } = usePredictionAnalysisSearch(
    projectHash,
    predictionHash ?? "",
    filters.predictionOutcome,
    filters.iou,
    filters.orderBy,
    filters.analysisDomain === "data",
    filters.desc,
    0,
    1000,
    filters.filters,
    similarityItem,
    typeof search === "string" ? search : undefined,
    typeof search !== "string" ? search : undefined,
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

  const isSortedByMetric = !similarityItem && !search;

  const canResetFilters =
    similarityItem ||
    search ||
    predictionOutcome !== "tp" ||
    iou !== 0.5 ||
    dataFilters.ordering.length !== 0 ||
    annotationFilters.ordering.length !== 0 ||
    embeddingFilter !== undefined;

  const reset = (clearFilters: boolean = true) => {
    setSimilarityItem(undefined);
    setSearch(undefined);
    if (clearFilters) {
      setIsAscending(true);
      setPredictionOutcome("tp");
      setIou(0.5);
      setDataFilters(DefaultFilters);
      setAnnotationFilters(DefaultFilters);
      setEmbeddingFilter(undefined);
    }
  };

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  const itemsToRender: readonly string[] = useMemo(() => {
    if (sortedItems == null) {
      return [];
    }
    if (similarityItem != null) {
      return [similarityItem, ...sortedItems.results];
    }
    return sortedItems.results;
  }, [sortedItems, similarityItem]);
  const itemSimilarities: readonly number[] | undefined = useMemo(() => {
    if (sortedItems == null) {
      return undefined;
    }
    const { similarities } = sortedItems;
    if (similarities !== undefined && similarityItem != null) {
      return [0.0, ...similarities];
    }
    return similarities;
  }, [sortedItems, similarityItem]);
  const itemTruncated = sortedItems?.truncated ?? false;

  const toggleImageSelection = useCallback((id: string) => {
    setSelectedItems((prev) => {
      if (prev === "ALL") {
        return "ALL";
      }
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const closePreview = useCallback(
    () => setPreviewedItem(undefined),
    [setPreviewedItem]
  );
  const setSimilaritySearch = useCallback(
    (itemId: string | undefined) => {
      setSearch(undefined);
      closePreview();
      setSimilarityItem(itemId);
    },
    [closePreview, setSearch]
  );

  const allowTaggingAnnotations = selectedMetric?.domain === "annotation";

  const loadingDescription = useMemo(() => {
    const descriptions = [
      {
        isLoading: isLoadingMetrics,
        description: "Loading available metrics",
      },
      {
        isLoading: isLoadingSortedItems,
        description: "Loading search results",
      },
    ];

    return descriptions.reduce(
      (res, item) => (!res && item.isLoading ? item.description : res),
      ""
    );
  }, [isLoadingMetrics, isLoadingSortedItems]);

  // Modal state
  const [open, setOpen] = useState<undefined | "subset" | "upload">();
  const close = () => setOpen(undefined);

  return (
    <div className="h-full">
      <CreateSubsetModal
        open={open === "subset"}
        close={close}
        projectHash={projectHash}
        filters={filters.filters}
      />
      <UploadToEncordModal
        open={open === "upload"}
        close={close}
        projectHash={projectHash}
        setSelectedProjectHash={setSelectedProjectHash}
      />
      <Modal
        title={`Changing domain to ${selectedMetric.domain}`}
        onOk={() => {
          setSelectedItems(new Set());
          setClearSelectionModalVisible(false);
        }}
        onCancel={() => setClearSelectionModalVisible(false)}
        open={clearSelectionModalVisible}
        okText="Clear"
        cancelText="Preserve"
        okButtonProps={{ style: { backgroundColor: "#5555ff" } }}
      >
        You have selected items from the previous domain, do you want to clear
        the selection?
      </Modal>
      <ItemPreviewModal
        projectHash={projectHash}
        predictionHash={predictionHash}
        previewItem={previewItem}
        domain={selectedMetric.domain}
        onClose={closePreview}
        onShowSimilar={() =>
          previewItem != null ? setSimilaritySearch(previewItem) : undefined
        }
        editUrl={editUrl}
      />
      <Row className="h-full">
        <Col span={18} className="h-full">
          <Tabs
            style={{
              height: "100%",
            }}
            className="h-full"
            tabBarStyle={{
              margin: 0,
            }}
            tabBarExtraContent={{
              left: (
                <ExplorerPremiumSearch
                  premiumSearchState={{
                    ...premiumSearchState,
                    setSearch: (args) => {
                      setSimilarityItem(undefined);
                      premiumSearchState.setSearch(args);
                    },
                  }}
                />
              ),
              right: (
                <Segmented
                  className="mr-4"
                  selected
                  options={["Data", "Labels"]}
                />
              ),
            }}
            items={[
              {
                label: (
                  <span className="flex items-center gap-2">
                    <img src={TableIcon} alt="grid" className="h-4 w-4" />
                    Grid View
                  </span>
                ),
                key: "gridView",
                children: (
                  <div className="flex h-full flex-col bg-gray-100">
                    <Space
                      style={{
                        flex: "0 1 0",
                        height: "100%",
                      }}
                      wrap
                    >
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
                          className="w-80"
                          options={[
                            {
                              label: "Data Metrics",
                              options: Object.entries(
                                dataMetricsSummary.metrics
                              ).map(([metricKey, metric]) => ({
                                label: `D: ${metric?.title ?? metricKey}`,
                                value: `data-${metricKey}`,
                              })),
                            },
                            {
                              label:
                                predictionHash === undefined
                                  ? "Annotation Metrics"
                                  : "Prediction Metrics",
                              options: Object.entries(
                                annotationMetricsSummary.metrics
                              ).map(([metricKey, metric]) => ({
                                label: `${
                                  predictionHash === undefined ? "A" : "P"
                                }: ${metric?.title ?? metricKey}`,
                                value: `annotation-${metricKey}`,
                              })),
                            },
                          ]}
                        />
                        <Button
                          disabled={!isSortedByMetric}
                          onClick={() => setIsAscending(!isAscending)}
                          icon={
                            isAscending ? (
                              <TbSortAscending />
                            ) : (
                              <TbSortDescending />
                            )
                          }
                        />
                        <Button onClick={toggleShowAnnotations}>
                          {`${
                            showAnnotations ? "Show" : "hide"
                          } all annotations`}
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
                              tags={tags ?? []}
                              collaborators={collaborators ?? []}
                            />
                          }
                          trigger="click"
                        >
                          <Button>
                            Data Filters
                            {` (${dataFilters.ordering.length})`}
                          </Button>
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
                              tags={tags ?? []}
                              collaborators={collaborators ?? []}
                            />
                          }
                          trigger="click"
                        >
                          <Button>
                            Annotation Filters
                            {` (${annotationFilters.ordering.length})`}
                          </Button>
                        </Popover>
                        {/* <Button
                          disabled={!canResetFilters}
                          onClick={() => reset()}
                          icon={<MdFilterAltOff />}
                        >
                          Reset filters
                        </Button> */}
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
                          <Tooltip
                            title={
                              !selectedItems.size
                                ? "Select items to tag first"
                                : ""
                            }
                          >
                            <Button
                              icon={<HiOutlineTag />}
                              disabled={!selectedItems.size}
                            >
                              Tag
                            </Button>
                          </Tooltip>
                        </Popover>
                        <Button
                          disabled={!selectedItems.size}
                          onClick={() => setSelectedItems(new Set())}
                          icon={<VscClearAll />}
                        >
                          Clear selection ({selectedItems.size})
                        </Button>
                        <Button
                          onClick={() =>
                            setSelectedItems(new Set(itemsToRender))
                          }
                          disabled={itemsToRender.length === 0}
                          icon={<BiSelectMultiple />}
                        >
                          Select all ({itemsToRender.length})
                        </Button>
                      </Space.Compact>
                      {env !== "sandbox" && !(remoteProject || !local) ? (
                        <Space.Compact size="large">
                          <Button
                            onClick={() => setOpen("subset")}
                            disabled={!canResetFilters}
                            icon={<BiWindows />}
                          >
                            Create Project subset
                          </Button>
                          <Button
                            onClick={() => setOpen("upload")}
                            disabled={!!canResetFilters}
                            icon={<BiCloudUpload />}
                          >
                            Upload project
                          </Button>
                        </Space.Compact>
                      ) : (
                        <>
                          <Button
                            onClick={() => setOpen("subset")}
                            disabled={!canResetFilters}
                            hidden={env === "sandbox"}
                            icon={<BiWindows />}
                            size="large"
                          >
                            Create Project subset
                          </Button>
                          <Button
                            onClick={() => setOpen("upload")}
                            disabled={!!canResetFilters}
                            hidden={remoteProject || !local}
                            icon={<BiCloudUpload />}
                            size="large"
                          >
                            Upload project
                          </Button>
                        </>
                      )}
                    </Space>
                    <div
                      style={{
                        flex: "1 1 auto",
                        height: "100%",
                      }}
                    >
                      <ExplorerSearchResults
                        projectHash={projectHash}
                        predictionHash={predictionHash}
                        itemsToRender={itemsToRender}
                        itemSimilarities={itemSimilarities}
                        itemSimilarityItemAtIndex0={similarityItem != null}
                        truncated={itemTruncated}
                        loadingDescription={loadingDescription}
                        selectedMetric={selectedMetric}
                        toggleImageSelection={toggleImageSelection}
                        setPreviewedItem={setPreviewedItem}
                        setSimilaritySearch={setSimilaritySearch}
                        selectedItems={selectedItems}
                        showAnnotations={showAnnotations}
                        featureHashMap={featureHashMap}
                        iou={iou}
                      />
                    </div>
                  </div>
                ),
              },
              {
                label: (
                  <span className="flex items-center gap-2">
                    <img src={EmbeddingsIcon} alt="grid" className="h-4 w-4" />
                    Embeddings View
                  </span>
                ),
                key: "embeddingsView",
                children: (
                  <ExplorerEmbeddings
                    projectHash={projectHash}
                    predictionHash={predictionHash}
                    reductionHash={reductionHash}
                    reductionHashLoading={reductionHashLoading}
                    filters={filters}
                    setEmbeddingSelection={setEmbeddingFilter}
                    featureHashMap={featureHashMap}
                  />
                ),
              },
            ]}
          />
        </Col>
        <Col span={6} className=" border-l border-l-gray-200 px-2">
          <Tabs
            items={[
              {
                label: "Overview",
                key: "overview",
                children: (
                  <Space direction="vertical" size="large" className="p-4">
                    <div className="gap-f flex flex-col">
                      <div className="text-base text-gray-500">
                        Total No. of frames
                      </div>
                      <div className="text-2xl">48,326</div>
                    </div>
                    <Space size="small" direction="vertical">
                      <div className="text-base text-gray-500">
                        Data Quality Score <InfoCircleOutlined />
                      </div>
                      <div className="text-2xl">50</div>
                    </Space>
                    <Space size="small" direction="vertical">
                      <div className="text-base text-gray-500">
                        Issue Types <InfoCircleOutlined />
                      </div>
                      <div className="text-sm">Duplicate</div>
                      <div className="text-sm">Blur</div>
                      <div className="text-sm">Dark</div>
                      <div className="text-sm">Bright</div>
                    </Space>
                  </Space>
                ),
              },
              {
                label: "Filter",
                key: "filter",
                children: <>Filters come here</>,
              },
              {
                label: "Display",
                key: "display",
                children: <>This is display for something for sure</>,
              },
            ]}
          />
        </Col>
      </Row>
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
    <Space.Compact size="large">
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
        <div
          className={classy(
            "box-border h-10 w-80 rounded-r-lg border-r border-b border-t border-solid border-gray-200",
            "flex items-center gap-1 px-2"
          )}
        >
          <Typography.Text strong className="min-w-fit">
            IOU:
          </Typography.Text>
          <Slider
            className="w-full"
            tooltip={{
              formatter: (val: number | undefined) => `${val}`,
            }}
            value={iou}
            onChange={setIou}
            min={0}
            max={1}
            step={0.01}
          />
        </div>
      )}
    </Space.Compact>
  );
}
