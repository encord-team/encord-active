import {
  useMemo,
  useState,
  useEffect,
  useCallback,
  Dispatch,
  SetStateAction,
} from "react";
import { BiSelectMultiple } from "react-icons/bi";
import { VscClearAll } from "react-icons/vsc";
import { useDebounce, useLocalStorage, useToggle } from "usehooks-ts";
import {
  Button,
  Col,
  Dropdown,
  Modal,
  Popover,
  Row,
  Segmented,
  Space,
  Tabs,
} from "antd";
import { useNavigate, useParams } from "react-router";
import {
  DotChartOutlined,
  DownOutlined,
  TableOutlined,
} from "@ant-design/icons";
import { BulkTaggingForm } from "./Tagging";
import {
  FilterState,
  DefaultFilters,
  DefaultAnnotationFilters,
} from "../util/MetricFilter";
import { UploadToEncordModal } from "../tabs/modals/UploadToEncordModal";
import { ExplorerEmbeddings } from "./ExplorerEmbeddings";
import { CreateSubsetModal } from "../tabs/modals/CreateSubsetModal";
import {
  AnalysisDomain,
  DomainSearchFilters,
  Embedding2DFilter,
  PredictionDomain,
  ProjectDomainSummary,
} from "../../openapi/api";
import { useProjectListReductions } from "../../hooks/queries/useProjectListReductions";
import { useProjectAnalysisSummary } from "../../hooks/queries/useProjectAnalysisSummary";
import { useProjectAnalysisSearch } from "../../hooks/queries/useProjectAnalysisSearch";
import {
  ExplorerFilterState,
  Metric,
  analysisDomainLabelOverrides,
} from "./ExplorerTypes";
import { ItemPreviewModal } from "../preview/ItemPreviewModal";
import { usePredictionAnalysisSearch } from "../../hooks/queries/usePredictionAnalysisSearch";
import {
  ExplorerPremiumSearch,
  useExplorerPremiumSearch,
} from "./ExplorerPremiumSearch";
import { ExplorerSearchResults } from "./ExplorerSearchResults";
import { FeatureHashMap, ModalName } from "../Types";
import { Filters } from "./filters/Filters";
import { Overview } from "./overview/Overview";
import { Display } from "./display/Display";

import "./css/explorer.css";
import { classy } from "../../helpers/classy";
import { useUserSettings } from "../../hooks/useUserSettings";

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
  openModal: ModalName | undefined;
  setOpenModal: Dispatch<SetStateAction<ModalName | undefined>>;
  selectedItems: ReadonlySet<string> | "ALL";
  setSelectedItems: Dispatch<SetStateAction<ReadonlySet<string> | "ALL">>;
  hasSelectedItems: boolean;
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
  openModal,
  setOpenModal,
  selectedItems,
  setSelectedItems,
  hasSelectedItems,
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

  const [analysisDomain, setAnalysisDomain] = useState<AnalysisDomain>("data");
  const [selectedMetricData, setSelectedMetricData] =
    useState<string>("metric_random");
  const [selectedMetricLabel, setSelectedMetricLabel] =
    useState<string>("metric_random");

  const selectedMetric =
    analysisDomain === AnalysisDomain.Data
      ? selectedMetricData
      : selectedMetricLabel;

  const handleMetricChange = (val: string) => {
    if (analysisDomain === AnalysisDomain.Data) {
      setSelectedMetricData(val);
    } else if (analysisDomain === AnalysisDomain.Annotation) {
      setSelectedMetricLabel(val);
    }
  };

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
  }, [analysisDomain]);

  // Set show animations view state.
  const [showAnnotations, toggleShowAnnotations, setShowAnnotations] =
    useToggle(true);
  useEffect(() => {
    setShowAnnotations(analysisDomain === AnalysisDomain.Annotation);
  }, [analysisDomain, setShowAnnotations]);

  // Data or Label selection
  const analysisDomainOptions = useMemo(
    () =>
      Object.entries(AnalysisDomain).map(([key, value]) => ({
        label: Object.prototype.hasOwnProperty.call(
          analysisDomainLabelOverrides,
          value
        )
          ? analysisDomainLabelOverrides[value as AnalysisDomain]
          : key,
        value,
      })),
    []
  );

  // Filter State
  const [isAscending, setIsAscending] = useState(true);
  const [predictionOutcome, setPredictionOutcome] =
    useState<PredictionDomain>("p");
  const [iou, setIou] = useState<number>(0.5);
  const [dataFilters, setDataFilters] = useState<FilterState>(DefaultFilters);
  const [annotationFilters, setAnnotationFilters] = useState<FilterState>(
    DefaultAnnotationFilters
  );
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
      analysisDomain: analysisDomain,
      filters: {
        data: {
          // FIXME: the 'as' casts should NOT! be needed
          metrics: dataFilters.metricFilters as DomainSearchFilters["metrics"],
          enums: removeTagFilter(dataFilters.enumFilters),
          reduction:
            analysisDomain === AnalysisDomain.Data
              ? embeddingFilter
              : undefined,
          tags: dataFilters.enumFilters.tags as DomainSearchFilters["tags"],
        },
        annotation: {
          metrics:
            annotationFilters.metricFilters as DomainSearchFilters["metrics"],
          enums: removeTagFilter(annotationFilters.enumFilters),
          reduction:
            analysisDomain === AnalysisDomain.Annotation
              ? embeddingFilter
              : undefined,
          tags: annotationFilters.enumFilters
            .tags as DomainSearchFilters["tags"],
        },
      },
      orderBy: selectedMetric,
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

  // Load metric ranges
  const { isLoading: isLoadingDataMetrics } = useProjectAnalysisSummary(
    projectHash,
    "data"
  );
  const { isLoading: isLoadingAnnotationMetrics } = useProjectAnalysisSummary(
    projectHash,
    "annotation"
  );
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

  const [userDisplaySettings, updateUserDisplaySettings] = useUserSettings();

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

  const toggleImageSelection = useCallback(
    (id: string) => {
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
    },
    [setSelectedItems]
  );

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

  const allowTaggingAnnotations = analysisDomain === AnalysisDomain.Annotation;

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
  const close = () => setOpenModal(undefined);

  // view state
  const [activeView, setActiveView] = useState<"gridView" | "embeddingsView">(
    "gridView"
  );

  // Load metric ranges
  const { data: dataMetricRanges } = useProjectAnalysisSummary(
    projectHash,
    "data"
  );

  const { data: annotationMetricRanges } = useProjectAnalysisSummary(
    projectHash,
    "annotation"
  );

  return (
    <div className="h-full">
      <CreateSubsetModal
        open={openModal === "subset"}
        close={close}
        projectHash={projectHash}
        filters={filters.filters}
      />
      <UploadToEncordModal
        open={openModal === "upload"}
        close={close}
        projectHash={projectHash}
        setSelectedProjectHash={setSelectedProjectHash}
      />
      <Modal
        title={`Changing domain to ${analysisDomain}`}
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
        domain={analysisDomain}
        onClose={closePreview}
        onShowSimilar={() =>
          previewItem != null ? setSimilaritySearch(previewItem) : undefined
        }
        editUrl={editUrl}
      />
      <Row className="h-full">
        <Col span={18} className="h-full">
          <Tabs
            centered
            activeKey={activeView}
            onChange={setActiveView as (val: string) => void}
            className="explorer-tabs h-full"
            tabBarStyle={{
              margin: 0,
            }}
            tabBarExtraContent={{
              left: (
                <div className=" flex h-full items-center">
                  <ExplorerPremiumSearch
                    premiumSearchState={{
                      ...premiumSearchState,
                      setSearch: (args) => {
                        setSimilarityItem(undefined);
                        premiumSearchState.setSearch(args);
                      },
                    }}
                  />
                  <Segmented
                    selected
                    value={analysisDomain}
                    options={[
                      ...analysisDomainOptions,
                      {
                        value: "predictions",
                        label: "Predictions",
                        disabled: true,
                      },
                    ]}
                    onChange={(val) => {
                      setAnalysisDomain(val as AnalysisDomain);
                    }}
                  />
                </div>
              ),
            }}
            items={[
              {
                label: (
                  <span className="flex items-center gap-2">
                    <TableOutlined />
                    Grid View
                  </span>
                ),
                key: "gridView",
                children: (
                  <div className="flex h-full flex-col items-center bg-gray-100 py-2">
                    <div
                      className={classy(
                        "top-1.5 z-[1000] flex flex-shrink flex-grow-0 basis-0 items-center gap-3 rounded-md bg-white py-2 px-6 opacity-0",
                        {
                          "opacity-100": hasSelectedItems,
                        }
                      )}
                    >
                      <Dropdown
                        menu={{
                          items: [
                            {
                              label: (
                                <Button
                                  onClick={() =>
                                    setSelectedItems((oldState) => {
                                      if (oldState === "ALL") {
                                        return "ALL";
                                      }
                                      return new Set([
                                        ...oldState,
                                        ...itemsToRender.slice(
                                          (page - 1) *
                                            userDisplaySettings.explorerPageSize,
                                          page *
                                            userDisplaySettings.explorerPageSize
                                        ),
                                      ]);
                                    })
                                  }
                                  disabled={itemsToRender.length === 0}
                                  icon={<BiSelectMultiple />}
                                  className="text-md font-medium text-gray-9"
                                >
                                  Select page (
                                  {
                                    itemsToRender.slice(
                                      (page - 1) *
                                        userDisplaySettings.explorerPageSize,
                                      page *
                                        userDisplaySettings.explorerPageSize
                                    ).length
                                  }
                                  )
                                </Button>
                              ),
                              key: "0",
                            },
                            {
                              label: (
                                <Button
                                  onClick={() => setSelectedItems("ALL")}
                                  disabled={selectedItems === "ALL"}
                                  icon={<BiSelectMultiple />}
                                >
                                  Select All
                                </Button>
                              ),
                              key: "1",
                            },
                          ],
                        }}
                        trigger={["click"]}
                      >
                        <Space>
                          Select
                          <DownOutlined />
                        </Space>
                      </Dropdown>

                      <Button
                        className="border-none bg-gray-9 text-white"
                        disabled={!hasSelectedItems}
                        onClick={() => setSelectedItems(new Set())}
                        icon={<VscClearAll />}
                      >
                        Clear selection{" "}
                        <span className="text-gray-5 w-6">
                          (
                          {selectedItems === "ALL" ? "All" : selectedItems.size}
                          )
                        </span>
                      </Button>

                      <Popover
                        placement="bottomRight"
                        content={
                          <BulkTaggingForm
                            projectHash={projectHash}
                            selectedItems={selectedItems}
                            filtersDomain={filters.analysisDomain}
                            filters={filters.filters}
                            allowTaggingAnnotations={allowTaggingAnnotations}
                          />
                        }
                        trigger="click"
                      >
                        <Button
                          className="border-none bg-gray-9 text-white"
                          type="default"
                          disabled={!hasSelectedItems}
                        >
                          Tag
                        </Button>
                      </Popover>
                    </div>
                    <div className="-mt-10 h-full w-full flex-auto">
                      <ExplorerSearchResults
                        projectHash={projectHash}
                        predictionHash={predictionHash}
                        itemsToRender={itemsToRender}
                        itemSimilarities={itemSimilarities}
                        itemSimilarityItemAtIndex0={similarityItem != null}
                        truncated={itemTruncated}
                        loadingDescription={loadingDescription}
                        selectedMetric={selectedMetric}
                        analysisDomain={analysisDomain}
                        toggleImageSelection={toggleImageSelection}
                        setPreviewedItem={setPreviewedItem}
                        setSimilaritySearch={setSimilaritySearch}
                        selectedItems={selectedItems}
                        showAnnotations={showAnnotations}
                        featureHashMap={featureHashMap}
                        iou={iou}
                        setPage={setPage}
                        page={page}
                        setPageSize={(val: number) => {
                          updateUserDisplaySettings({ explorerPageSize: val });
                        }}
                        pageSize={userDisplaySettings.explorerPageSize}
                        gridCount={userDisplaySettings.explorerGridCount}
                      />
                    </div>
                  </div>
                ),
              },
              {
                label: (
                  <span className="mr-2 flex items-center gap-2">
                    <DotChartOutlined /> Embeddings View
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
                  <Overview
                    projectHash={projectHash}
                    analysisDomain={
                      analysisDomain === "data" ? "data" : "annotation"
                    }
                  />
                ),
              },
              {
                label: "Filter",
                key: "filter",
                children: (
                  <Filters
                    projectHash={projectHash}
                    dataMetricsSummary={dataMetricsSummary}
                    annotationMetricsSummary={annotationMetricsSummary}
                    annotationFilters={annotationFilters}
                    setAnnotationFilters={setAnnotationFilters}
                    dataFilters={dataFilters}
                    setDataFilters={setDataFilters}
                    featureHashMap={featureHashMap}
                    reset={reset}
                    canResetFilters={canResetFilters}
                    dataMetricRanges={dataMetricRanges?.metrics}
                    annotationMetricRanges={annotationMetricRanges?.metrics}
                  />
                ),
              },
              {
                label: "Display",
                key: "display",
                children: (
                  <Display
                    dataFilters={dataFilters}
                    annotationFilters={annotationFilters}
                    dataMetricRanges={dataMetricRanges?.metrics}
                    annotationMetricRanges={annotationMetricRanges?.metrics}
                    metricsSummary={dataMetricsSummary}
                    selectedMetric={selectedMetric}
                    analysisDomain={analysisDomain}
                    isSortedByMetric={isSortedByMetric}
                    predictionHash={predictionHash}
                    dataMetricsSummary={dataMetricsSummary}
                    annotationMetricsSummary={annotationMetricsSummary}
                    setSelectedMetric={handleMetricChange}
                    isAscending={isAscending}
                    setIsAscending={setIsAscending}
                    showAnnotations={showAnnotations}
                    toggleShowAnnotations={toggleShowAnnotations}
                    setConfidenceFilter={setAnnotationFilters}
                    gridCount={userDisplaySettings.explorerGridCount}
                    setGridCount={(val: number) => {
                      updateUserDisplaySettings({ explorerGridCount: val });
                    }}
                  />
                ),
              },
            ]}
          />
        </Col>
      </Row>
    </div>
  );
}
