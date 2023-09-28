import { Button, Popover, Select, Space } from "antd";
import { ProjectDomainSummary } from "../../../openapi/api";
import { FilterState, MetricFilter } from "../../util/MetricFilter";
import { Metric } from "../Explorer";
import { MdFilterAltOff } from "react-icons/md";
import { FeatureHashMap } from "../../Types";
import { TbSortAscending, TbSortDescending } from "react-icons/tb";
import { useProjectListCollaborators } from "../../../hooks/queries/useProjectListCollaborators";
import { useProjectListTags } from "../../../hooks/queries/useProjectListTags";
import { useProjectAnalysisSummary } from "../../../hooks/queries/useProjectAnalysisSummary";

type Props = {
  projectHash: string;
  selectedMetric: Metric;
  predictionHash: string | undefined;
  dataMetricsSummary: ProjectDomainSummary;
  annotationMetricsSummary: ProjectDomainSummary;
  isSortedByMetric: boolean;
  setSelectedMetric: (val: Metric) => void;
  isAscending: boolean;
  setIsAscending: React.Dispatch<React.SetStateAction<boolean>>;
  annotationFilters: FilterState;
  setAnnotationFilters: React.Dispatch<React.SetStateAction<FilterState>>;
  dataFilters: FilterState;
  setDataFilters: React.Dispatch<React.SetStateAction<FilterState>>;
  canResetFilters: string | boolean | File;
  reset: any;
  featureHashMap: FeatureHashMap;
  showAnnotations: boolean;
  toggleShowAnnotations: any;
};
export function Filters({
  projectHash,
  predictionHash,
  dataMetricsSummary,
  annotationMetricsSummary,
  isSortedByMetric,
  selectedMetric,
  setSelectedMetric,
  isAscending,
  setIsAscending,
  annotationFilters,
  setAnnotationFilters,
  canResetFilters,
  reset,
  featureHashMap,
  dataFilters,
  setDataFilters,
  showAnnotations,
  toggleShowAnnotations,
}: Props) {
  // Load metric ranges
  const { data: dataMetricRanges, isLoading: isLoadingDataMetrics } =
    useProjectAnalysisSummary(projectHash, "data");
  const {
    data: annotationMetricRanges,
    isLoading: isLoadingAnnotationMetrics,
  } = useProjectAnalysisSummary(projectHash, "annotation");
  const isLoadingMetrics = isLoadingDataMetrics || isLoadingAnnotationMetrics;

  // Load all collaborators & tags -> needed to support filters
  const { data: collaborators } = useProjectListCollaborators(projectHash);
  const { data: tags } = useProjectListTags(projectHash);

  return (
    <Space.Compact size="large" direction="vertical">
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
            options: Object.entries(dataMetricsSummary.metrics).map(
              ([metricKey, metric]) => ({
                label: `D: ${metric?.title ?? metricKey}`,
                value: `data-${metricKey}`,
              })
            ),
          },
          {
            label:
              predictionHash === undefined
                ? "Annotation Metrics"
                : "Prediction Metrics",
            options: Object.entries(annotationMetricsSummary.metrics).map(
              ([metricKey, metric]) => ({
                label: `${predictionHash === undefined ? "A" : "P"}: ${
                  metric?.title ?? metricKey
                }`,
                value: `annotation-${metricKey}`,
              })
            ),
          },
        ]}
      />
      <Button
        disabled={!isSortedByMetric}
        onClick={() => setIsAscending(!isAscending)}
        icon={isAscending ? <TbSortAscending /> : <TbSortDescending />}
      />
      <Button onClick={toggleShowAnnotations}>
        {`${showAnnotations ? "Show" : "hide"} all annotations`}
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
      <Button
        disabled={!canResetFilters}
        onClick={() => reset()}
        icon={<MdFilterAltOff />}
      >
        Reset filters
      </Button>
    </Space.Compact>
  );
}
