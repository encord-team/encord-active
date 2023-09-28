import { Button, Select, Space } from "antd";
import { Dispatch, SetStateAction } from "react";
import { TbSortAscending, TbSortDescending } from "react-icons/tb";
import { Metric } from "../ExplorerTypes";
import { ProjectDomainSummary } from "../../../openapi/api";

type Props = {
  projectHash: string;
  selectedMetric: Metric;
  predictionHash: string | undefined;
  dataMetricsSummary: ProjectDomainSummary;
  annotationMetricsSummary: ProjectDomainSummary;
  isSortedByMetric: boolean;
  setSelectedMetric: (val: Metric) => void;
  isAscending: boolean;
  setIsAscending: Dispatch<SetStateAction<boolean>>;

  showAnnotations: boolean;
  toggleShowAnnotations: any;
};
export function Display({
  predictionHash,
  dataMetricsSummary,
  annotationMetricsSummary,
  isSortedByMetric,
  selectedMetric,
  setSelectedMetric,
  isAscending,
  setIsAscending,
  showAnnotations,
  toggleShowAnnotations,
}: Props) {
  return (
    <Space.Compact size="large" direction="vertical" className="w-full">
      <Select
        value={`${selectedMetric.domain}-${selectedMetric.metric_key}`}
        onChange={(strKey: string) => {
          const [domain, metric_key] = strKey.split("-");
          setSelectedMetric({
            domain: domain as "data" | "annotation",
            metric_key,
          });
        }}
        className="w-full"
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
    </Space.Compact>
  );
}
