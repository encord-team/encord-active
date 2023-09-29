import { Button, Select, Slider, Space, Switch } from "antd";
import { Dispatch, SetStateAction } from "react";
import { TbSortAscending, TbSortDescending } from "react-icons/tb";
import { Metric } from "../ExplorerTypes";
import { ProjectDomainSummary } from "../../../openapi/api";
import { GrDisabledOutline } from "react-icons/gr";
import { ChartOutlierSummaryBar } from "../../charts/ChartOutlierSummaryBar";
import {
  MinusSquareFilled,
  TableOutlined,
  TabletFilled,
} from "@ant-design/icons";

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
    <div className="flex flex-col gap-2 divide-y ">
      <div className="icon-wrapper flex w-full items-center gap-2">
        <MinusSquareFilled className="text-base text-gray-8" />
        <Slider
          tooltip={{ open: false }}
          className="w-full"
          trackStyle={{
            backgroundColor: "#434343",
          }}
        />
        <TableOutlined className="text-base text-gray-8" />
      </div>
      <div className="flex items-center justify-between">
        <div className="px-2 py-4 text-sm text-gray-8">Display Labels</div>
        <Switch
          onClick={toggleShowAnnotations}
          className="bg-gray-9"
          checked={!showAnnotations}
        />
      </div>
      <div className="flex items-center justify-between">
        <div className="px-2 py-4 text-sm text-gray-8">Display Predictions</div>
        <Switch disabled className="bg-gray-9" />
      </div>
      <div className="flex flex-col items-start justify-between">
        <div className="px-2 py-4 text-sm text-gray-8">Confidence</div>
        <Slider range defaultValue={[20, 50]} className="w-full" />
      </div>
      <div className="flex flex-col items-start justify-between">
        <div className="px-2 py-4 text-sm text-gray-8">IOU</div>
        <Slider range defaultValue={[20, 50]} className="w-full" />
      </div>
      <Space.Compact size="large" className="w-full">
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
      </Space.Compact>
    </div>
  );
}
