import { Button, Select, Slider, Space, Switch, Tooltip } from "antd";
import { Dispatch, SetStateAction } from "react";
import { TbSortAscending, TbSortDescending } from "react-icons/tb";
import {
  InfoCircleOutlined,
  MinusSquareFilled,
  TableOutlined,
} from "@ant-design/icons";
import {
  AnalysisDomain,
  ProjectDomainSummary,
  QuerySummary,
} from "../../../openapi/api";
import {
  FilterState,
  getMetricBounds,
  toFixedNumber,
  updateValue,
} from "../../util/MetricFilter";
import { Colors, GRID_MAX_COUNT, GRID_MIN_COUNT } from "../../../constants";

type Props = {
  annotationFilters: FilterState;
  annotationMetricRanges: QuerySummary["metrics"] | undefined;
  selectedMetric: string;
  analysisDomain: AnalysisDomain;
  predictionHash: string | undefined;
  dataMetricsSummary: ProjectDomainSummary;
  annotationMetricsSummary: ProjectDomainSummary;
  isSortedByMetric: boolean;
  setSelectedMetric: (val: string) => void;
  isAscending: boolean;
  setIsAscending: Dispatch<SetStateAction<boolean>>;
  showAnnotations: boolean;
  toggleShowAnnotations: any;
  setConfidenceFilter: (
    arg: FilterState | ((old: FilterState) => FilterState)
  ) => void;
  gridCount: number;
  setGridCount: (val: number) => void;
};
export function Display({
  annotationFilters,
  annotationMetricRanges,
  predictionHash,
  dataMetricsSummary,
  annotationMetricsSummary,
  isSortedByMetric,
  selectedMetric,
  analysisDomain,
  setSelectedMetric,
  isAscending,
  setIsAscending,
  showAnnotations,
  toggleShowAnnotations,
  setConfidenceFilter,
  gridCount,
  setGridCount,
}: Props) {
  // Setting confidence metric specific parameters
  const METRIC_CONFIDENCE = "metric_confidence";
  const confidenceFilters = annotationFilters.metricFilters[METRIC_CONFIDENCE];
  const confidenceRange = annotationMetricRanges
    ? annotationMetricRanges[METRIC_CONFIDENCE]
    : undefined;
  const confidence = annotationMetricsSummary.metrics[METRIC_CONFIDENCE];
  const confidenceBounds =
    confidenceRange !== undefined && confidence !== undefined
      ? getMetricBounds(confidenceRange, confidence)
      : undefined;

  return (
    <div className="flex flex-col gap-2 divide-y ">
      <div className="icon-wrapper flex w-full items-center gap-2">
        <MinusSquareFilled className="text-base text-gray-8" />
        <Slider
          min={GRID_MIN_COUNT}
          max={GRID_MAX_COUNT}
          defaultValue={gridCount ?? 4}
          className="w-full"
          trackStyle={{
            backgroundColor: Colors.darkGray,
          }}
          onChange={(val) => setGridCount(val)}
        />
        <TableOutlined className="text-base text-gray-8" />
      </div>
      <div className="flex items-center justify-between">
        <div className="text-gray-1 flex items-center gap-1 px-2 py-4 text-sm">
          Display Labels
          <Tooltip title="Hide and show annotated labels on your data units.">
            <InfoCircleOutlined />
          </Tooltip>
        </div>
        <Switch
          onClick={toggleShowAnnotations}
          className="bg-gray-9"
          checked={!showAnnotations}
        />
      </div>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1 px-2 py-4 text-sm text-gray-8">
          Display Predictions
          <Tooltip title="Hide and show predictions to compare against labels and assess model performance.">
            <InfoCircleOutlined />
          </Tooltip>
        </div>
        <Switch disabled className="bg-gray-9" />
      </div>
      <div className="flex flex-col items-start justify-between p-4">
        <div className=" flex items-center gap-1 text-sm text-gray-8">
          Confidence
          <Tooltip title="Set a minimum/maximum confidence level to filter and display only those predictions that meet or exceed this value.">
            <InfoCircleOutlined />
          </Tooltip>
        </div>
        {confidenceBounds !== undefined && (
          <Slider
            disabled
            value={
              confidenceFilters != null
                ? [confidenceFilters[0], confidenceFilters[1]]
                : [
                    toFixedNumber(confidenceBounds.min, 2),
                    toFixedNumber(confidenceBounds.max, 2),
                  ]
            }
            range={{ draggableTrack: true }}
            min={confidenceBounds.min}
            max={confidenceBounds.max}
            step={0.01}
            className="w-full"
            onChange={(newRange: [number, number]) => {
              setConfidenceFilter(
                updateValue(METRIC_CONFIDENCE, newRange, "metricFilters")
              );
            }}
          />
        )}
      </div>
      <div>
        <Space.Compact size="large" className="mt-4 w-full">
          <Select
            value={`${analysisDomain}-${selectedMetric}`}
            onChange={(strKey: string) => {
              const metric_key = strKey.split("-")[1];
              setSelectedMetric(metric_key);
            }}
            className="w-full"
            options={
              analysisDomain === AnalysisDomain.Data
                ? Object.entries(dataMetricsSummary.metrics).map(
                    ([metricKey, metric]) => ({
                      label: `D: ${metric?.title ?? metricKey}`,
                      value: `data-${metricKey}`,
                    })
                  )
                : Object.entries(annotationMetricsSummary.metrics).map(
                    ([metricKey, metric]) => ({
                      label: `${predictionHash === undefined ? "A" : "P"}: ${
                        metric?.title ?? metricKey
                      }`,
                      value: `annotation-${metricKey}`,
                    })
                  )
            }
          />
          <Button
            disabled={!isSortedByMetric}
            onClick={() => setIsAscending(!isAscending)}
            icon={isAscending ? <TbSortAscending /> : <TbSortDescending />}
          />
        </Space.Compact>
      </div>
    </div>
  );
}
