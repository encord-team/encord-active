import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import * as React from "react";
import { scaleLinear } from "d3-scale";
import { useMemo } from "react";
import { ActiveProjectMetricSummary } from "../ActiveTypes";
import { formatTooltip } from "../util/ActiveFormatter";

function ActiveChartPredictionMetricVBar(props: {
  data: Readonly<Record<string, number>> | undefined;
  predictionMetric:
    | "importance"
    | "correlations"
    | "feature-precision"
    | "feature-recall";
  metricsSummary?: ActiveProjectMetricSummary;
  featureHashMap?: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { data, predictionMetric, metricsSummary, featureHashMap } = props;
  const barData = useMemo(() => {
    if (data === undefined) {
      return [];
    }
    const sortedData = Object.entries(data).map(([key, score]) => {
      let group = key;
      if (metricsSummary != null) {
        group = metricsSummary.metrics[group]?.title ?? group;
      } else if (featureHashMap != null) {
        group = featureHashMap[group]?.name ?? group;
      }
      return { group, score };
    });
    sortedData.sort((a, b) => b.score - a.score);
    return sortedData;
  }, [data, featureHashMap, metricsSummary]);

  const formattedBarData = useMemo(() => {
    const getColor =
      predictionMetric === "correlations"
        ? scaleLinear([-1.0, 1.0], ["#ef4444", "#22c55e"])
        : scaleLinear([0.0, 1], ["#ffffff", "#0000ef"]);
    return barData.map((data) => ({ ...data, fill: getColor(data.score) }));
  }, [barData, predictionMetric]);

  let chartName = "Unknown";
  if (predictionMetric === "importance") {
    chartName = "Metric Importance";
  } else if (predictionMetric === "correlations") {
    chartName = "Metric Correlation";
  } else if (predictionMetric === "feature-precision") {
    chartName = "Label Precision";
  } else if (predictionMetric === "feature-recall") {
    chartName = "Label Recall";
  }

  return (
    <ResponsiveContainer
      width="100%"
      height={Math.max(40 * barData.length, 120)}
    >
      <BarChart data={formattedBarData} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" />
        <YAxis
          name={metricsSummary != null ? "Metric" : "Class"}
          type="category"
          dataKey="group"
        />
        <XAxis
          name="Score"
          type="number"
          domain={
            predictionMetric === "correlations" ? [-1.0, 1.0] : [0.0, 1.0]
          }
        />
        <Tooltip formatter={formatTooltip} />
        <Legend />
        <Bar
          dataKey="score"
          name={chartName}
          fill={predictionMetric === "correlations" ? "#2b8d4d" : "#5e5ec4"}
          isAnimationActive={false}
          layout="vertical"
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

export default ActiveChartPredictionMetricVBar;