import * as React from "react";
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
import { useMemo } from "react";
import { ProjectAnalysisSummary, ProjectMetricSummary } from "../Types";

export function ChartOutlierSummaryBar(props: {
  summaryData: ProjectAnalysisSummary | undefined;
  metricsSummary: ProjectMetricSummary;
}) {
  const { metricsSummary, summaryData } = props;

  // Derived: Bar data on outliers
  const barData = useMemo(() => {
    if (summaryData == null) {
      return null;
    }
    const filteredData = Object.entries(summaryData.metrics)
      .map(([metric, metricData]) => ({
        metric: metricsSummary.metrics[metric]?.title ?? null,
        moderate: metricData.moderate,
        severe: metricData.severe,
      }))
      .filter(({ moderate, severe }) => moderate + severe > 0);

    filteredData.sort((outliers1, outliers2) => {
      if (outliers1.severe > outliers2.severe) {
        return -1;
      }
      if (outliers1.severe < outliers2.severe) {
        return 1;
      }
      if (outliers1.moderate > outliers2.moderate) {
        return -1;
      }
      if (outliers1.moderate < outliers2.moderate) {
        return 1;
      }
      return 0;
    });

    return filteredData;
  }, [summaryData, metricsSummary.metrics]);

  // Bar chart
  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={barData ?? []} className="active-chart">
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="metric"
          name="Metric"
          label={{
            value: "Metrics",
            angle: 0,
            position: "insideBottom",
            offset: -3,
          }}
        />
        <YAxis
          name="Outlier Count"
          label={{
            value: "Outlier count",
            angle: -90,
            position: "insideLeft",
          }}
        />
        <Tooltip />
        <Legend />
        <Bar
          dataKey="severe"
          name="Severe"
          fill="#ff6347"
          isAnimationActive={false}
        />
        <Bar
          dataKey="moderate"
          name="Moderate"
          fill="#ffa600"
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
