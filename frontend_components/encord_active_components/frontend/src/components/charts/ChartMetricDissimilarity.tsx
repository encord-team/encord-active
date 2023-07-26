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
import { scaleLinear } from "d3-scale";
import {
  ProjectAnalysisDomain,
  ProjectMetricSummary,
  QueryAPI,
} from "../Types";

export function ChartMetricDissimilarity(props: {
  projectHash: string;
  compareProjectHash: string;
  analysisDomain: ProjectAnalysisDomain;
  metricsSummary: ProjectMetricSummary;
  queryAPI: QueryAPI;
}) {
  const {
    projectHash,
    compareProjectHash,
    analysisDomain,
    metricsSummary,
    queryAPI,
  } = props;

  const { data } = queryAPI.useProjectAnalysisCompareMetricDissimilarity(
    projectHash,
    analysisDomain,
    compareProjectHash
  );

  const barData = useMemo(() => {
    if (data === undefined) {
      return [];
    }
    const getColor = scaleLinear([0.0, 1], ["#ffffff", "#0000ef"]);
    const results = Object.entries(data.results).map(
      ([metricKey, dissimilarity]) => ({
        metric: metricsSummary.metrics[metricKey]?.title ?? metricKey,
        dissimilarity,
        fill: getColor(dissimilarity),
      })
    );
    results.sort((a, b) => b.dissimilarity - a.dissimilarity);
    return results;
  }, [data, metricsSummary]);

  return (
    <ResponsiveContainer
      width="100%"
      height={Math.max(40 * barData.length, 120)}
    >
      <BarChart data={barData} layout="vertical" className="active-chart">
        <CartesianGrid strokeDasharray="3 3" />
        <YAxis name="Metric" type="category" dataKey="metric" />
        <XAxis name="Score" type="number" domain={[0.0, 1.0]} />
        <Tooltip />
        <Legend />
        <Bar
          dataKey="dissimilarity"
          name="Metric Dissimilarity"
          fill="#5e5ec4"
          isAnimationActive={false}
          layout="vertical"
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
