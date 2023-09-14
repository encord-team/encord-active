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
import { useProjectCompareDissimilarity } from "../../hooks/queries/useProjectCompareDissimilarity";
import { AnalysisDomain, ProjectDomainSummary } from "../../openapi/api";

export function ChartMetricDissimilarity(props: {
  projectHash: string;
  compareProjectHash: string;
  analysisDomain: AnalysisDomain;
  metricsSummary: ProjectDomainSummary;
}) {
  const { projectHash, compareProjectHash, analysisDomain, metricsSummary } =
    props;

  const { data } = useProjectCompareDissimilarity(
    projectHash,
    analysisDomain,
    compareProjectHash
  );

  const barData = useMemo(() => {
    if (data === undefined) {
      return [];
    }
    const getColor = scaleLinear([0.0, 1], ["#7bfdee", "#0000ef"]);
    const results = Object.entries(data.results).map(
      ([metricKey, dissimilarity]) => ({
        metric: metricsSummary.metrics[metricKey]?.title ?? metricKey,
        dissimilarity,
        fill: getColor(dissimilarity ?? 0.0),
      })
    );
    results.sort((a, b) => (b.dissimilarity ?? 0.0) - (a.dissimilarity ?? 0.0));
    return results;
  }, [data, metricsSummary]);

  return (
    <ResponsiveContainer
      width="100%"
      height={Math.max(40 * barData.length, 120)}
    >
      <BarChart data={barData} layout="vertical" className="active-chart">
        <CartesianGrid strokeDasharray="3 3" />
        <YAxis name="Metric" type="category" dataKey="metric" width={150} />
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
