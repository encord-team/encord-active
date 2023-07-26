import { useEffect, useMemo, useState } from "react";
import { Checkbox, Select, Space, Typography } from "antd";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { scaleLinear } from "d3-scale";
import {
  ProjectAnalysisDomain,
  ProjectMetricSummary,
  QueryAPI,
} from "../Types";
import { formatTooltip } from "../util/Formatter";

export function ChartMetricCompareScatter(props: {
  metricsSummary: ProjectMetricSummary;
  analysisDomain: ProjectAnalysisDomain;
  projectHash: string;
  compareProjectHash?: string | undefined;
  queryAPI: QueryAPI;
  allowTrend?: boolean;
}) {
  const {
    metricsSummary,
    analysisDomain,
    projectHash,
    queryAPI,
    allowTrend,
    compareProjectHash,
  } = props;
  const [xMetric, setXMetric] = useState<undefined | string>();
  const [yMetric, setYMetric] = useState<undefined | string>();
  const [showTrend, setShowTrend] = useState<boolean>(true);
  useEffect(() => {
    const allMetricKeys = Object.keys(metricsSummary.metrics);
    const m1 = allMetricKeys[0] ?? "null";
    const m2 = allMetricKeys[1] ?? m1;
    setXMetric((oldM1) => {
      if (oldM1 !== undefined && oldM1 in metricsSummary.metrics) {
        return oldM1;
      } else {
        return m1;
      }
    });
    setYMetric((oldM2) => {
      if (oldM2 !== undefined && oldM2 in metricsSummary.metrics) {
        return oldM2;
      } else {
        return m2;
      }
    });
  }, [metricsSummary.metrics]);
  const sampledState = queryAPI.useProjectAnalysisMetricScatter(
    projectHash,
    analysisDomain,
    xMetric ?? "",
    yMetric ?? "",
    { enabled: xMetric !== undefined && yMetric !== undefined }
  );
  const compareSampledState = queryAPI.useProjectAnalysisMetricScatter(
    compareProjectHash ?? "",
    analysisDomain,
    xMetric ?? "",
    yMetric ?? "",
    {
      enabled:
        xMetric !== undefined &&
        yMetric !== undefined &&
        compareProjectHash != null,
    }
  );
  const sampledData = sampledState.data;
  const compareSampledData =
    compareProjectHash != null ? compareSampledState.data : null;
  const metricOptions = Object.entries(metricsSummary.metrics).map(
    ([metricKey, metricData]) => ({
      value: metricKey,
      label: metricData.title,
    })
  );

  const data = useMemo(() => {
    if (sampledData == null) {
      return null;
    }
    const getFill = scaleLinear([1, 100], ["#9090ff", "#000000"]);
    return sampledData.samples.map((sample) => ({
      ...sample,
      fill: getFill(sample.n),
    }));
  }, [sampledData]);

  return (
    <>
      <Space align="center" wrap>
        <Typography.Text strong>X Metric: </Typography.Text>
        <Select value={xMetric} onChange={setXMetric} options={metricOptions} style={{width: 265}}/>
        <Typography.Text strong>Y Metric: </Typography.Text>
        <Select value={yMetric} onChange={setYMetric} options={metricOptions} style={{width: 265}}/>
        {allowTrend ? (
          <>
            <Typography.Text strong>Show trend: </Typography.Text>
            <Checkbox
              checked={showTrend}
              onChange={() => setShowTrend((e) => !e)}
            />
          </>
        ) : null}
      </Space>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart className="active-chart">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="x"
            type="number"
            label={{
              value: metricsSummary.metrics[xMetric ?? ""]?.title ?? "x",
              angle: 0,
              position: "insideBottom",
              offset: -3,
            }}
            name={metricsSummary.metrics[xMetric ?? ""]?.title ?? "x"}
          />
          <YAxis
            dataKey="y"
            type="number"
            label={{
              value: metricsSummary.metrics[yMetric ?? ""]?.title ?? "y",
              angle: -90,
              position: "insideLeft",
            }}
            name={metricsSummary.metrics[yMetric ?? ""]?.title ?? "y"}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            formatter={formatTooltip}
          />
          {data != null ? (
            <Scatter
              data={data}
              fill="#8884d8"
              isAnimationActive={false}
              line={showTrend}
              lineType="fitting"
            />
          ) : null}
          {compareSampledData != null ? (
            <Scatter
              data={compareSampledData.samples}
              fill="#fd0a5a"
              isAnimationActive={false}
              line={showTrend}
              lineType="fitting"
            />
          ) : null}
        </ScatterChart>
      </ResponsiveContainer>
    </>
  );
}
