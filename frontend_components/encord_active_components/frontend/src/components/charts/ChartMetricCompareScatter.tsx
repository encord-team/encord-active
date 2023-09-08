import * as React from "react";
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
import { formatTooltip } from "../util/Formatter";
import { useProjectAnalysisMetricScatter } from "../../hooks/queries/useProjectAnalysisMetricScatter";
import {
  AnalysisDomain,
  ProjectDomainSummary,
  QuerySummary,
} from "../../openapi/api";

export function ChartMetricCompareScatter(props: {
  metricsSummary: ProjectDomainSummary;
  analysisSummary?: undefined | QuerySummary;
  analysisDomain: AnalysisDomain;
  projectHash: string;
  compareProjectHash?: string | undefined;
  allowTrend?: boolean;
}) {
  const {
    metricsSummary,
    analysisSummary,
    analysisDomain,
    projectHash,
    allowTrend,
    compareProjectHash,
  } = props;
  const [xMetric, setXMetric] = useState<undefined | string>();
  const [yMetric, setYMetric] = useState<undefined | string>();
  const [showTrend, setShowTrend] = useState<boolean>(true);
  const sampledState = useProjectAnalysisMetricScatter(
    projectHash,
    analysisDomain,
    xMetric ?? "",
    yMetric ?? "",
    undefined,
    undefined,
    { enabled: xMetric !== undefined && yMetric !== undefined }
  );
  const compareSampledState = useProjectAnalysisMetricScatter(
    compareProjectHash ?? "",
    analysisDomain,
    xMetric ?? "",
    yMetric ?? "",
    undefined,
    undefined,
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
  const metricOptions = useMemo(
    () =>
      Object.entries(metricsSummary.metrics)
        .filter(([metricKey]) => {
          if (analysisSummary == null) {
            return true;
          }
          const value = analysisSummary.metrics[metricKey];

          return value == null || value.count > 0;
        })
        .map(([metricKey, metricData]) => ({
          value: metricKey,
          label: metricData?.title ?? metricKey,
        })),
    [metricsSummary, analysisSummary]
  );
  useEffect(() => {
    const metrics = new Set(metricOptions.map((v) => v.value));
    let chosenXMetric = xMetric;
    if (xMetric === undefined || !metrics.has(xMetric)) {
      setXMetric(metricOptions[0]?.value);
      chosenXMetric = metricOptions[0]?.value;
    }
    if (yMetric === undefined || !metrics.has(yMetric)) {
      if (metricOptions[0]?.value !== chosenXMetric) {
        setYMetric(metricOptions[0]?.value);
      } else {
        setYMetric(metricOptions[1]?.value);
      }
    }
  }, [metricOptions, xMetric, yMetric]);

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
        <Select
          value={xMetric}
          onChange={setXMetric}
          options={metricOptions}
          style={{ width: 265 }}
        />
        <Typography.Text strong>Y Metric: </Typography.Text>
        <Select
          value={yMetric}
          onChange={setYMetric}
          options={metricOptions}
          style={{ width: 265 }}
        />
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
