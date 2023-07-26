import { Checkbox, Select, Space, Typography } from "antd";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ProjectAnalysisDomain,
  ProjectMetricSummary,
  QueryAPI,
} from "../Types";
import { formatTooltip } from "../util/Formatter";

export function ChartDistributionBar(props: {
  metricsSummary: ProjectMetricSummary;
  analysisDomain: ProjectAnalysisDomain;
  projectHash: string;
  queryAPI: QueryAPI;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    projectHash,
    queryAPI,
    metricsSummary,
    analysisDomain,
    featureHashMap,
  } = props;
  const [showQuartiles, setShowQuartiles] = useState(false);

  const allProperties = useMemo(() => {
    const properties = Object.entries(metricsSummary.metrics).map(
      ([metricKey, metric]) => ({
        label: metric.title,
        value: metricKey,
      })
    );
    Object.entries(metricsSummary.enums).forEach(([enumName, enumMeta]) => {
      properties.push({
        label: enumMeta.title,
        value: enumName,
      });
    });
    return properties;
  }, [metricsSummary]);
  const [selectedProperty, setSelectedProperty] = useState<
    undefined | string
  >();

  const groupingData = queryAPI.useProjectAnalysisDistribution(
    projectHash,
    analysisDomain,
    selectedProperty ?? "",
    { enabled: selectedProperty !== undefined }
  );

  const isMetric =
    selectedProperty !== undefined &&
    selectedProperty in metricsSummary.metrics;

  const barData = useMemo(() => {
    if (groupingData.data == null) {
      return [];
    }
    let getFill: (score: number) => string = () => "#ffa600";
    const results = [...groupingData.data.results];
    if (!isMetric) {
      results.sort((a, b) => b.count - a.count);
      const median = results[(results.length / 2) | 0];
      if (median !== undefined) {
        getFill = (score) => (score < median.count ? "#ef4444" : "#ffa600");
      }
    } else {
      results.sort((a, b) => Number(a.group) - Number(b.group));
    }
    return results.map((grouping) => ({
      ...grouping,
      group:
        (selectedProperty === "feature_hash"
          ? featureHashMap[grouping.group]?.name
          : null) ?? grouping.group,
      fill: getFill(grouping.count),
    }));
  }, [featureHashMap, groupingData.data, isMetric, selectedProperty]);

  const lookupGrouping = (value: number) =>
    Number(
      barData.find((candidate) => Number(candidate.group) >= value)?.group
    );

  const allMetricsSummary = queryAPI.useProjectAnalysisSummary(
    projectHash,
    analysisDomain,
    { enabled: isMetric && showQuartiles }
  );
  const metadata =
    allMetricsSummary.data == null || selectedProperty === undefined
      ? undefined
      : allMetricsSummary.data.metrics[selectedProperty];

  useEffect(() => {
    if (
      selectedProperty === undefined ||
      allProperties.findIndex((value) => value.value === selectedProperty) ===
        -1
    ) {
      const chosenProperty = allProperties[allProperties.length - 1];
      if (chosenProperty !== undefined) {
        setSelectedProperty(chosenProperty.value);
      }
    }
  }, [selectedProperty, allProperties]);

  return (
    <>
      <Space align="center" wrap>
        <Typography.Text strong>Metric or Property: </Typography.Text>
        <Select
          value={selectedProperty}
          onChange={setSelectedProperty}
          options={allProperties}
          style={{width: 265}}
        />
        {!isMetric ? null : (
          <>
            <Typography.Text strong>Show quartiles: </Typography.Text>
            <Checkbox
              checked={showQuartiles}
              onChange={() => setShowQuartiles((e) => !e)}
            />
          </>
        )}
      </Space>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={barData} className="active-chart">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="group"
            label={{ value: "Metrics", angle: 0, position: "insideBottom", offset: -3, }}
          />
          <YAxis
            label={{
              value: "Count",
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip formatter={formatTooltip} />
          <Bar dataKey="count" isAnimationActive={false} />
          {metadata === undefined || !showQuartiles ? null : (
            <>
              <ReferenceLine
                label="Q1"
                stroke="black"
                strokeDasharray="3 3"
                x={lookupGrouping(metadata.q1)}
              />
              <ReferenceLine
                label="Median"
                stroke="black"
                strokeDasharray="3 3"
                x={lookupGrouping(metadata.median)}
              />
              <ReferenceLine
                label="Q3"
                stroke="black"
                strokeDasharray="3 3"
                x={lookupGrouping(metadata.q3)}
              />
            </>
          )}
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}
