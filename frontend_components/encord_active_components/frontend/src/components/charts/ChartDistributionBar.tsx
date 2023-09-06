import * as React from "react";
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
import { formatTooltip } from "../util/Formatter";
import { QueryContext } from "../../hooks/Context";
import { useProjectAnalysisDistribution } from "../../hooks/queries/useProjectAnalysisDistribution";
import { useProjectAnalysisSummary } from "../../hooks/queries/useProjectAnalysisSummary";
import {
  AnalysisDomain,
  ProjectDomainSummary,
  QuerySummary,
} from "../../openapi/api";

export function ChartDistributionBar(props: {
  metricsSummary: ProjectDomainSummary;
  analysisSummary?: undefined | QuerySummary;
  analysisDomain: AnalysisDomain;
  projectHash: string;
  queryContext: QueryContext;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    projectHash,
    queryContext,
    metricsSummary,
    analysisSummary,
    analysisDomain,
    featureHashMap,
  } = props;
  const [showQuartiles, setShowQuartiles] = useState(false);

  const allProperties = useMemo(() => {
    const properties = Object.entries(metricsSummary.metrics)
      .filter(([metricKey]) => {
        if (analysisSummary == null) {
          return true;
        }
        const value = analysisSummary.metrics[metricKey];

        return value == null || value.count > 0;
      })
      .map(([metricKey, metric]) => ({
        label: metric?.title ?? metricKey,
        value: metricKey,
      }));
    Object.entries(metricsSummary.enums).forEach(([enumName, enumMeta]) => {
      properties.push({
        label: enumMeta?.title ?? enumName,
        value: enumName,
      });
    });
    return properties;
  }, [metricsSummary, analysisSummary]);
  const [selectedProperty, setSelectedProperty] = useState<
    undefined | string
  >();

  const groupingData = useProjectAnalysisDistribution(
    queryContext,
    projectHash,
    analysisDomain,
    selectedProperty ?? "",
    undefined,
    undefined,
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
    let keyValues: Readonly<Record<string, string | undefined>> | undefined;
    if (!isMetric) {
      results.sort((a, b) => b.count - a.count);
      const median = results[(results.length / 2) | 0];
      if (median !== undefined) {
        getFill = (score) =>
          score <= median.count / 2.0 ? "#ef4444" : "#ffa600";
      }
      const enumEntry =
        selectedProperty != null
          ? metricsSummary.enums[selectedProperty]
          : null;
      if (enumEntry != null) {
        keyValues = enumEntry.values;
      }
    } else {
      results.sort((a, b) => Number(a.group) - Number(b.group));
    }
    const getGroupName = (group: string | number): string | number => {
      console.log("ke", keyValues, selectedProperty, group);
      if (selectedProperty === "feature_hash") {
        return featureHashMap[group]?.name ?? group;
      } else if (keyValues !== undefined) {
        return keyValues[group] ?? group;
      } else {
        return group;
      }
    };

    return results.map((grouping) => ({
      ...grouping,
      group: getGroupName(isMetric ? Number(grouping.group) : grouping.group),
      fill: getFill(grouping.count),
    }));
  }, [featureHashMap, groupingData.data, isMetric, selectedProperty]);

  const lookupGrouping = (value: number) =>
    Number(
      barData.find((candidate) => Number(candidate.group) >= value)?.group
    );

  const allMetricsSummary = useProjectAnalysisSummary(
    queryContext,
    projectHash,
    analysisDomain,
    undefined,
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
      const hasFeatureHash =
        allProperties.find((value) => value.value === "feature_hash") != null;
      const chosenProperty = allProperties[allProperties.length - 1];
      if (hasFeatureHash) {
        setSelectedProperty("feature_hash");
      } else if (chosenProperty !== undefined) {
        setSelectedProperty(chosenProperty.value);
      }
    }
  }, [selectedProperty, allProperties]);

  // Correct layout for reference line
  const referenceLines = (metadata: QuerySummary["metrics"][string]) => {
    const lineQ1 = lookupGrouping(metadata?.q1 ?? 0.0);
    const lineMedian = lookupGrouping(metadata?.median ?? 0.0);
    const lineQ3 = lookupGrouping(metadata?.q3 ?? 0.0);

    return (
      <>
        <ReferenceLine
          label={`Q1${
            lineQ1 === lineMedian
              ? `, Median${lineQ3 === lineMedian ? ", Q3" : ""}`
              : ""
          }`}
          stroke="black"
          strokeDasharray="3 3"
          x={lineQ1}
        />
        {lineQ1 !== lineMedian ? (
          <ReferenceLine
            label={`Median${lineQ3 === lineMedian ? ", Q3" : ""}`}
            stroke="black"
            strokeDasharray="3 3"
            x={lineMedian}
          />
        ) : null}
        {lineMedian !== lineQ3 ? (
          <ReferenceLine
            label="Q3"
            stroke="black"
            strokeDasharray="3 3"
            x={lineQ3}
          />
        ) : null}
      </>
    );
  };

  return (
    <>
      <Space align="center" wrap>
        <Typography.Text strong>Metric or Property: </Typography.Text>
        <Select
          value={selectedProperty}
          onChange={setSelectedProperty}
          options={allProperties}
          style={{ width: 265 }}
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
            type={isMetric && barData.length > 1 ? "number" : "category"}
            domain={
              isMetric && barData.length > 1
                ? [barData[0].group, barData[barData.length - 1].group]
                : undefined
            }
            padding="no-gap"
            label={{
              value: "Metrics",
              angle: 0,
              position: "insideBottom",
              offset: -3,
            }}
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
          {metadata === undefined || !showQuartiles
            ? null
            : referenceLines(metadata)}
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}
