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
import { useProjectAnalysisDistribution } from "../../hooks/queries/useProjectAnalysisDistribution";
import { useProjectAnalysisSummary } from "../../hooks/queries/useProjectAnalysisSummary";
import {
  AnalysisDomain,
  ProjectCollaboratorEntry,
  ProjectDomainSummary,
  QuerySummary,
} from "../../openapi/api";
import { useProjectListCollaborators } from "../../hooks/queries/useProjectListCollaborators";

export function ChartDistributionBar(props: {
  metricsSummary: ProjectDomainSummary;
  analysisSummary?: undefined | QuerySummary;
  analysisDomain: AnalysisDomain;
  projectHash: string;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    projectHash,
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
    Object.entries(metricsSummary.enums)
      .filter(([_, enumMeta]) => enumMeta.type !== "tags")
      .forEach(([enumName, enumMeta]) => {
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
    projectHash,
    analysisDomain,
    selectedProperty ?? "",
    undefined,
    undefined,
    { enabled: selectedProperty !== undefined }
  );
  const { data: collaborators } = useProjectListCollaborators(projectHash, {
    enabled: selectedProperty === "annotation_user_id",
  });
  const collaboratorsMap = useMemo(() => {
    const map: Record<string, ProjectCollaboratorEntry> = {};
    collaborators?.forEach((v) => {
      map[v.id] = v;
    });
    return map;
  }, [collaborators]);

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
      if (selectedProperty === "feature_hash") {
        return featureHashMap[group]?.name ?? group;
      } else if (selectedProperty === "annotation_user_id") {
        return collaboratorsMap[group]?.email ?? group;
      } else if (keyValues != null) {
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
  }, [
    featureHashMap,
    groupingData.data,
    isMetric,
    selectedProperty,
    metricsSummary.enums,
    collaboratorsMap,
  ]);

  const lookupGrouping = (value: number) =>
    Number(
      barData.find((candidate) => Number(candidate.group) >= value)?.group
    );

  const allMetricsSummary = useProjectAnalysisSummary(
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
          className="w-64"
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
