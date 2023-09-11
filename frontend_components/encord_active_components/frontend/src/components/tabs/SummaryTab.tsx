import * as React from "react";
import { useMemo } from "react";
import { Card, Space, Row, Typography, Statistic, Divider } from "antd";
import {
  CheckCircleOutlined,
  DatabaseOutlined,
  FullscreenOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { isEmpty } from "radash";
import { ChartMetricCompareScatter } from "../charts/ChartMetricCompareScatter";
import { ChartDistributionBar } from "../charts/ChartDistributionBar";
import { ChartOutlierSummaryBar } from "../charts/ChartOutlierSummaryBar";
import { useProjectAnalysisSummary } from "../../hooks/queries/useProjectAnalysisSummary";
import { AnalysisDomain, ProjectDomainSummary } from "../../openapi/api";

const AnalysisDomainToName: Record<AnalysisDomain, string> = {
  data: "Frames",
  annotation: "Annotations",
};

export function SummaryTab(props: {
  projectHash: string;
  metricsSummary: ProjectDomainSummary;
  analysisDomain: AnalysisDomain;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { projectHash, metricsSummary, analysisDomain, featureHashMap } = props;
  const summary = useProjectAnalysisSummary(projectHash, analysisDomain);
  const { data } = summary;

  // Derived: Total outliers
  const [totalSevereOutlier, totalModerateOutlier, totalMetricOutlier] =
    useMemo(() => {
      if (data == null || isEmpty(data.metrics)) {
        return [0, 0, 0];
      }
      const severe = Object.values(data.metrics)
        .map((metric) => metric?.severe ?? 0)
        .reduce((a, b) => a + b);
      const moderate = Object.values(data.metrics)
        .map((metric) => metric?.moderate ?? 0)
        .reduce((a, b) => a + b);
      const metrics = Object.values(data.metrics).filter(
        (metric) => (metric?.severe ?? 0) > 0 || (metric?.moderate ?? 0) > 0
      ).length;

      return [severe, moderate, metrics];
    }, [data]);

  if (data?.count === 0) {
    // Summary is useless as no data is present.
    return (
      <Typography.Title>
        No {AnalysisDomainToName[analysisDomain]} to analyse.
      </Typography.Title>
    );
  }

  return (
    <Space direction="vertical" style={{ width: "100%" }}>
      <Row wrap justify="space-evenly">
        <Card bordered={false} loading={data == null}>
          <Statistic
            title={`Number of ${AnalysisDomainToName[analysisDomain]}`}
            value={data?.count ?? 0}
            prefix={<DatabaseOutlined />}
          />
        </Card>
        <Card bordered={false} loading={data == null}>
          <Statistic
            title="Severe Outliers"
            value={totalSevereOutlier}
            valueStyle={{
              color: totalSevereOutlier === 0 ? "#3f8600" : "#cf1322",
            }}
            prefix={
              totalSevereOutlier === 0 ? (
                <CheckCircleOutlined />
              ) : (
                <WarningOutlined />
              )
            }
          />
        </Card>
        <Card bordered={false} loading={data == null}>
          <Statistic
            title="Moderate Outliers"
            value={totalModerateOutlier}
            valueStyle={{
              color: totalModerateOutlier === 0 ? "#3f8600" : "#ec9c27",
            }}
            prefix={
              totalModerateOutlier === 0 ? (
                <CheckCircleOutlined />
              ) : (
                <WarningOutlined />
              )
            }
          />
        </Card>
        {data == null ||
        ("metric_width" in data.metrics &&
          "metric_height" in data.metrics &&
          (data.metrics.metric_height?.count ?? 0) > 0) ? (
          <Card bordered={false} loading={data == null}>
            <Statistic
              title="Median Image Size"
              value={
                data == null
                  ? ""
                  : `${data.metrics.metric_width?.median}x${data.metrics.metric_height?.median}`
              }
              prefix={<FullscreenOutlined />}
            />
          </Card>
        ) : null}
      </Row>
      {data != null && totalMetricOutlier === 0 ? null : (
        <>
          <Divider orientation="left">
            <Typography.Title level={3}>
              {totalMetricOutlier} Metrics with outliers
            </Typography.Title>
          </Divider>
          <Card type="inner" bordered={false} loading={data == null}>
            <ChartOutlierSummaryBar
              summaryData={data}
              metricsSummary={metricsSummary}
            />
          </Card>
        </>
      )}
      <Divider orientation="left">
        <Typography.Title level={3}>2D Metrics view</Typography.Title>
      </Divider>
      <Card type="inner" bordered={false}>
        <ChartMetricCompareScatter
          metricsSummary={metricsSummary}
          analysisSummary={data}
          analysisDomain={analysisDomain}
          projectHash={projectHash}
          allowTrend
        />
      </Card>
      <Divider orientation="left">
        <Typography.Title level={3}>Metric Distribution</Typography.Title>
      </Divider>
      <Card type="inner" bordered={false}>
        <ChartDistributionBar
          metricsSummary={metricsSummary}
          analysisSummary={data}
          analysisDomain={analysisDomain}
          projectHash={projectHash}
          featureHashMap={featureHashMap}
        />
      </Card>
    </Space>
  );
}
