import { Col, Divider, Row, Select, Slider, Typography } from "antd";
import { useMemo, useState } from "react";
import { ProjectMetricSummary, QueryAPI } from "../../Types";
import { ChartPredictionMetricPerformanceChart } from "../../charts/ChartPredictionMetricPerformanceChart";

const bucketOptions: { label: number; value: number }[] = [
  {
    label: 100,
    value: 100,
  },
  {
    label: 50,
    value: 50,
  },
  {
    label: 40,
    value: 40,
  },
  {
    label: 30,
    value: 30,
  },
  {
    label: 20,
    value: 20,
  },
  {
    label: 10,
    value: 10,
  },
  {
    label: 5,
    value: 5,
  },
];

export function PredictionsMetricPerformanceTab(props: {
  metricsSummary: ProjectMetricSummary;
  predictionHash: string;
  projectHash: string;
  queryAPI: QueryAPI;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    metricsSummary,
    projectHash,
    predictionHash,
    queryAPI,
    featureHashMap,
  } = props;
  const [iou, setIOU] = useState(0.5);
  const [selectedMetric, setSelectedMetric] = useState<undefined | string>();
  const metricOptions = useMemo(
    () =>
      Object.entries(metricsSummary.metrics).map(([metricKey, metric]) => ({
        value: metricKey,
        label: metric.title,
      })),
    [metricsSummary]
  );
  const [bucketCount, setBucketCount] = useState(50);
  const [classList, setClassList] = useState<string[]>([]);

  const queryPerformance = queryAPI.useProjectPredictionMetricPerformance(
    projectHash,
    predictionHash,
    bucketCount,
    iou,
    selectedMetric ?? "",
    { enabled: selectedMetric != null }
  );

  const classOptions = useMemo(
    () =>
      Object.entries(featureHashMap).map(([featureHash, feature]) => ({
        label: feature.name,
        value: featureHash,
      })),
    [featureHashMap]
  );

  const scoreLabel =
    metricsSummary.metrics[selectedMetric ?? ""]?.title ?? "Unknown";

  return (
    <>
      <Row align="middle">
        <Col span={8}>
          <Row align="middle">
            <Typography.Text strong>IOU:</Typography.Text>
            <Slider
              value={iou}
              onChange={setIOU}
              min={0}
              max={1}
              step={0.01}
              style={{ width: 300 }}
            />
          </Row>
        </Col>
        <Col span={8}>
          <Typography.Text strong>Metric:</Typography.Text>
          <Select
            allowClear
            value={selectedMetric}
            onChange={setSelectedMetric}
            options={metricOptions}
            style={{ width: 300 }}
            bordered={false}
          />
        </Col>
        <Col span={8}>
          <Typography.Text strong>Number of Buckets:</Typography.Text>
          <Select
            bordered={false}
            value={bucketCount}
            onChange={setBucketCount}
            options={bucketOptions}
          />
        </Col>
      </Row>
      <Row align="middle">
        <Typography.Text strong>Class:</Typography.Text>
        <Select
          mode="multiple"
          allowClear
          value={classList}
          onChange={setClassList}
          options={classOptions}
          style={{ width: "50%" }}
          bordered={false}
        />
      </Row>
      <Divider orientation="left">
        <Typography.Title level={3}>Precision</Typography.Title>
      </Divider>
      <ChartPredictionMetricPerformanceChart
        data={queryPerformance.data?.precision ?? {}}
        selectedClass={classList}
        classDecomposition="auto"
        featureHashMap={featureHashMap}
        scoreLabel={scoreLabel}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>False Negative Rate</Typography.Title>
      </Divider>
      <ChartPredictionMetricPerformanceChart
        data={queryPerformance.data?.fns ?? {}}
        selectedClass={classList}
        classDecomposition="auto"
        featureHashMap={featureHashMap}
        scoreLabel={scoreLabel}
      />
    </>
  );
}
