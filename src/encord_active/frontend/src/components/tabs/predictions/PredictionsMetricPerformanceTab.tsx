import { Checkbox, Col, Divider, Row, Select, Slider, Typography } from "antd";
import { useEffect, useMemo, useState } from "react";
import { useDebounce } from "usehooks-ts";
import { ChartPredictionMetricPerformanceChart } from "../../charts/ChartPredictionMetricPerformanceChart";
import { ProjectDomainSummary } from "../../../openapi/api";
import { useProjectPredictionMetricPerformance } from "../../../hooks/queries/useProjectPredictionMetricPerformance";

const bucketOptions: { label: number; value: number }[] = [
  {
    label: 1000,
    value: 1000,
  },
  {
    label: 100,
    value: 100,
  },
  {
    label: 10,
    value: 10,
  },
];

export function PredictionsMetricPerformanceTab(props: {
  metricsSummary: ProjectDomainSummary;
  predictionHash: string;
  projectHash: string;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { metricsSummary, projectHash, predictionHash, featureHashMap } = props;
  const [iou, setIOU] = useState(0.5);
  const [selectedMetric, setSelectedMetric] = useState<undefined | string>();
  const metricOptions = useMemo(
    () =>
      Object.entries(metricsSummary.metrics).map(([metricKey, metric]) => ({
        value: metricKey,
        label: metric?.title ?? metricKey,
      })),
    [metricsSummary]
  );
  const [bucketCount, setBucketCount] = useState(10);
  const [classList, setClassList] = useState<string[]>([]);
  const [showDistributionBar, setShowDistributionBar] = useState(false);

  useEffect(() => {
    if (selectedMetric == null) {
      const exists =
        metricOptions.find(({ value }) => "metric_confidence") != null;
      if (exists) {
        setSelectedMetric("metric_confidence");
      }
    }
  }, [selectedMetric, metricOptions]);

  const rawQueryState = useMemo(
    () => ({ bucketCount, iou, selectedMetric }),
    [bucketCount, iou, selectedMetric]
  );
  const debounceQueryState = useDebounce(rawQueryState);
  const queryPerformance = useProjectPredictionMetricPerformance(
    projectHash,
    predictionHash,
    debounceQueryState.iou,
    debounceQueryState.selectedMetric ?? "",
    debounceQueryState.bucketCount as 1000 | 100 | 10 | undefined,
    undefined,
    { enabled: debounceQueryState.selectedMetric != null }
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
    selectedMetric == null
      ? "No Metric Selected"
      : metricsSummary.metrics[selectedMetric]?.title ?? "Unknown";

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
              className="w-80"
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
            className="w-80"
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
        <Col span={8}>
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
        </Col>
        <Col span={8}>
          <Typography.Text strong>Show Distribution: </Typography.Text>
          <Checkbox
            checked={showDistributionBar}
            onChange={() => setShowDistributionBar((e) => !e)}
          />
        </Col>
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
        showDistributionBar={showDistributionBar}
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
        showDistributionBar={showDistributionBar}
      />
    </>
  );
}
