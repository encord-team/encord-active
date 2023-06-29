import { Card, Divider, Row, Slider, Statistic, Typography } from "antd";
import * as React from "react";
import { useState } from "react";
import { useDebounce } from "usehooks-ts";
import ActiveChartPredictionMetricVBar from "../../charts/ActiveChartPredictionMetricVBar";
import { ActiveProjectMetricSummary, ActiveQueryAPI } from "../../ActiveTypes";
import ActiveChartPredictionRecallCurve from "../../charts/ActiveChartPredictionRecallCurve";

function ActivePredictionSummaryTab(props: {
  metricsSummary: ActiveProjectMetricSummary;
  predictionHash: string;
  projectHash: string;
  queryAPI: ActiveQueryAPI;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    metricsSummary,
    predictionHash,
    projectHash,
    queryAPI,
    featureHashMap,
  } = props;
  const [iou, setIOU] = useState(0.5);
  const debounceIOU = useDebounce(iou, 1000);

  const { data: predictionSummaryData } = queryAPI.useProjectPredictionSummary(
    projectHash,
    predictionHash,
    debounceIOU
  );

  return (
    <>
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
      <Divider />
      <Row wrap justify="space-evenly">
        <Card bordered={false} loading={predictionSummaryData == null}>
          <Statistic
            title="mAP"
            value={(predictionSummaryData?.mAP ?? 0).toFixed(3)}
          />
        </Card>
        <Card bordered={false} loading={predictionSummaryData == null}>
          <Statistic
            title="mAR"
            value={(predictionSummaryData?.mAR ?? 0).toFixed(3)}
          />
        </Card>
      </Row>
      <Divider orientation="left">
        <Typography.Title level={3}>Metric Importance</Typography.Title>
      </Divider>
      <ActiveChartPredictionMetricVBar
        metricsSummary={metricsSummary}
        predictionMetric="importance"
        data={predictionSummaryData?.importance}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>Metric Correlation</Typography.Title>
      </Divider>
      <ActiveChartPredictionMetricVBar
        metricsSummary={metricsSummary}
        predictionMetric="correlations"
        data={predictionSummaryData?.correlation}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>
          Per Class Average Precision
        </Typography.Title>
      </Divider>
      <ActiveChartPredictionMetricVBar
        predictionMetric="feature-precision"
        data={predictionSummaryData?.precisions}
        featureHashMap={featureHashMap}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>Per Class Average Recall</Typography.Title>
      </Divider>
      <ActiveChartPredictionMetricVBar
        predictionMetric="feature-recall"
        data={predictionSummaryData?.recalls}
        featureHashMap={featureHashMap}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>
          Per Class Precision-Recall Curve
        </Typography.Title>
      </Divider>
      <ActiveChartPredictionRecallCurve
        data={predictionSummaryData?.prs ?? {}}
        featureHashMap={featureHashMap}
      />
    </>
  );
}

export default ActivePredictionSummaryTab;
