import { Card, Divider, Row, Slider, Statistic, Typography } from "antd";
import { useState } from "react";
import { useDebounce } from "usehooks-ts";
import { ChartPredictionMetricVBar } from "../../charts/ChartPredictionMetricVBar";
import { ProjectMetricSummary, QueryAPI } from "../../Types";
import { ChartPredictionRecallCurve } from "../../charts/ChartPredictionRecallCurve";

export function PredictionSummaryTab(props: {
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

  const classificationOnlyProject = predictionSummaryData?.classification_only !== false;

  // FIXME: expose all information for classifications!!
  // classification
  // Average Precision = TP / (TP + FP)
  // Average Recall = TP / (TP + FN)
  // Average Accuracy = (TP + TN) / (TP + FP + FN) {FN makes only fo classifications}
  // FN (currently) = {num_frames * feature_hashes} - tp - fp - fn
  // So need num_frames as exposed value (only used for classification project).
  const cTP = predictionSummaryData?.tTP ?? 0;
  const cFP = predictionSummaryData?.tFP ?? 0;
  const cFN = predictionSummaryData?.tFN ?? 0;
  const cNF = predictionSummaryData?.num_frames ?? 0;
  const cTN = cNF - cFN - cFP - cTP;// FIXME: wrong if multiple classifications per frame is allowed

  return (
    <>
       {classificationOnlyProject ? null : (
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
         </>
      )}
      <Row wrap justify="space-evenly">
        {classificationOnlyProject ? (
          <>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Average Precision"
                value={(
                    cTP / (cTP + cFP)
                ).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Average Recall"
                value={(
                    cTP / (cTP + cFN)
                ).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Average Accuracy"
                value={(
                    (cTP + cTN) / (cTP + cFN + cFP)
                ).toFixed(3)}
              />
            </Card>
          </>
        ) : (
            <>
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
            </>
        )}
        <Card bordered={false} loading={predictionSummaryData == null}>
          <Statistic
            title="True Positives"
            value={(predictionSummaryData?.tTP ?? 0).toFixed(0)}
          />
        </Card>
        <Card bordered={false} loading={predictionSummaryData == null}>
          <Statistic
            title="False Positives"
            value={(predictionSummaryData?.tFP ?? 0).toFixed(0)}
          />
        </Card>
        <Card bordered={false} loading={predictionSummaryData == null}>
          <Statistic
            title="False Negatives"
            value={(predictionSummaryData?.tFN ?? 0).toFixed(0)}
          />
        </Card>
        <Card bordered={false} loading={predictionSummaryData == null}>
          <Statistic
            title="Labels"
            value={(
              (predictionSummaryData?.tTP ?? 0) +
              (predictionSummaryData?.tFN ?? 0)
            ).toFixed(0)}
          />
        </Card>
      </Row>
      <Divider orientation="left">
        <Typography.Title level={3}>Metric Importance</Typography.Title>
      </Divider>
      <ChartPredictionMetricVBar
        metricsSummary={metricsSummary}
        predictionMetric="importance"
        data={predictionSummaryData?.importance}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>Metric Correlation</Typography.Title>
      </Divider>
      <ChartPredictionMetricVBar
        metricsSummary={metricsSummary}
        predictionMetric="correlations"
        data={predictionSummaryData?.correlation}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>
          Per Class Average Precision
        </Typography.Title>
      </Divider>
      <ChartPredictionMetricVBar
        predictionMetric="feature-precision"
        data={predictionSummaryData?.precisions}
        featureHashMap={featureHashMap}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>Per Class Average Recall</Typography.Title>
      </Divider>
      <ChartPredictionMetricVBar
        predictionMetric="feature-recall"
        data={predictionSummaryData?.recalls}
        featureHashMap={featureHashMap}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>
          Per Class Precision-Recall Curve
        </Typography.Title>
      </Divider>
      <ChartPredictionRecallCurve
        data={predictionSummaryData?.prs ?? {}}
        featureHashMap={featureHashMap}
      />
    </>
  );
}
