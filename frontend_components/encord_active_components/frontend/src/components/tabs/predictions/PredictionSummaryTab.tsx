import * as React from "react";
import { Card, Divider, Row, Slider, Statistic, Typography } from "antd";
import { useMemo, useState } from "react";
import { useDebounce } from "usehooks-ts";
import { ChartPredictionMetricVBar } from "../../charts/ChartPredictionMetricVBar";
import { ChartPredictionRecallCurve } from "../../charts/ChartPredictionRecallCurve";
import { ProjectDomainSummary } from "../../../openapi/api";
import { QueryContext } from "../../../hooks/Context";
import { useProjectPredictionSummary } from "../../../hooks/queries/useProjectPredictionSummary";

export function PredictionSummaryTab(props: {
  metricsSummary: ProjectDomainSummary;
  predictionHash: string;
  projectHash: string;
  queryContext: QueryContext;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    metricsSummary,
    predictionHash,
    projectHash,
    queryContext,
    featureHashMap,
  } = props;
  const [iou, setIOU] = useState(0.5);
  const debounceIOU = useDebounce(iou, 1000);

  const { data: predictionSummaryData } = useProjectPredictionSummary(
    queryContext,
    projectHash,
    predictionHash,
    debounceIOU
  );

  const classificationOnlyProject =
    predictionSummaryData?.classification_only !== false;

  const [predictionSummaryPrecisions, predictionSummaryRecalls] =
    useMemo(() => {
      if (predictionSummaryData == null) {
        return [undefined, undefined];
      }
      const p = Object.entries(
        predictionSummaryData?.feature_properties ?? {}
      ).map(([k, v]) => [k, v?.ap ?? 0.0]);
      const r = Object.entries(
        predictionSummaryData?.feature_properties ?? {}
      ).map(([k, v]) => [k, v?.ar ?? 0.0]);

      return [Object.fromEntries(p), Object.fromEntries(r)];
    }, [predictionSummaryData]);

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
                title="Mean Precision"
                value={(predictionSummaryData?.mP ?? 0).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Mean Recall"
                value={(predictionSummaryData?.mR ?? 0).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Accuracy"
                value={(
                  predictionSummaryData?.classification_accuracy ?? 0
                ).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="F1"
                value={(predictionSummaryData?.mF1 ?? 0).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="TP"
                value={(predictionSummaryData?.tTP ?? 0).toFixed(0)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="FP"
                value={(predictionSummaryData?.tFP ?? 0).toFixed(0)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="FN"
                value={(predictionSummaryData?.tFN ?? 0).toFixed(0)}
              />
            </Card>
          </>
        ) : (
          <>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title={
                  <div>
                    mAP<sup>@IOU={iou}</sup>
                  </div>
                }
                value={(predictionSummaryData?.mAP ?? 0).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title={
                  <div>
                    mAR<sup>@IOU={iou}</sup>
                  </div>
                }
                value={(predictionSummaryData?.mR ?? 0).toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title={
                  <div>
                    TP<sup>@IOU={iou}</sup>
                  </div>
                }
                value={(predictionSummaryData?.tTP ?? 0).toFixed(0)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title={
                  <div>
                    FP<sup>@IOU={iou}</sup>
                  </div>
                }
                value={(predictionSummaryData?.tFP ?? 0).toFixed(0)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title={
                  <div>
                    FN<sup>@IOU={iou}</sup>
                  </div>
                }
                value={(predictionSummaryData?.tFN ?? 0).toFixed(0)}
              />
            </Card>
          </>
        )}
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
        data={predictionSummaryPrecisions}
        featureHashMap={featureHashMap}
      />
      <Divider orientation="left">
        <Typography.Title level={3}>Per Class Average Recall</Typography.Title>
      </Divider>
      <ChartPredictionMetricVBar
        predictionMetric="feature-recall"
        data={predictionSummaryRecalls}
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
