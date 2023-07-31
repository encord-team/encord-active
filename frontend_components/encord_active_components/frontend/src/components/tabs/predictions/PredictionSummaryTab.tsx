import { Card, Divider, Row, Slider, Statistic, Typography } from "antd";
import {useMemo, useState} from "react";
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

  const classificationOnlySummary = useMemo(() => {
    if (predictionSummaryData == null || !classificationOnlyProject) {
      return null;
    }
    const featureHashes: Record<string, { fp: number, fn: number, tp: number }> = {};
    Object.entries(predictionSummaryData.tp).forEach(([f, tp]) => {
      featureHashes[f] = { tp, fn: 0, fp: 0};
    });
    Object.entries(predictionSummaryData.fp).forEach(([f, fp]) => {
      if (f in featureHashes) {
        featureHashes[f].fp = fp;
      } else {
        featureHashes[f] = { fp, tp: 0, fn: 0}
      }
    });
    Object.entries(predictionSummaryData.fn).forEach(([f, fn]) => {
      if (f in featureHashes) {
        featureHashes[f].fn = fn;
      } else {
        featureHashes[f] = { fn, tp: 0, fp: 0}
      }
    });
    const perFeatureProps = Object.values(featureHashes).map(({fp, fn, tp}) => {
      const p = (tp + fp) === 0 ? 0 : tp / (tp + fp);
      const r = (tp + fn) === 0 ? 0 : tp / (tp + fn);
      return {
        p,
        r,
        f1: (p + r) === 0 ? 0 : (2 * p * r) / (p + r),
      }
    });
    const featureCount = perFeatureProps.length;
    const reduced = perFeatureProps.reduce(
      (c, n) =>
        ({p: c.p + n.p, r: c.r + n.r, f1: c.f1 + n.f1}),
      {r: 0, p: 0, f1: 0}
    );
    const { tTP, tFN, tFP} = predictionSummaryData;
    const tTN = (
      predictionSummaryData.num_frames - tTP - tFN // FIXME: FP <= FN (and cover same values)
    );
    console.log({tTP, tTN, tFP, tFN, NF: predictionSummaryData.num_frames})
    return {
      a: (tTP + tTN) / predictionSummaryData.num_frames,
      r: reduced.r / featureCount,
      p: reduced.p / featureCount,
      f1: reduced.f1 / featureCount,
    }
  }, [predictionSummaryData, classificationOnlyProject]);

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
        {classificationOnlySummary ? (
          <>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Mean Precision"
                value={classificationOnlySummary.p.toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Mean Recall"
                value={classificationOnlySummary.r.toFixed(3)}
              />
            </Card>
            <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Accuracy"
                value={classificationOnlySummary.a.toFixed(3)}
              />
            </Card>
             <Card bordered={false} loading={predictionSummaryData == null}>
              <Statistic
                title="Average F1"
                value={classificationOnlySummary.f1.toFixed(3)}
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
