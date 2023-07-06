import * as React from "react";
import { useMemo, useState } from "react";
import { Divider, Select, Space, Tabs, Typography } from "antd";
import { ActiveProjectMetricSummary, ActiveQueryAPI } from "../../ActiveTypes";
import ActivePredictionSummaryTab from "./ActivePredictionSummaryTab";
import ActivePredictionsMetricPerformanceTab from "./ActivePredictionsMetricPerformanceTab";
import ActivePredictionsExplorerTab from "./ActivePredictionsExplorerTab";
import { Explorer } from "../../../../explorer";

function ActivePredictionsTab(props: {
  queryAPI: ActiveQueryAPI;
  projectHash: string;
  metricsSummary: ActiveProjectMetricSummary;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { queryAPI, projectHash, metricsSummary, featureHashMap } = props;
  const [predictionHash, setPredictionHash] = useState<undefined | string>();
  const { data: allPredictions } =
    queryAPI.useProjectListPredictions(projectHash);
  const allPredictionOptions = useMemo(
    () =>
      allPredictions?.results?.map((prediction) => ({
        label: prediction.name,
        value: prediction.prediction_hash,
      })),
    [allPredictions]
  );
  const [currentTab, setCurrentTab] = useState("0");

  return (
    <>
      <Space align="center" wrap>
        <Typography.Text strong>Prediction: </Typography.Text>
        <Select
          value={predictionHash}
          options={allPredictionOptions ?? []}
          onChange={setPredictionHash}
        />
      </Space>
      <Divider />
      {predictionHash === undefined ? null : (
        <Tabs
          items={[
            {
              label: "Summary",
              key: "0",
              children: (
                <ActivePredictionSummaryTab
                  projectHash={projectHash}
                  predictionHash={predictionHash}
                  queryAPI={queryAPI}
                  metricsSummary={metricsSummary}
                  featureHashMap={featureHashMap}
                />
              ),
            },
            {
              label: "Metric Performance",
              key: "1",
              children: (
                <ActivePredictionsMetricPerformanceTab
                  projectHash={projectHash}
                  predictionHash={predictionHash}
                  queryAPI={queryAPI}
                  metricsSummary={metricsSummary}
                  featureHashMap={featureHashMap}
                />
              ),
            },
            {
              label: "Explorer",
              key: "2",
              children: (
                <Explorer
                  baseUrl="http://localhost:8502"
                  projectHash={projectHash}
                  scope={"prediction"}
                  featureHashMap={featureHashMap}
                  metricsSummary={metricsSummary}
                  queryAPI={queryAPI}
                />
              ),
            },
          ]}
          activeKey={currentTab}
          onChange={setCurrentTab}
        />
      )}
    </>
  );
}

export default ActivePredictionsTab;
