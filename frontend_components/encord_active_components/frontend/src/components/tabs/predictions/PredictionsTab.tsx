import {useEffect, useMemo, useState} from "react";
import { Divider, Select, Space, Tabs, Typography } from "antd";
import { ProjectMetricSummary, QueryAPI } from "../../Types";
import { PredictionSummaryTab } from "./PredictionSummaryTab";
import { PredictionsMetricPerformanceTab } from "./PredictionsMetricPerformanceTab";
import { Explorer, Props as ExplorerProps } from "../../explorer";

export function PredictionsTab(
  props: {
    queryAPI: QueryAPI;
    projectHash: string;
    metricsSummary: ProjectMetricSummary;
    featureHashMap: Record<
      string,
      { readonly color: string; readonly name: string }
    >;
  } & Pick<ExplorerProps, "remoteProject" | "setSelectedProjectHash">
) {
  const {
    queryAPI,
    projectHash,
    metricsSummary,
    featureHashMap,
    remoteProject,
    setSelectedProjectHash,
  } = props;
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

  // Auto-select prediction if one exists & unselect if an invalid prediction hash is selected;
  useEffect(() => {
    if (
      allPredictionOptions?.length &&
      !allPredictionOptions?.find((v) => v.value === predictionHash)
    ) {
      setPredictionHash(allPredictionOptions[0].value);
    }
  }, [predictionHash, allPredictionOptions]);

  // No prediction can be selected
  if (allPredictionOptions != null && allPredictionOptions.length === 0 && predictionHash == null) {
      return <Typography.Title>No Predictions for Active Project</Typography.Title>
  }

  return (
    <>
        {allPredictionOptions != null && allPredictionOptions.length > 1 ? (
            <>
                <Space align="center" wrap>
                    <Typography.Text strong>Prediction: </Typography.Text>
                    <Select
                      value={predictionHash}
                      options={allPredictionOptions ?? []}
                      onChange={setPredictionHash}
                      style={{width: 300}}
                    />
                </Space>
                <Divider />
            </>
        ) : null}
      {predictionHash === undefined ? null : (
        <Tabs
          items={[
            {
              label: "Summary",
              key: "0",
              children: (
                <PredictionSummaryTab
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
                <PredictionsMetricPerformanceTab
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
                  projectHash={projectHash}
                  scope={"prediction"}
                  featureHashMap={featureHashMap}
                  metricsSummary={metricsSummary}
                  queryAPI={queryAPI}
                  setSelectedProjectHash={setSelectedProjectHash}
                  remoteProject={remoteProject}
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
