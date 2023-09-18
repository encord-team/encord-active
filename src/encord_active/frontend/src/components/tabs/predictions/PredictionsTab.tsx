import { useEffect, useMemo, useState } from "react";
import { Divider, Select, Space, Tabs, Typography } from "antd";
import { PredictionSummaryTab } from "./PredictionSummaryTab";
import { PredictionsMetricPerformanceTab } from "./PredictionsMetricPerformanceTab";
import { Explorer, Props as ExplorerProps } from "../../explorer";
import { ProjectDomainSummary } from "../../../openapi/api";
import { useProjectListPredictions } from "../../../hooks/queries/useProjectListPredictions";
import { FeatureHashMap } from "../../Types";

export function PredictionsTab(
  props: {
    projectHash: string;
    dataMetricsSummary: ProjectDomainSummary;
    annotationMetricsSummary: ProjectDomainSummary;
    featureHashMap: FeatureHashMap;
  } & Pick<ExplorerProps, "remoteProject" | "setSelectedProjectHash">
) {
  const {
    projectHash,
    dataMetricsSummary,
    annotationMetricsSummary,
    featureHashMap,
    remoteProject,
    setSelectedProjectHash,
  } = props;
  const [predictionHash, setPredictionHash] = useState<undefined | string>();
  const { data: allPredictions } = useProjectListPredictions(projectHash);

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
  if (
    allPredictionOptions != null &&
    allPredictionOptions.length === 0 &&
    predictionHash == null
  ) {
    return (
      <Typography.Title>No Predictions for Active Project</Typography.Title>
    );
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
              className="w-80"
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
                  metricsSummary={annotationMetricsSummary}
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
                  metricsSummary={annotationMetricsSummary}
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
                  predictionHash={predictionHash}
                  featureHashMap={featureHashMap}
                  dataMetricsSummary={dataMetricsSummary}
                  annotationMetricsSummary={annotationMetricsSummary}
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
