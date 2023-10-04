import { useState } from "react";
import { Tabs } from "antd";
import { SummaryTab } from "./AnalyticsTab";
import { ProjectSummary } from "../../openapi/api";
import { FeatureHashMap } from "../Types";

export function SummaryView(props: {
  projectHash: string;
  projectSummary: ProjectSummary;
  featureHashMap: FeatureHashMap;
}) {
  const { projectSummary, projectHash, featureHashMap } = props;
  const [currentTab, setCurrentTab] = useState<string>("0");

  return (
    <Tabs
      className="h-full overflow-y-auto p-4"
      items={[
        {
          label: "Data",
          key: "0",
          children: (
            <SummaryTab
              projectHash={projectHash}
              metricsSummary={projectSummary.data}
              analysisDomain="data"
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: "Annotations",
          key: "1",
          children: (
            <SummaryTab
              projectHash={projectHash}
              metricsSummary={projectSummary.annotation}
              analysisDomain="annotation"
              featureHashMap={featureHashMap}
            />
          ),
        },
      ]}
      activeKey={currentTab}
      onChange={setCurrentTab}
    />
  );
}
