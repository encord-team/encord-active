import * as React from "react";
import { useState } from "react";
import { Tabs } from "antd";
import { SummaryTab } from "./SummaryTab";
import { ProjectSummary } from "../../openapi/api";

export function SummaryView(props: {
  projectHash: string;
  projectSummary: ProjectSummary;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { projectSummary, projectHash, featureHashMap } = props;
  const [currentTab, setCurrentTab] = useState<string>("0");

  return (
    <Tabs
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