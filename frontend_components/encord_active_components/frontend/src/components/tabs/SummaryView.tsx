import * as React from "react";
import { useState } from "react";
import { Tabs } from "antd";
import { QueryAPI, ProjectSummary } from "../Types";
import { SummaryTab } from "./SummaryTab";

export function SummaryView(props: {
  projectHash: string;
  queryAPI: QueryAPI;
  projectSummary: ProjectSummary;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { queryAPI, projectSummary, projectHash, featureHashMap } = props;
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
              queryAPI={queryAPI}
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
              queryAPI={queryAPI}
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
