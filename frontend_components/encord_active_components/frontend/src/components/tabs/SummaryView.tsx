import * as React from "react";
import { useState } from "react";
import { Tabs } from "antd";
import { SummaryTab } from "./SummaryTab";
import { QueryContext } from "../../hooks/Context";
import { ProjectSummary } from "../../openapi/api";

export function SummaryView(props: {
  projectHash: string;
  queryContext: QueryContext;
  projectSummary: ProjectSummary;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { queryContext, projectSummary, projectHash, featureHashMap } = props;
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
              queryContext={queryContext}
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
              queryContext={queryContext}
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
