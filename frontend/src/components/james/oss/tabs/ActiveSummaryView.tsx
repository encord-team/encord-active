import { useEffect, useState } from "react";
import { Tabs } from "antd";
import * as React from "react";
import { useMap } from "usehooks-ts";
import {
  ActiveQueryAPI,
  ActiveProjectMetricSummary,
  ActiveProjectAnalysisDomain,
  ActiveProjectSummary,
} from "../ActiveTypes";
import { ActiveSearchTab } from "./ActiveSearchTab";
import ActiveSelectedTab from "./ActiveSelectedTab";
import ActiveSummaryTab from "./ActiveSummaryTab";
import ActiveProjectComparisonTab from "./ActiveProjectComparisonTab";
import { ActiveFilterState } from "../util/ActiveMetricFilter";
import { Explorer } from "../../../explorer";

function ActiveSummaryView(props: {
  projectHash: string;
  queryAPI: ActiveQueryAPI;
  projectSummary: ActiveProjectSummary;
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
            <ActiveSummaryTab
              projectHash={projectHash}
              queryAPI={queryAPI}
              metricsSummary={projectSummary.data}
              analysisDomain={"data"}
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: "Annotations",
          key: "1",
          children: (
            <ActiveSummaryTab
              projectHash={projectHash}
              queryAPI={queryAPI}
              metricsSummary={projectSummary.annotations}
              analysisDomain={"annotation"}
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

export default ActiveSummaryView;
