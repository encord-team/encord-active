import { useEffect, useState } from "react";
import { Tabs } from "antd";
import * as React from "react";
import { useMap } from "usehooks-ts";
import {
  ActiveQueryAPI,
  ActiveProjectMetricSummary,
  ActiveProjectAnalysisDomain,
} from "../ActiveTypes";
import { ActiveSearchTab } from "./ActiveSearchTab";
import ActiveSelectedTab from "./ActiveSelectedTab";
import ActiveSummaryTab from "./ActiveSummaryTab";
import ActiveProjectComparisonTab from "./ActiveProjectComparisonTab";
import { ActiveFilterState } from "../util/ActiveMetricFilter";
import { Explorer } from "../../../explorer";

function ActiveAnalysisDomainTab(props: {
  projectHash: string;
  editUrl?:
    | undefined
    | ((dataHash: string, projectHash: string, frame: number) => string);
  queryAPI: ActiveQueryAPI;
  metricsSummary: ActiveProjectMetricSummary;
  analysisDomain: ActiveProjectAnalysisDomain;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    editUrl,
    projectHash,
    queryAPI,
    metricsSummary,
    analysisDomain,
    featureHashMap,
  } = props;
  const [filters, setFilters] = useState<ActiveFilterState>({
    metricFilters: {},
    enumFilters: {},
    ordering: [],
  });
  const [selectedItems, setSelectedItems] = useMap<string, null>();
  const [currentTab, setCurrentTab] = useState<string>("0");

  // When projectHash changes, reset selected item array.
  useEffect(
    () => setSelectedItems.reset(),
    [projectHash] // eslint-disable-line react-hooks/exhaustive-deps
  );

  return (
    <Tabs
      items={[
        {
          label: "Summary",
          key: "0",
          children: (
            <ActiveSummaryTab
              projectHash={projectHash}
              queryAPI={queryAPI}
              metricsSummary={metricsSummary}
              analysisDomain={analysisDomain}
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: "Explorer",
          key: "1",
          children: (
            <Explorer
              baseUrl="http://localhost:8502"
              projectHash={projectHash}
              filters={filters as any}
              queryAPI={queryAPI}
              scope={
                analysisDomain === "data" ? "data_quality" : "label_quality"
              }
            />
          ),
        },
      ]}
      activeKey={currentTab}
      onChange={setCurrentTab}
    />
  );
}

export default ActiveAnalysisDomainTab;
