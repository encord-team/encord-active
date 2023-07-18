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
import { ActiveFilterState } from "../util/ActiveMetricFilter";

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
          label: "Search",
          key: "1",
          children: (
            <ActiveSearchTab
              projectHash={projectHash}
              editUrl={editUrl}
              queryAPI={queryAPI}
              analysisDomain={analysisDomain}
              metricsSummary={metricsSummary}
              itemWidth={200}
              filters={filters}
              setFilters={setFilters}
              selectedItems={selectedItems}
              setSelectedItems={setSelectedItems}
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: `${selectedItems.size} Tagged Items`,
          key: "2",
          children: (
            <ActiveSelectedTab
              projectHash={projectHash}
              editUrl={editUrl}
              itemWidth={200}
              selectedItems={selectedItems}
              setSelectedItems={setSelectedItems}
              queryAPI={queryAPI}
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
