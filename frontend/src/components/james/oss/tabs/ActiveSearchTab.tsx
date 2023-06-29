import * as React from "react";
import { useCallback, useMemo, useState } from "react";
import { Button, Pagination, Popover, Space, Spin } from "antd";
import { Actions } from "usehooks-ts/dist/esm/useMap/useMap";
import { useDebounce } from "usehooks-ts";
import { FilterOutlined, TagOutlined, TagsOutlined } from "@ant-design/icons";
import ActiveViewImageCard from "../view/ActiveViewImageCard";
import {
  ActiveProjectAnalysisDomain,
  ActiveProjectMetricSummary,
  ActiveProjectSearchResult,
  ActiveQueryAPI,
} from "../ActiveTypes";
import ActiveMetricFilter, {
  ActiveFilterState,
} from "../util/ActiveMetricFilter";

function hasTaggedRange(
  selectedItems: Omit<Map<string, null>, "set" | "clear" | "delete">,
  newResults: ActiveProjectSearchResult["results"] | undefined
): boolean {
  if (newResults == null || newResults.length === 0) {
    return false;
  }
  return (
    newResults.find(
      (resultItem) =>
        !selectedItems.has(
          `${resultItem.du_hash}/${resultItem.frame}/${
            resultItem.object_hash ?? ""
          }`
        )
    ) == null
  );
}

export function ActiveSearchTab(props: {
  projectHash: string;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  queryAPI: ActiveQueryAPI;
  analysisDomain: ActiveProjectAnalysisDomain;
  metricsSummary: ActiveProjectMetricSummary;
  itemWidth: number;
  filters: ActiveFilterState;
  setFilters: (
    newState:
      | ActiveFilterState
      | ((old: ActiveFilterState) => ActiveFilterState)
  ) => void;
  selectedItems: Omit<Map<string, null>, "set" | "clear" | "delete">;
  setSelectedItems: Actions<string, null>;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const {
    editUrl,
    projectHash,
    queryAPI,
    analysisDomain,
    metricsSummary,
    filters,
    setFilters,
    selectedItems,
    setSelectedItems,
    itemWidth,
    featureHashMap,
  } = props;
  const [pageSize, setPageSize] = useState<number>(20);
  const [pageIdx, setPageIdx] = useState<number>(1);

  const summary = queryAPI.useProjectAnalysisSummary(
    projectHash,
    analysisDomain
  );
  const metricRanges = summary.data?.metrics;

  const debouncedFilters = useDebounce(filters, 500);
  const searchQueryResult = queryAPI.useProjectAnalysisSearch(
    projectHash,
    analysisDomain,
    debouncedFilters.metricFilters,
    null,
    debouncedFilters.enumFilters,
    null,
    false
  );
  const searchResults = searchQueryResult.data?.results;
  const searchTruncated = searchQueryResult.data?.truncated ?? false;

  // Post-processing hack for standardised API
  const explorerItems = useMemo(() => {
    if (searchResults == null) {
      return null;
    }
    return {
      total: searchResults.length,
      results: searchResults.slice(
        (pageIdx - 1) * pageSize,
        Math.min(searchResults.length, pageIdx * pageSize)
      ),
    };
  }, [searchResults, pageIdx, pageSize]);

  const setSelectedItemBool = useCallback(
    (selectedKey: string, selected: boolean) => {
      if (selected) {
        setSelectedItems.set(selectedKey, null);
      } else {
        setSelectedItems.remove(selectedKey);
      }
    },
    [setSelectedItems]
  );
  const hasSelectedItemBool = useCallback(
    (selectedKey: string) => selectedItems.has(selectedKey),
    [selectedItems]
  );

  const hasTaggedSearchResults = useMemo(
    () => hasTaggedRange(selectedItems, searchResults),
    [searchResults, selectedItems]
  );

  const hasTaggedExplorerItems = useMemo(
    () => hasTaggedRange(selectedItems, explorerItems?.results),
    [explorerItems?.results, selectedItems]
  );

  const tagRange = (
    newResults: ActiveProjectSearchResult["results"] | undefined | null,
    set: boolean
  ) => {
    if (newResults == null || newResults.length === 0) {
      return;
    }
    const currentMap = new Map(selectedItems);
    newResults.forEach((resultItem) => {
      const key = `${resultItem.du_hash}/${resultItem.frame}/${
        resultItem.object_hash ?? ""
      }`;
      if (set) {
        currentMap.set(key, null);
      } else {
        currentMap.delete(key);
      }
    });
    setSelectedItems.setAll(currentMap);
  };

  return (
    <>
      <Space.Compact block style={{ marginBottom: 15 }}>
        <Popover
          placement="bottomLeft"
          content={
            <ActiveMetricFilter
              filters={filters}
              setFilters={setFilters}
              metricsSummary={metricsSummary}
              metricRanges={metricRanges}
              featureHashMap={featureHashMap}
            />
          }
          trigger="click"
        >
          <Button type="primary" icon={<FilterOutlined />}>
            Filters
          </Button>
        </Popover>
        <Button
          onClick={() =>
            tagRange(explorerItems?.results, !hasTaggedExplorerItems)
          }
          type="primary"
          icon={<TagOutlined />}
        >
          {hasTaggedExplorerItems ? "Untag" : "Tag"} current page
        </Button>
        <Button
          onClick={() => tagRange(searchResults, !hasTaggedSearchResults)}
          type="primary"
          icon={<TagsOutlined />}
        >
          {hasTaggedSearchResults ? "Untag" : "Tag"} all pages
          {searchTruncated ? " (Truncated)" : ""}
        </Button>
      </Space.Compact>
      <Space wrap>
        {explorerItems == null ? (
          <Spin />
        ) : (
          explorerItems.results.map((exploreItem) => {
            const key = `${exploreItem.du_hash}/${exploreItem.frame}/${
              exploreItem.object_hash ?? ""
            }`;
            return (
              <ActiveViewImageCard
                key={key}
                queryAPI={queryAPI}
                projectHash={projectHash}
                duHash={exploreItem.du_hash}
                frame={exploreItem.frame}
                objectHash={exploreItem.object_hash}
                width={itemWidth}
                editUrl={editUrl}
                selectedKey={key}
                getSelected={hasSelectedItemBool}
                setSelected={setSelectedItemBool}
              />
            );
          })
        )}
      </Space>
      <Pagination
        style={{ marginTop: 10 }}
        total={explorerItems?.total ?? 1}
        showSizeChanger
        showQuickJumper
        pageSize={pageSize}
        current={pageIdx}
        onChange={(page, pageSize) => {
          setPageIdx(page);
          setPageSize(pageSize);
        }}
        showTotal={(total, range) =>
          `${total} results ${searchTruncated ? " (Truncated)" : ""}`
        }
      />
    </>
  );
}

export default ActiveSearchTab;
