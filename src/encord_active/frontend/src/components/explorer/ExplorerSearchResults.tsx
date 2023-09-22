import { List, PaginationProps } from "antd";

import { useMemo } from "react";
import { GalleryCard } from "../preview/GalleryCard";
import { loadingIndicator } from "../Spin";
import { FeatureHashMap } from "../Types";

const ExplorerSearchLocale = {
  emptyText: "No Results",
};

const ExplorerSearchPagination: PaginationProps = {
  defaultPageSize: 30,
};

const ExplorerSearchGrid = {};

export const ExplorerSearchResults = ExplorerSearchResultsRaw; // FIXME: react.memo

function ExplorerSearchResultsRaw(props: {
  projectHash: string;
  predictionHash: string | undefined;
  itemsToRender: readonly string[];
  itemSimilarities: readonly number[] | undefined;
  loadingDescription: string;
  selectedMetric: { domain: "annotation" | "data"; metric_key: string };
  toggleImageSelection: (itemId: string) => void;
  setPreviewedItem: (itemId: string) => void;
  showSimilarItems: (itemId: string) => void;
  selectedItems: ReadonlySet<string>;
  showAnnotations: boolean;
  featureHashMap: FeatureHashMap;
  iou: number;
}) {
  const {
    projectHash,
    predictionHash,
    itemsToRender,
    itemSimilarities,
    loadingDescription,
    selectedMetric,
    toggleImageSelection,
    setPreviewedItem,
    showSimilarItems,
    selectedItems,
    showAnnotations,
    featureHashMap,
    iou,
  } = props;

  const loading = useMemo(
    () => ({
      spinning: loadingDescription !== "",
      tip: loadingDescription,
      indicator: loadingIndicator,
    }),
    [loadingDescription]
  );

  return (
    <List
      className="mt-2.5"
      dataSource={itemsToRender as string[]}
      grid={ExplorerSearchGrid}
      loading={loading}
      locale={ExplorerSearchLocale}
      pagination={ExplorerSearchPagination}
      renderItem={(item: string, index: number) => (
        <GalleryCard
          projectHash={projectHash}
          predictionHash={predictionHash}
          selectedMetric={selectedMetric}
          key={item}
          itemId={item}
          itemSimilarity={
            itemSimilarities == null || index > itemSimilarities.length
              ? undefined
              : itemSimilarities[index]
          }
          onClick={toggleImageSelection}
          onExpand={setPreviewedItem}
          onShowSimilar={showSimilarItems}
          selected={selectedItems.has(item)}
          hideExtraAnnotations={showAnnotations}
          featureHashMap={featureHashMap}
          iou={iou}
        />
      )}
    />
  );
}
