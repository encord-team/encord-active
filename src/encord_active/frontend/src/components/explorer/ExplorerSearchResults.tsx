import { List, PaginationProps } from "antd";

import { useMemo } from "react";
import { GalleryCard } from "../preview/GalleryCard";
import { loadingIndicator } from "../Spin";
import { FeatureHashMap } from "../Types";

const ExplorerSearchLocale = {
  emptyText: "No Results",
};

const ExplorerSearchPagination: PaginationProps = {
  defaultPageSize: 20,
  showTotal: (total) => `${total} Search Results`,
};

const ExplorerSearchPaginationTruncated: PaginationProps = {
  defaultPageSize: 20,
  showTotal: (total) => `${total}+ Search Results`,
};

const ExplorerSearchGrid = {};

export const ExplorerSearchResults = ExplorerSearchResultsRaw; // FIXME: react.memo

function ExplorerSearchResultsRaw(props: {
  projectHash: string;
  predictionHash: string | undefined;
  itemsToRender: readonly string[];
  itemSimilarities: readonly number[] | undefined;
  itemSimilarityItemAtIndex0: boolean;
  truncated: boolean;
  loadingDescription: string;
  selectedMetric: { domain: "annotation" | "data"; metric_key: string };
  toggleImageSelection: (itemId: string) => void;
  setPreviewedItem: (itemId: string) => void;
  setSimilaritySearch: (itemId: string | undefined) => void;
  selectedItems: ReadonlySet<string> | "ALL";
  showAnnotations: boolean;
  featureHashMap: FeatureHashMap;
  iou: number;
  setPage: (page: number) => void;
  page: number;
  setPageSize: (pageSize: number) => void;
  pageSize: number;
}) {
  const {
    projectHash,
    predictionHash,
    itemsToRender,
    itemSimilarities,
    itemSimilarityItemAtIndex0,
    loadingDescription,
    selectedMetric,
    toggleImageSelection,
    setPreviewedItem,
    setSimilaritySearch,
    selectedItems,
    showAnnotations,
    featureHashMap,
    iou,
    truncated,
    page,
    pageSize,
    setPage,
    setPageSize,
  } = props;

  const loading = useMemo(
    () => ({
      spinning: loadingDescription !== "",
      tip: loadingDescription,
      indicator: loadingIndicator,
    }),
    [loadingDescription]
  );

  // Cannot search by index. So we need to unify.
  const dataSource = useMemo((): {
    item: string;
    similarity?: number;
    similaritySearchCard?: true;
  }[] => {
    if (itemSimilarities == null) {
      return itemsToRender.map((item) => ({ item }));
    }
    return itemsToRender.map((item, index) =>
      index === 0 && itemSimilarityItemAtIndex0
        ? {
            item,
            similarity: itemSimilarities[index] ?? NaN,
            similaritySearchCard: true,
          }
        : { item, similarity: itemSimilarities[index] ?? NaN }
    );
  }, [itemsToRender, itemSimilarities, itemSimilarityItemAtIndex0]);

  return (
    <div className="relative h-full overflow-auto">
      <List
        className="absolute mt-2.5 px-2"
        dataSource={dataSource}
        grid={ExplorerSearchGrid}
        loading={loading}
        locale={ExplorerSearchLocale}
        pagination={{
          pageSize,
          current: page,
          onChange: (page, pageSize) => {
            setPage(page);
            setPageSize(pageSize);
          },
          ...(truncated
            ? ExplorerSearchPaginationTruncated
            : ExplorerSearchPagination),
        }}
        renderItem={({ item, similarity, similaritySearchCard }) => (
          <GalleryCard
            projectHash={projectHash}
            predictionHash={predictionHash}
            selectedMetric={selectedMetric}
            key={item}
            itemId={item}
            itemSimilarity={similarity}
            similaritySearchCard={similaritySearchCard ?? false}
            setSelectedToggle={toggleImageSelection}
            setItemPreview={setPreviewedItem}
            setSimilaritySearch={setSimilaritySearch}
            selected={selectedItems === "ALL" || selectedItems.has(item)}
            hideExtraAnnotations={showAnnotations}
            featureHashMap={featureHashMap}
            iou={iou}
          />
        )}
      />
      <div className="hidden h-full">1</div>
    </div>
  );
}
