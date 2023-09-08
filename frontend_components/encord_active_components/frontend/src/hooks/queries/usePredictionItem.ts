import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { CACHE_TIME_ITEM, STALE_TIME_ITEM } from "../queryConstants";

export function usePredictionItem(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  dataItem: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "usePredictionItem",
      queryContext.baseUrl,
      projectHash,
      predictionHash,
      dataItem,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .routePredictionDataItemProjectsV2ProjectHashPredictionsPredictionHashPreviewDataItemGet(
          projectHash,
          predictionHash,
          dataItem
        )
        .then((r) => r.data),
    { ...options, staleTime: STALE_TIME_ITEM, cacheTime: CACHE_TIME_ITEM }
  );
}
