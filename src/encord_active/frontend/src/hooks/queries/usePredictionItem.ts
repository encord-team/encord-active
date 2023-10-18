import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { CACHE_TIME_ITEM, STALE_TIME_ITEM } from "../queryConstants";
import { useQuerier } from "../Context";

export function usePredictionItem(
  projectHash: string,
  predictionHash: string,
  dataItem: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "usePredictionItem",
      querier.baseUrl,
      projectHash,
      predictionHash,
      dataItem,
    ],
    () =>
      querier
        .getPredictionAPI()
        .routePredictionDataItem(projectHash, predictionHash, dataItem)
        .then((r) => r.data),
    { ...options, staleTime: STALE_TIME_ITEM, cacheTime: CACHE_TIME_ITEM }
  );
}
