import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectPredictionSummary(
  projectHash: string,
  predictionHash: string,
  iou: number,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier()
  return useQuery(
    [
      "useProjectPredictionSummary",
      querier.baseUrl,
      projectHash,
      predictionHash,
      iou,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routePredictionSummaryProjectsV2ProjectHashPredictionsPredictionHashSummaryGet(
          predictionHash,
          projectHash,
          iou,
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_ANALYTICS,
      cacheTime: CACHE_TIME_ANALYTICS,
    }
  );
}
