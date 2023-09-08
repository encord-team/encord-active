import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { PredictionDomain, SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function usePredictionAnalysisSearch(
  projectHash: string,
  predictionHash: string,
  domain: PredictionDomain,
  iou: number,
  orderBy: string,
  orderByDesc: boolean,
  offset: number,
  limit: number,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier()
  return useQuery(
    [
      "usePredictionAnalysisSearch",
      querier.baseUrl,
      projectHash,
      predictionHash,
      domain,
      iou,
      orderBy,
      orderByDesc,
      offset,
      limit,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routePredictionSearchProjectsV2ProjectHashPredictionsPredictionHashAnalyticsDomainSearchGet(
          projectHash,
          predictionHash,
          domain,
          iou,
          orderBy,
          orderByDesc,
          offset,
          limit,
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
