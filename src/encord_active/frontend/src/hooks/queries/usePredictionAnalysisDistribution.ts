import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  AnalysisBuckets,
  PredictionDomain,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function usePredictionAnalysisDistribution(
  projectHash: string,
  predictionHash: string,
  domain: PredictionDomain,
  iou: number,
  group: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "usePredictionAnalysisDistribution",
      querier.baseUrl,
      projectHash,
      predictionHash,
      domain,
      iou,
      group,
      buckets,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routePredictionDistributionApiProjectsV2ProjectHashPredictionsPredictionHashAnalyticsDomainDistributionGet(
          projectHash,
          predictionHash,
          domain,
          iou,
          group,
          buckets,
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
