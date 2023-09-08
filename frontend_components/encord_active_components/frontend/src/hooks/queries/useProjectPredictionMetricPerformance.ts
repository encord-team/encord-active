import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { AnalysisBuckets, SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectPredictionMetricPerformance(
  projectHash: string,
  predictionHash: string,
  iou: number,
  metricName: string,
  buckets: AnalysisBuckets | undefined = undefined,
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
      metricName,
      buckets,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routePredictionMetricPerformanceApiProjectsV2ProjectHashPredictionsPredictionHashMetricPerformanceGet(
          projectHash,
          predictionHash,
          iou,
          metricName,
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
