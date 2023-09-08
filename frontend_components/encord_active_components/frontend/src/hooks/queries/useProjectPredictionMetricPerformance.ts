import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { AnalysisBuckets, SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";

export function useProjectPredictionMetricPerformance(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  iou: number,
  metricName: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectPredictionSummary",
      queryContext.baseUrl,
      projectHash,
      predictionHash,
      iou,
      metricName,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .routePredictionMetricPerformanceProjectsV2ProjectHashPredictionsPredictionHashMetricPerformanceGet(
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
