import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  AnalysisBuckets,
  PredictionDomain,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";

export function usePredictionAnalysisReducedEmbeddings(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  domain: PredictionDomain,
  iou: number,
  reductionHash: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "usePredictionAnalysisReducedEmbeddings",
      queryContext.baseUrl,
      projectHash,
      predictionHash,
      domain,
      reductionHash,
      iou,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .routePredictionReductionScatterProjectsV2ProjectHashPredictionsPredictionHashAnalyticsDomainReductionsReductionHashSummaryGet(
          projectHash,
          predictionHash,
          domain,
          reductionHash,
          iou,
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
