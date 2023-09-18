import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  AnalysisBuckets,
  PredictionDomain,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function usePredictionAnalysisReducedEmbeddings(
  projectHash: string,
  predictionHash: string,
  domain: PredictionDomain,
  iou: number,
  reductionHash: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "usePredictionAnalysisReducedEmbeddings",
      querier.baseUrl,
      projectHash,
      predictionHash,
      domain,
      reductionHash,
      iou,
      buckets,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routePredictionReductionScatterApiProjectsV2ProjectHashPredictionsPredictionHashAnalyticsDomainReductionsReductionHashSummaryGet(
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
