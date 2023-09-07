import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  Get2dEmbeddingSummaryProjectsV2ProjectHashAnalysisDomainReductionsReductionHashSummaryGetBucketsEnum,
  PredictionDomain,
  SearchFilters,
} from "../../openapi/api";

export function usePredictionAnalysisReducedEmbeddings(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  domain: PredictionDomain,
  reductionHash: string,
  buckets: 10 | 100 | 1000 | undefined = undefined,
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
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .get2dEmbeddingSummaryPredictionProjectsV2ProjectHashPredictionsPredictionHashAnalyticsDomainReductionsReductionHashSummaryGet(
          projectHash,
          predictionHash,
          domain,
          reductionHash,
          buckets === undefined
            ? undefined
            : (String(
                buckets
              ) as Get2dEmbeddingSummaryProjectsV2ProjectHashAnalysisDomainReductionsReductionHashSummaryGetBucketsEnum),
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    options
  );
}
