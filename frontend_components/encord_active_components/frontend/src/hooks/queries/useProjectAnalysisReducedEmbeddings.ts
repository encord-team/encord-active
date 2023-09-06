import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  AnalysisDomain,
  Get2dEmbeddingSummaryProjectsV2ProjectHashAnalysisDomainReductionsReductionHashSummaryGetBucketsEnum,
  SearchFilters,
} from "../../openapi/api";

export function useProjectAnalysisReducedEmbeddings(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  reductionHash: string,
  buckets: 10 | 100 | 1000 | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisReducedEmbeddings",
      queryContext.baseUrl,
      projectHash,
      domain,
      reductionHash,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .get2dEmbeddingSummaryProjectsV2ProjectHashAnalysisDomainReductionsReductionHashSummaryGet(
          projectHash,
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
