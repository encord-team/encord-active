import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  AnalysisBuckets,
  AnalysisDomain,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";

export function useProjectAnalysisReducedEmbeddings(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  reductionHash: string,
  buckets: AnalysisBuckets | undefined = undefined,
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
        .routeProjectReductionScatterProjectsV2ProjectHashAnalysisDomainReductionsReductionHashSummaryGet(
          projectHash,
          domain,
          reductionHash,
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
