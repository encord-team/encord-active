import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  AnalysisBuckets,
  AnalysisDomain,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectAnalysisReducedEmbeddings(
  projectHash: string,
  domain: AnalysisDomain,
  reductionHash: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "useProjectAnalysisReducedEmbeddings",
      querier.baseUrl,
      projectHash,
      domain,
      reductionHash,
      buckets,
      filters,
    ],
    () =>
      querier
        .getProjectAPI()
        .routeProjectReductionScatterApiProjectsV2ProjectHashAnalysisDomainReductionsReductionHashSummaryGet(
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
