import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  AnalysisDomain,
  AnalysisBuckets,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";

export function useProjectAnalysisDistribution(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  group: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisDistribution",
      queryContext.baseUrl,
      projectHash,
      domain,
      group,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .routeProjectDistributionProjectsV2ProjectHashAnalysisDomainDistributionGet(
          projectHash,
          domain,
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
