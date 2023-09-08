import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  AnalysisDomain,
  AnalysisBuckets,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";

export function useProjectAnalysisMetricScatter(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  xMetric: string,
  yMetric: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisMetricScatter",
      queryContext.baseUrl,
      projectHash,
      domain,
      xMetric,
      yMetric,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .routeProjectScatterProjectsV2ProjectHashAnalysisDomainScatterGet(
          projectHash,
          domain,
          xMetric,
          yMetric,
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
