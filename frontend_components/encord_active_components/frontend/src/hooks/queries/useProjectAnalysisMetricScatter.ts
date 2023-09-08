import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  AnalysisDomain,
  AnalysisBuckets,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectAnalysisMetricScatter(
  projectHash: string,
  domain: AnalysisDomain,
  xMetric: string,
  yMetric: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier()
  return useQuery(
    [
      "useProjectAnalysisMetricScatter",
      querier.baseUrl,
      projectHash,
      domain,
      xMetric,
      yMetric,
      buckets,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routeProjectScatterApiProjectsV2ProjectHashAnalysisDomainScatterGet(
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
