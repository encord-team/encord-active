import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  AnalysisDomain,
  AnalysisBuckets,
  SearchFilters,
} from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectAnalysisDistribution(
  projectHash: string,
  domain: AnalysisDomain,
  group: string,
  buckets: AnalysisBuckets | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "useProjectAnalysisDistribution",
      querier.baseUrl,
      projectHash,
      domain,
      group,
      buckets,
      filters,
    ],
    () =>
      querier
        .getProjectAPI()
        .routeProjectAnalysisDistribution(
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
