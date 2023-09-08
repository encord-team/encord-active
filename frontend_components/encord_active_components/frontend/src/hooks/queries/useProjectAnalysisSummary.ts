import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectAnalysisSummary(
  projectHash: string,
  domain: AnalysisDomain,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier()
  return useQuery(
    [
      "useProjectAnalysisSummary",
      querier.baseUrl,
      projectHash,
      domain,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routeProjectSummaryApiProjectsV2ProjectHashAnalysisDomainSummaryGet(
          projectHash,
          domain,
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
