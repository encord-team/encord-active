import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectAnalysisSearch(
  projectHash: string,
  domain: AnalysisDomain,
  orderBy: string,
  orderByDesc: boolean,
  offset: number,
  limit: number,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "useProjectAnalysisSearch",
      querier.baseUrl,
      projectHash,
      domain,
      orderBy,
      orderByDesc,
      offset,
      limit,
      filters,
    ],
    () =>
      querier
        .getProjectV2API()
        .routeProjectSearchApiProjectsV2ProjectHashAnalysisDomainSearchGet(
          projectHash,
          domain,
          orderBy,
          orderByDesc,
          offset,
          limit,
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
