import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectAnalysisSearch(
  projectHash: string,
  domain: AnalysisDomain,
  orderBy: string | undefined,
  orderByDesc: boolean,
  offset: number,
  limit: number,
  filters: SearchFilters | undefined = undefined,
  similarityItem: string | undefined = undefined,
  similarityText: string | undefined = undefined,
  similarityImage: File | undefined = undefined,
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
      similarityItem,
      similarityText,
      similarityImage,
    ],
    () =>
      querier
        .getProjectV2API()
        .routeProjectSearchApiProjectsV2ProjectHashAnalysisDomainSearchPost(
          projectHash,
          domain,
          filters !== undefined ? JSON.stringify(filters) : undefined,
          orderBy,
          orderByDesc,
          offset,
          limit,
          similarityText,
          similarityImage,
          similarityItem
        )
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_ANALYTICS,
      cacheTime: CACHE_TIME_ANALYTICS,
    }
  );
}
