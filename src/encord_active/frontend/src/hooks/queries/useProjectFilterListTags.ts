import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  CACHE_TIME_LIST_TOP_LEVEL,
  STALE_TIME_LIST_TOP_LEVEL,
} from "../queryConstants";
import { useQuerier } from "../Context";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";

export function useProjectFilterListTags(
  projectHash: string,
  domain: AnalysisDomain,
  filters: SearchFilters,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    ["useProjectFilterListTags", querier.baseUrl, projectHash, domain, filters],
    () =>
      querier
        .getProjectAPI()
        .routeFilterAllTagsApiProjectsV2ProjectHashTagsDomainFilterAllTagsPost(
          projectHash,
          domain,
          filters
        )
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_LIST_TOP_LEVEL,
      cacheTime: CACHE_TIME_LIST_TOP_LEVEL,
    }
  );
}
