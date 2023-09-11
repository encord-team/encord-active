import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import {
  CACHE_TIME_LIST_TOP_LEVEL,
  STALE_TIME_LIST_TOP_LEVEL,
} from "../queryConstants";

export function useProjectList(options: Pick<UseQueryOptions, "enabled"> = {}) {
  const querier = useQuerier();

  return useQuery(
    ["useProjectList", querier.baseUrl],
    () =>
      querier
        .getProjectV2API()
        .routeListProjectsApiProjectsV2Get()
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_LIST_TOP_LEVEL,
      cacheTime: CACHE_TIME_LIST_TOP_LEVEL,
    }
  );
}
