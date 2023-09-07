import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  CACHE_TIME_LIST_TOP_LEVEL,
  STALE_TIME_LIST_TOP_LEVEL,
} from "../queryConstants";

export function useProjectList(
  queryContext: QueryContext,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectList", queryContext.baseUrl],
    () =>
      queryContext
        .getProjectV2API()
        .getAllProjectsProjectsV2Get()
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_LIST_TOP_LEVEL,
      cacheTime: CACHE_TIME_LIST_TOP_LEVEL,
    }
  );
}
