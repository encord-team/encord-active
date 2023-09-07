import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { CACHE_TIME_ITEM, STALE_TIME_ITEM } from "../queryConstants";

export function useProjectItem(
  queryContext: QueryContext,
  projectHash: string,
  dataItem: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectItem", queryContext.baseUrl, projectHash, dataItem],
    () =>
      queryContext
        .getProjectV2API()
        .projectItemProjectsV2ProjectHashItemDataItemGet(projectHash, dataItem)
        .then((r) => r.data),
    { ...options, staleTime: STALE_TIME_ITEM, cacheTime: CACHE_TIME_ITEM }
  );
}
