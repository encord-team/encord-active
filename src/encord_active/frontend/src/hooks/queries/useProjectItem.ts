import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { CACHE_TIME_ITEM, STALE_TIME_ITEM } from "../queryConstants";

export function useProjectItem(
  projectHash: string,
  dataItem: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    ["useProjectItem", querier.baseUrl, projectHash, dataItem],
    () =>
      querier
        .getProjectAPI()
        .routeProjectDataItem(projectHash, dataItem)
        .then((r) => r.data),
    { ...options, staleTime: STALE_TIME_ITEM, cacheTime: CACHE_TIME_ITEM }
  );
}
