import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectMetadata(
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier()
  return useQuery(
    ["useProjectMetadata", querier.baseUrl, projectHash],
    () =>
      querier
        .getProjectV2API()
        .routeProjectMetadataProjectsV2ProjectHashMetadataGet(projectHash)
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_ANALYTICS,
      cacheTime: CACHE_TIME_ANALYTICS,
    }
  );
}
