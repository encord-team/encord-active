import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";

export function useProjectSummary(
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    ["useProjectSummary", querier.baseUrl, projectHash],
    () =>
      querier
        .getProjectAPI()
        .routeProjectSummary(projectHash)
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_ANALYTICS,
      cacheTime: CACHE_TIME_ANALYTICS,
    }
  );
}
