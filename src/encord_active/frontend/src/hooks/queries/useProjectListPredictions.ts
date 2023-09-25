import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  CACHE_TIME_LIST_TOP_LEVEL,
  STALE_TIME_LIST_TOP_LEVEL,
} from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectListPredictions(
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    ["useProjectListPredictions", querier.baseUrl, projectHash],
    () =>
      querier
        .getProjectAPI()
        .routeProjectListPredictionsApiProjectsV2ProjectHashPredictionsGet(
          projectHash
        )
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_LIST_TOP_LEVEL,
      cacheTime: CACHE_TIME_LIST_TOP_LEVEL,
    }
  );
}
