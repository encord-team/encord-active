import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import {
  CACHE_TIME_LIST_TOP_LEVEL,
  STALE_TIME_LIST_TOP_LEVEL,
} from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectListCollaborators(
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    ["useProjectListCollaborators", querier.baseUrl, projectHash],
    () =>
      querier
        .getProjectV2API()
        .routeProjectListCollaboratorsApiProjectsV2ProjectHashCollaboratorsGet(
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
