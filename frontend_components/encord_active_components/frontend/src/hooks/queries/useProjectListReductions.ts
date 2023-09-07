import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  CACHE_TIME_LIST_TOP_LEVEL,
  STALE_TIME_LIST_TOP_LEVEL,
} from "../queryConstants";

export function useProjectListReductions(
  queryContext: QueryContext,
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectListReductions", queryContext.baseUrl, projectHash],
    () =>
      queryContext
        .getProjectV2API()
        .listSupported2dEmbeddingReductionsProjectsV2ProjectHashReductionsGet(
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
