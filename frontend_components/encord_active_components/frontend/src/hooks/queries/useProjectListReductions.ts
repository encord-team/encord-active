import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";

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
    options
  );
}
