import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";

export function useProjectMetadata(
  queryContext: QueryContext,
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectMetadata", queryContext.baseUrl, projectHash],
    () =>
      queryContext
        .getProjectV2API()
        .getProjectMetadataProjectsV2ProjectHashMetadataGet(projectHash)
        .then((r) => r.data),
    options
  );
}
