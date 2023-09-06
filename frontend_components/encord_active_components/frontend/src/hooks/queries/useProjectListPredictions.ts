import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";

export function useProjectListPredictions(
  queryContext: QueryContext,
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectListPredictions", queryContext.baseUrl, projectHash],
    () =>
      queryContext
        .getProjectV2API()
        .listProjectPredictionsProjectsV2ProjectHashPredictionsGet(projectHash)
        .then((r) => r.data),
    options
  );
}
