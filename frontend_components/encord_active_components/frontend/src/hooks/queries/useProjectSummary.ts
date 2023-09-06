import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";

export function useProjectSummary(
  queryContext: QueryContext,
  projectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectSummary", queryContext.baseUrl, projectHash],
    () =>
      queryContext
        .getProjectV2API()
        .getProjectSummaryProjectsV2ProjectHashSummaryGet(projectHash)
        .then((r) => r.data),
    options
  );
}
