import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";

export function useProjectList(
  queryContext: QueryContext,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectList", queryContext.baseUrl],
    () =>
      queryContext
        .getProjectV2API()
        .getAllProjectsProjectsV2Get()
        .then((r) => r.data),
    options
  );
}
