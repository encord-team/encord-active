import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";

export function useProjectDataItem(
  queryContext: QueryContext,
  projectHash: string,
  dataItem: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    ["useProjectItem", queryContext.baseUrl, projectHash, dataItem],
    () =>
      queryContext
        .getProjectV2API()
        .projectItemProjectsV2ProjectHashItemDataItemGet(projectHash, dataItem)
        .then((r) => r.data),
    options
  );
}
