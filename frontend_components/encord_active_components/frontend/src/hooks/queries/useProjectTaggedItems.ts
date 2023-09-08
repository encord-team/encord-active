import { useQuery } from "@tanstack/react-query";
import { useQuerier } from "../Context";

export function useProjectTaggedItems(
  projectHash: string,
) {

  const querier = useQuerier()
  return useQuery(
    ["useProjectItem", querier.baseUrl, projectHash],
    async () =>
      await querier
        .getProjectV2API()
        .routeTaggedItemsProjectsV2ProjectHashTagsTaggedItemsGet(
          projectHash,
        )
  );
}

