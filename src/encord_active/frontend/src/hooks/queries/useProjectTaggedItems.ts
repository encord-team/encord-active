import { useQuery } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { GroupedTags } from "../../openapi/api";

export function useProjectTaggedItems(projectHash: string) {
  const querier = useQuerier();

  return useQuery(
    ["useTaggedItems", querier.baseUrl, projectHash],
    async () => {
      const { data } = await querier
        .getProjectAPI()
        .routeTaggedItemsApiProjectsV2ProjectHashTagsTaggedItemsGet(
          projectHash
        );

      return new Map(
        Object.entries(data).filter(([_, tags]) => tags != null) as [
          string,
          GroupedTags
        ][]
      );
    }
  );
}
