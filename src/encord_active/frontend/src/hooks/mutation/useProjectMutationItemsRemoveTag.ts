import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { toDataItemID } from "../../components/util/ItemIdUtil";

export function useProjectMutationItemsRemoveTag(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationItemsRemoveTag", querier.baseUrl, projectHash],
    async (tag: { items: Array<string>; tags: Array<string> }) => {
      await querier
        .getProjectAPI()
        .routeItemsUntagAllApiProjectsV2ProjectHashTagsItemsUntagAllPost(
          projectHash,
          tag
        );
      const invalidateSet = new Set(tag.items.map(toDataItemID));
      await Promise.all(
        [...invalidateSet].map((dataItem) =>
          queryClient.invalidateQueries({
            queryKey: [
              "useProjectItem",
              querier.baseUrl,
              projectHash,
              dataItem,
            ],
          })
        )
      );
      await queryClient.invalidateQueries({
        queryKey: ["useProjectItemsListTags", querier.baseUrl, projectHash],
      });
    }
  );
}