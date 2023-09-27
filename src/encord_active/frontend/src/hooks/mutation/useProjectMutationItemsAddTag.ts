import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { toDataItemID } from "../../components/util/ItemIdUtil";

export function useProjectMutationItemsAddTag(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationItemsAddTag", querier.baseUrl, projectHash],
    async (tag: { items: Array<string>; tags: Array<string> }) => {
      await querier
        .getProjectAPI()
        .routeItemsTagAllApiProjectsV2ProjectHashTagsItemsTagAllPost(
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
