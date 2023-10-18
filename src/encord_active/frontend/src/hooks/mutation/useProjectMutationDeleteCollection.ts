import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";

export function useProjectMutationDeleteCollection(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationDeleteCollection", querier.baseUrl, projectHash],
    (tagHashList: string[]) =>
      querier
        .getProjectAPI()
        .routeDeleteTag(projectHash, tagHashList, {})
        .then(async () => {
          await queryClient.invalidateQueries([
            "useProjectItemsListTags",
            querier.baseUrl,
            projectHash,
          ]);
          await queryClient.invalidateQueries([
            "useProjectListTags",
            querier.baseUrl,
            projectHash,
          ]);
          await queryClient.invalidateQueries([
            "useProjectListTagsMeta",
            querier.baseUrl,
            projectHash,
          ]);
          await queryClient.invalidateQueries([
            "useProjectFilterListTags",
            querier.baseUrl,
            projectHash,
          ]);
          await queryClient.invalidateQueries([
            "useProjectItem",
            querier.baseUrl,
            projectHash,
          ]);
        })
  );
}
