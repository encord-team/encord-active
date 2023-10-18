import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { ProjectTagRequest } from "../../openapi/api";

export function useProjectMutationUpdateTag(
  projectHash: string,
  tag_hash: string
) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationUpdateTag", querier.baseUrl, projectHash],
    (tag: ProjectTagRequest) =>
      querier
        .getProjectAPI()
        .routeUpdateTag(projectHash, tag_hash, tag)
        .then(async (r) => {
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
          return r.data;
        })
  );
}
