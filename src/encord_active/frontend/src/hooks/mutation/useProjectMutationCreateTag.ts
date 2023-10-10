import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { ProjectTagRequest } from "../../openapi/api";

export function useProjectMutationCreateTag(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationCreateTag", querier.baseUrl, projectHash],
    (tags: ProjectTagRequest[]) =>
      querier
        .getProjectAPI()
        .routeCreateTagsApiProjectsV2ProjectHashTagsPost(projectHash, tags)
        .then(async (r) => {
          await queryClient.invalidateQueries([
            "useProjectListTags",
            querier.baseUrl,
            projectHash,
          ]);
          return r.data;
        })
  );
}
