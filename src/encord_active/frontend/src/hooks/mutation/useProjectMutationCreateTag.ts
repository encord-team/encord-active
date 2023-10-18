import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";

export function useProjectMutationCreateTag(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationCreateTag", querier.baseUrl, projectHash],
    (tagNames: string[]) =>
      querier
        .getProjectAPI()
        .routeCreateTags(projectHash, tagNames)
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
