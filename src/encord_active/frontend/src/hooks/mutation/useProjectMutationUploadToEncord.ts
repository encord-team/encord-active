import { useMutation, useQueryClient } from "@tanstack/react-query";
import { UploadProjectToEncordPostAction } from "../../openapi/api";
import { useQuerier } from "../Context";

export function useProjectMutationUploadToEncord(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationCreateSubset", querier.baseUrl, projectHash],
    (uploadEncordAction: UploadProjectToEncordPostAction) =>
      querier
        .getProjectV2API()
        .routeActionUploadProjectToEncordApiProjectsV2ProjectHashActionsUploadToEncordPost(
          projectHash,
          uploadEncordAction
        )
        .then(async (r) => {
          await queryClient.invalidateQueries([
            "useProjectList",
            querier.baseUrl,
          ]);
          return r;
        })
  );
}
