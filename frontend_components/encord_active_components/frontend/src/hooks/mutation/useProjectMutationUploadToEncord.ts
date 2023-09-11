import { useMutation } from "@tanstack/react-query";
import { UploadProjectToEncordPostAction } from "../../openapi/api";
import { useQuerier } from "../Context";

export function useProjectMutationUploadToEncord(projectHash: string) {
  const querier = useQuerier();

  return useMutation(
    ["useProjectMutationCreateSubset", querier.baseUrl, projectHash],
    (uploadEncordAction: UploadProjectToEncordPostAction) =>
      querier
        .getProjectV2API()
        .routeActionUploadProjectToEncordApiProjectsV2ProjectHashActionsUploadToEncordPost(
          projectHash,
          uploadEncordAction
        )
  );
}
