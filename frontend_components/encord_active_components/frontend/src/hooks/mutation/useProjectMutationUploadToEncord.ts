import { useMutation } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { UploadProjectToEncordPostAction } from "../../openapi/api";

export function useProjectMutationUploadToEncord(
  queryContext: QueryContext,
  projectHash: string
) {
  return useMutation(
    ["useProjectMutationCreateSubset", queryContext.baseUrl, projectHash],
    (uploadEncordAction: UploadProjectToEncordPostAction) =>
      queryContext
        .getProjectV2API()
        .uploadProjectToEncordProjectsV2ProjectHashActionsUploadToEncordPost(
          projectHash,
          uploadEncordAction
        )
  );
}
