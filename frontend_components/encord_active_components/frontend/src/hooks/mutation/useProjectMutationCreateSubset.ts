import { QueryContext } from "../Context";
import { useMutation } from "@tanstack/react-query";
import { CreateProjectSubsetPostAction } from "../../openapi/api";

export function useProjectMutationCreateSubset(
  queryContext: QueryContext,
  projectHash: string
) {
  return useMutation(
    ["useProjectMutationCreateSubset", queryContext.baseUrl, projectHash],
    (createSubsetAction: CreateProjectSubsetPostAction) =>
      queryContext
        .getProjectV2API()
        .createActiveSubsetProjectsV2ProjectHashActionsCreateProjectSubsetPost(
          projectHash,
          createSubsetAction
        )
  );
}
