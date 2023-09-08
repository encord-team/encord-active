import { useMutation } from "@tanstack/react-query";
import { QueryContext } from "../Context";
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
        .routeActionCreateProjectSubsetProjectsV2ProjectHashActionsCreateProjectSubsetPost(
          projectHash,
          createSubsetAction
        )
  );
}
