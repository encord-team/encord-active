import { useMutation } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { CreateProjectSubsetPostAction } from "../../openapi/api";

export function useProjectMutationCreateSubset(
  projectHash: string
) {
  const querier = useQuerier()

  return useMutation(
    ["useProjectMutationCreateSubset", querier.baseUrl, projectHash],
    (createSubsetAction: CreateProjectSubsetPostAction) =>
      querier
        .getProjectV2API()
        .routeActionCreateProjectSubsetApiProjectsV2ProjectHashActionsCreateProjectSubsetPost(
          projectHash,
          createSubsetAction
        )
  );
}
