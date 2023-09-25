import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { CreateProjectSubsetPostAction } from "../../openapi/api";

export function useProjectMutationCreateSubset(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationCreateSubset", querier.baseUrl, projectHash],
    (createSubsetAction: CreateProjectSubsetPostAction) =>
      querier
        .getProjectAPI()
        .routeActionCreateProjectSubsetApiProjectsV2ProjectHashActionsCreateProjectSubsetPost(
          projectHash,
          createSubsetAction
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
