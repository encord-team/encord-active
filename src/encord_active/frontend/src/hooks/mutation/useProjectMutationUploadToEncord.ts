import { useMutation, useQueryClient } from "@tanstack/react-query";
import { UploadProjectToEncordPostAction } from "../../openapi/api";
import { useQuerier } from "../Context";

export function useProjectMutationUploadToEncord(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationUploadToEncord", querier.baseUrl, projectHash],
    (uploadEncordAction: UploadProjectToEncordPostAction) =>
      querier
        .getProjectAPI()
        .routeActionUploadProjectToEncord(projectHash, uploadEncordAction)
        .then(async (r) => {
          await queryClient.invalidateQueries([
            "useProjectList",
            querier.baseUrl,
          ]);
          return r;
        })
  );
}
