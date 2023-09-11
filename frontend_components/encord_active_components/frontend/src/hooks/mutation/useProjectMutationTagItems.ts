import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { ProjectsV2Api } from "../../openapi/api";

export function useProjectMutationTagItems(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationUpdateItemTags", querier.baseUrl, projectHash],
    (
      itemTags: Parameters<
        ProjectsV2Api["routeTagItemsApiProjectsV2ProjectHashTagsTagItemsPut"]
      >[1]
    ) =>
      querier
        .getProjectV2API()
        .routeTagItemsApiProjectsV2ProjectHashTagsTagItemsPut(
          projectHash,
          itemTags
        ),
    {
      onSettled: () => {
        queryClient.invalidateQueries({ queryKey: [projectHash, "item"] });
        queryClient.invalidateQueries({
          queryKey: [projectHash, "tagged_items"],
        });
      },
    }
  );
}
