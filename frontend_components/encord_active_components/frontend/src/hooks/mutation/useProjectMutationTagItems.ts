import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { ProjectsV2Api } from "../../openapi/api";
import { useProjectTaggedItems } from "../queries/useProjectTaggedItems";

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
      onMutate: (variables) => {
        const key = ["useTaggedItems", querier.baseUrl, projectHash];
        const prevData = queryClient.getQueryData(key) as NonNullable<
          ReturnType<typeof useProjectTaggedItems>["data"]
        >;
        queryClient.setQueryData(key, () => {
          variables.forEach(({ id, grouped_tags }) => {
            prevData.set(id, grouped_tags);
          });
          return prevData;
        });
      },
      onSettled: () => {
        queryClient.invalidateQueries({
          queryKey: ["useProjectItem", querier.baseUrl, projectHash],
        });
        queryClient.invalidateQueries({
          queryKey: ["useTaggedItems", querier.baseUrl, projectHash],
        });
      },
    }
  );
}
