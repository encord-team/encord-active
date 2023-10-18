import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";

export function useProjectMutationFiltersRemoveTag(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationFiltersRemoveTag", querier.baseUrl, projectHash],
    (tag: {
      domain: AnalysisDomain;
      filters: SearchFilters;
      tags: Array<string>;
    }) =>
      querier
        .getProjectAPI()
        .routeFilterUntagAll(projectHash, tag.domain, {
          filters: tag.filters,
          tags: tag.tags,
        })
        .then(async () => {
          await queryClient.invalidateQueries({
            queryKey: ["useProjectItem", querier.baseUrl, projectHash],
          });
          await queryClient.invalidateQueries({
            queryKey: [
              "useProjectFilterListTags",
              querier.baseUrl,
              projectHash,
            ],
          });
        })
  );
}
