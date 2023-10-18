import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQuerier } from "../Context";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";

export function useProjectMutationFiltersAddTag(projectHash: string) {
  const querier = useQuerier();
  const queryClient = useQueryClient();

  return useMutation(
    ["useProjectMutationFiltersAddTag", querier.baseUrl, projectHash],
    (tag: {
      domain: AnalysisDomain;
      filters: SearchFilters;
      tags: Array<string>;
    }) =>
      querier
        .getProjectAPI()
        .routeFilterTagAll(projectHash, tag.domain, {
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
          await queryClient.invalidateQueries([
            "useProjectListTagsMeta",
            querier.baseUrl,
            projectHash,
          ]);
          await queryClient.invalidateQueries([
            "useProjectAnalysisSearch",
            querier.baseUrl,
            projectHash,
          ]);
        })
  );
}
