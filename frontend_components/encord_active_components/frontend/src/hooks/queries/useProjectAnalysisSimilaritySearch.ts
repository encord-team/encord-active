import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { AnalysisDomain } from "../../openapi/api";

export function useProjectAnalysisSimilaritySearch(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  item: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisSimilaritySearch",
      queryContext.baseUrl,
      projectHash,
      domain,
      item,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .searchSimilarityProjectsV2ProjectHashAnalysisDomainSimilarityItemGet(
          projectHash,
          domain,
          item,
          "embedding_clip"
        )
        .then((r) => r.data),
    options
  );
}
