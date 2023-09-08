import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { AnalysisDomain } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";

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
        .routeProjectSimilaritySearchProjectsV2ProjectHashAnalysisDomainSimilarityItemGet(
          projectHash,
          domain,
          item,
          "embedding_clip"
        )
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_ANALYTICS,
      cacheTime: CACHE_TIME_ANALYTICS,
    }
  );
}
