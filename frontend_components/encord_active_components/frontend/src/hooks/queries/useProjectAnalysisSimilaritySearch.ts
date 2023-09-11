import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { AnalysisDomain } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectAnalysisSimilaritySearch(
  projectHash: string,
  domain: AnalysisDomain,
  item: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "useProjectAnalysisSimilaritySearch",
      querier.baseUrl,
      projectHash,
      domain,
      item,
    ],
    () =>
      querier
        .getProjectV2API()
        .routeProjectSimilaritySearchApiProjectsV2ProjectHashAnalysisDomainSimilarityItemGet(
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
