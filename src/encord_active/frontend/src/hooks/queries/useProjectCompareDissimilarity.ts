import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { AnalysisDomain } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function useProjectCompareDissimilarity(
  projectHash: string,
  domain: AnalysisDomain,
  compareProjectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "useProjectCompareDissimilarity",
      querier.baseUrl,
      projectHash,
      domain,
      compareProjectHash,
    ],
    () =>
      querier
        .getProjectAPI()
        .routeProjectCompareMetricDissimilarity(
          projectHash,
          domain,
          compareProjectHash
        )
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_ANALYTICS,
      cacheTime: CACHE_TIME_ANALYTICS,
    }
  );
}
