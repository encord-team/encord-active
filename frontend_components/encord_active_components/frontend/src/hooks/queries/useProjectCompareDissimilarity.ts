import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { AnalysisDomain } from "../../openapi/api";

export function useProjectCompareDissimilarity(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  compareProjectHash: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectCompareDissimilarity",
      queryContext.baseUrl,
      projectHash,
      domain,
      compareProjectHash,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .compareMetricDissimilarityProjectsV2ProjectHashAnalysisDomainProjectCompareMetricDissimilarityGet(
          projectHash,
          domain,
          compareProjectHash
        )
        .then((r) => r.data),
    options
  );
}
