import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  AnalysisDomain,
  GetMetricDistributionProjectsV2ProjectHashAnalysisDomainDistributionGetBucketsEnum,
  SearchFilters,
} from "../../openapi/api";

export function useProjectAnalysisDistribution(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  group: string,
  buckets: 10 | 100 | 1000 | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisDistribution",
      queryContext.baseUrl,
      projectHash,
      domain,
      group,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .getMetricDistributionProjectsV2ProjectHashAnalysisDomainDistributionGet(
          projectHash,
          domain,
          group,
          buckets === undefined
            ? undefined
            : (String(
                buckets
              ) as GetMetricDistributionProjectsV2ProjectHashAnalysisDomainDistributionGetBucketsEnum),
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    options
  );
}
