import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  AnalysisDomain,
  Scatter2dDataMetricProjectsV2ProjectHashAnalysisDomainScatterGetBucketsEnum,
  SearchFilters,
} from "../../openapi/api";

export function useProjectAnalysisMetricScatter(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  xMetric: string,
  yMetric: string,
  buckets: 10 | 100 | 1000 | undefined = undefined,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisMetricScatter",
      queryContext.baseUrl,
      projectHash,
      domain,
      xMetric,
      yMetric,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .scatter2dDataMetricProjectsV2ProjectHashAnalysisDomainScatterGet(
          projectHash,
          domain,
          xMetric,
          yMetric,
          buckets === undefined
            ? undefined
            : (String(
                buckets
              ) as Scatter2dDataMetricProjectsV2ProjectHashAnalysisDomainScatterGetBucketsEnum),
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    options
  );
}
