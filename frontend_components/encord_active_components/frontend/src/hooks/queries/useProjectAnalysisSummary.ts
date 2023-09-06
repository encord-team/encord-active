import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";

export function useProjectAnalysisSummary(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisSummary",
      queryContext.baseUrl,
      projectHash,
      domain,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .metricSummaryProjectsV2ProjectHashAnalysisDomainSummaryGet(
          projectHash,
          domain,
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    options
  );
}
