import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { AnalysisDomain, SearchFilters } from "../../openapi/api";

export function useProjectAnalysisSearch(
  queryContext: QueryContext,
  projectHash: string,
  domain: AnalysisDomain,
  orderBy: string,
  orderByDesc: boolean,
  offset: number,
  limit: number,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectAnalysisSearch",
      queryContext.baseUrl,
      projectHash,
      domain,
      orderBy,
      orderByDesc,
      offset,
      limit,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .metricSearchProjectsV2ProjectHashAnalysisDomainSearchGet(
          projectHash,
          domain,
          orderBy,
          orderByDesc,
          offset,
          limit,
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    options
  );
}
