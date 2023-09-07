import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { PredictionDomain, SearchFilters } from "../../openapi/api";

export function usePredictionAnalysisSearch(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  domain: PredictionDomain,
  iou: number,
  orderBy: string,
  orderByDesc: boolean,
  offset: number,
  limit: number,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "usePredictionAnalysisSearch",
      queryContext.baseUrl,
      projectHash,
      predictionHash,
      domain,
      iou,
      orderBy,
      orderByDesc,
      offset,
      limit,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .predictionSearchProjectsV2ProjectHashPredictionsPredictionHashAnalyticsDomainSearchGet(
          projectHash,
          predictionHash,
          domain,
          iou,
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
