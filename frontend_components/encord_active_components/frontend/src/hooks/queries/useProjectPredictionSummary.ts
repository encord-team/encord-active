import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import { SearchFilters } from "../../openapi/api";

export function useProjectPredictionSummary(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  iou: number,
  filters: SearchFilters | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "useProjectPredictionSummary",
      queryContext.baseUrl,
      projectHash,
      predictionHash,
      iou,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .getProjectPredictionSummaryProjectsV2ProjectHashPredictionsPredictionHashSummaryGet(
          predictionHash,
          projectHash,
          iou,
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    options
  );
}
