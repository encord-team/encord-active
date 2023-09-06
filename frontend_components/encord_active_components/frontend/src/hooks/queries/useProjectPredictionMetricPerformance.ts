import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";
import {
  PredictionMetricPerformanceProjectsV2ProjectHashPredictionsPredictionHashMetricPerformanceGetBucketsEnum,
  SearchFilters,
} from "../../openapi/api";

export function useProjectPredictionMetricPerformance(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  iou: number,
  metricName: string,
  buckets: 10 | 100 | 1000 | undefined = undefined,
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
      metricName,
      buckets,
      filters,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .predictionMetricPerformanceProjectsV2ProjectHashPredictionsPredictionHashMetricPerformanceGet(
          projectHash,
          predictionHash,
          iou,
          metricName,
          buckets === undefined
            ? undefined
            : (String(
                buckets
              ) as PredictionMetricPerformanceProjectsV2ProjectHashPredictionsPredictionHashMetricPerformanceGetBucketsEnum),
          filters !== undefined ? JSON.stringify(filters) : undefined
        )
        .then((r) => r.data),
    options
  );
}
