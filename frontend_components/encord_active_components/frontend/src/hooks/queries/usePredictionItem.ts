import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { QueryContext } from "../Context";

export function usePredictionItem(
  queryContext: QueryContext,
  projectHash: string,
  predictionHash: string,
  dataItem: string,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  return useQuery(
    [
      "usePredictionItem",
      queryContext.baseUrl,
      projectHash,
      predictionHash,
      dataItem,
    ],
    () =>
      queryContext
        .getProjectV2API()
        .getPredictionItemProjectsV2ProjectHashPredictionsPredictionHashPreviewDataItemGet(
          projectHash,
          predictionHash,
          dataItem
        )
        .then((r) => r.data),
    options
  );
}
