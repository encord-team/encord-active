import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { PredictionDomain, SearchFilters } from "../../openapi/api";
import { CACHE_TIME_ANALYTICS, STALE_TIME_ANALYTICS } from "../queryConstants";
import { useQuerier } from "../Context";

export function usePredictionAnalysisSearch(
  projectHash: string,
  predictionHash: string,
  domain: PredictionDomain,
  iou: number,
  orderBy: string,
  orderByData: boolean,
  orderByDesc: boolean,
  offset: number,
  limit: number,
  filters: SearchFilters | undefined = undefined,
  similarityItem: string | undefined = undefined,
  similarityText: string | undefined = undefined,
  similarityImage: File | undefined = undefined,
  options: Pick<UseQueryOptions, "enabled"> = {}
) {
  const querier = useQuerier();

  return useQuery(
    [
      "usePredictionAnalysisSearch",
      querier.baseUrl,
      projectHash,
      predictionHash,
      domain,
      iou,
      orderBy,
      orderByData,
      orderByDesc,
      offset,
      limit,
      filters,
      similarityItem,
      similarityText,
      similarityImage,
    ],
    () =>
      querier
        .getPredictionAPI()
        .routePredictionSearch(
          projectHash,
          predictionHash,
          domain,
          iou,
          orderByDesc,
          filters !== undefined ? JSON.stringify(filters) : undefined,
          orderBy,
          offset,
          limit,
          similarityText,
          similarityImage,
          similarityItem
        )
        .then((r) => r.data),
    {
      ...options,
      staleTime: STALE_TIME_ANALYTICS,
      cacheTime: CACHE_TIME_ANALYTICS,
    }
  );
}
