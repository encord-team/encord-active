import { PredictionDomain, SearchFilters } from "../../openapi/api";
export type MetricDomain = "Data" | "Label";
export type ExplorerFilterState = {
  readonly analysisDomain: "data" | "annotation";
  readonly filters: SearchFilters;
  readonly orderBy: string;
  readonly desc: boolean;
  readonly iou: number;
  readonly predictionOutcome: PredictionDomain;
  readonly predictionHash: string | undefined;
};
