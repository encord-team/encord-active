import { SearchFilters } from "../../openapi/api";

export type ExplorerFilterState = {
  readonly analysisDomain: "data" | "annotation";
  readonly filters: SearchFilters;
  readonly orderBy: string;
  readonly desc: boolean;
  readonly iou: number | undefined;
  readonly predictionOutcome: "tp" | "fp" | "fn" | undefined;
  readonly predictionHash: string | undefined;
};
