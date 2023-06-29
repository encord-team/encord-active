import { UseQueryOptions, UseQueryResult } from "@tanstack/react-query";

export type ActiveProjectView = {
  readonly project_hash: string;
  readonly title: string;
  readonly description: string;
};

export type ActiveProjectMetricSummary = {
  readonly metrics: {
    readonly [metric_key: string]: {
      readonly title: string;
      readonly short_desc: string;
      readonly long_desc: string;
      readonly type: "normal" | "uint" | "ufloat" | "rank";
    };
  };
  readonly enums: {
    readonly [enum_key: string]:
      | {
          readonly type: "ontology";
        }
      | {
          readonly type: "enum";
          readonly title: string;
          readonly values: Readonly<Record<string, string>>;
        };
  };
};

export type ActiveProjectSummary = {
  readonly name: string;
  readonly description: string;
  readonly ontology: {
    readonly objects: readonly {
      readonly id: string;
      readonly name: string;
      readonly color: string;
      readonly shape: string;
      readonly featureNodeHash: string;
    }[];
    readonly classifications: readonly {
      readonly id: string;
      readonly name: string;
      readonly color: string;
      readonly shape: string;
      readonly featureNodeHash: string;
    }[];
  };
  readonly data: ActiveProjectMetricSummary;
  readonly annotations: ActiveProjectMetricSummary;
  readonly tags: {
    readonly [tag_hash: string]: string;
  };
  readonly preview: null | {
    readonly du_hash: string;
    readonly frame: number;
  };
};

export type ActiveProjectAnalysisDomain = "data" | "annotation";

export type ActiveProjectAnalysisSummary = {
  readonly count: number;
  readonly metrics: {
    [metric_key: string]: {
      readonly min: number;
      readonly q1: number;
      readonly median: number;
      readonly q3: number;
      readonly max: number;
      readonly count: number;
      readonly moderate: number;
      readonly severe: number;
    };
  };
  readonly enums: {
    readonly [enum_key: string]: {
      readonly _type_missing: null;
    };
  };
};

export type ActiveProjectAnalysisScatter = {
  readonly sampling: number;
  readonly samples: Array<{
    readonly x: number;
    readonly y: number;
    readonly n: number;
    readonly du_hash: string;
    readonly frame: number;
  }>;
};

export type ActiveProjectAnalysisDistribution = {
  readonly results:
    | ReadonlyArray<{
        readonly count: number;
        readonly group: number;
      }>
    | ReadonlyArray<{
        readonly count: number;
        readonly group: string;
      }>;
};

export type ActiveProjectSearchResult = {
  readonly truncated: boolean;
  readonly results: ReadonlyArray<{
    readonly du_hash: string;
    readonly frame: number;
    readonly object_hash?: string;
  }>;
};

type Point = {
  readonly x: number;
  readonly y: number;
};

export type ActiveProjectPreviewItemResult = {
  readonly url: string;

  // timestamp if video preview is needed.
  readonly timestamp: null | number;

  // Objects to display to the user.
  readonly objects: ReadonlyArray<
    {
      readonly name: string;
      readonly color: string;
      readonly objectHash: string;
      readonly featureHash: string;

      // Extra metadata (probably not used atm)
      readonly confidence?: number;
      readonly createdAt?: string;
      readonly lastEditedAt?: string;
      readonly lastEditedBy?: string;
      readonly manualAnnotation?: boolean;
    } & (
      | {
          readonly shape: "bounding_box";
          readonly boundingBox: Readonly<Record<"x" | "y" | "w" | "h", number>>;
        }
      | {
          readonly shape: "polygon";
          readonly polygon: Readonly<Record<number, Point>>;
        }
      | {
          readonly shape: "polyline";
          readonly polyline: Readonly<Record<number, Point>>;
        }
      | {
          readonly shape: "rotatable_bounding_box";
          readonly rotatableBoundingBox: Readonly<
            Record<"x" | "y" | "w" | "h" | "theta", number>
          >;
        }
      | {
          readonly shape: "point";
          readonly point: { "0": Point };
        }
    )
  >;

  // tag hash list
  readonly tags: ReadonlyArray<string>;
};

export type ActiveProjectItemDetailedSummary = {
  readonly metrics: Readonly<Record<string, number>>;
  readonly annotations: ReadonlyArray<{
    readonly metrics: Readonly<Record<string, number>>;
  }>;
  readonly dataset_title: string;
  readonly dataset_hash: string;
  readonly data_title: string;
  readonly data_hash: string;
  readonly label_hash: string;
  readonly num_frames: number;
  readonly frames_per_second: number;
  readonly data_type: "image" | "video" | "img_group" | "img_sequence";
};

export type ActiveProjectSimilarityResult = {
  readonly results: ReadonlyArray<{
    readonly du_hash: string;
    readonly frame: number;
    readonly object_hash?: string;
    readonly similarity: number;
  }>;
};

export type ActivePredictionView = {
  readonly name: string;
  readonly prediction_hash: string;
};

export type ActivePaginationResult<T> = {
  readonly total: number;
  readonly results: ReadonlyArray<T>;
};

export type ActiveProjectPredictionSummary = {
  readonly mAP: number;
  readonly mAR: number;
  readonly precisions: Readonly<Record<string, number>>;
  readonly recalls: Readonly<Record<string, number>>;
  readonly correlation: Readonly<Record<string, number>>;
  readonly importance: Readonly<Record<string, number>>;
  readonly prs: Readonly<
    Record<
      string,
      Array<{
        readonly p: number;
        readonly r: number;
      }>
    >
  >;
};

export type ActiveProjectMetricPerformance = {
  readonly precision: Readonly<
    Record<
      string,
      ReadonlyArray<{
        readonly m: number;
        readonly a: number;
        readonly n: number;
      }>
    >
  >;
  readonly fns: Readonly<
    Record<
      string,
      ReadonlyArray<{
        readonly m: number;
        readonly a: number;
        readonly n: number;
      }>
    >
  >;
};

export interface ActiveQueryAPI {
  // === Project search ===
  useListProjectViews(
    search: string,
    offset: number,
    limit: number
  ): UseQueryResult<ActivePaginationResult<ActiveProjectView>>;

  // === Project summary ===
  useProjectSummary(
    projectHash: string,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectSummary>;

  // === Project analytics ===
  useProjectAnalysisSummary(
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectAnalysisSummary>;
  useProjectAnalysisMetricScatter(
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    xMetric: string,
    yMetric: string,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectAnalysisScatter>;
  useProjectAnalysisDistribution(
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    metricOrEnum: string,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectAnalysisDistribution>;
  useProjectAnalysisSearch(
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    metricFilters: null | Readonly<Record<string, readonly [number, number]>>,
    metricOutliers: null | Readonly<Record<string, "warning" | "severe">>,
    enumFilters: null | Readonly<
      Partial<Record<string, ReadonlyArray<string>>>
    >,
    orderBy: null | string,
    desc: boolean,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectSearchResult>;

  // == Project visualisation ===
  useProjectItemPreview(
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash?: string | undefined | null,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectPreviewItemResult>;
  useProjectItemDetails(
    projectHash: string,
    duHash: string,
    frame: number,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectItemDetailedSummary>;
  useProjectItemSimilarity(
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash?: string | undefined | null,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectSimilarityResult>;

  // == Project predictions ===
  useProjectListPredictions(
    projectHash: string,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActivePaginationResult<ActivePredictionView>>;
  useProjectPredictionSummary(
    projectHash: string,
    predictionHash: string,
    iou: number,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectPredictionSummary>;
  useProjectPredictionMetricPerformance(
    projectHash: string,
    predictionHash: string,
    buckets: number,
    iou: number,
    metric: string,
    options?: Pick<UseQueryOptions, "enabled">
  ): UseQueryResult<ActiveProjectMetricPerformance>;
}
