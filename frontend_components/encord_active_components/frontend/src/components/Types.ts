import {
  UseMutationOptions,
  UseMutationResult,
  UseQueryOptions,
  UseQueryResult,
} from "@tanstack/react-query";
import { Filters } from "./explorer/api";

export type DomainSearchFilters = {
  readonly metrics: Readonly<Record<string, readonly [number, number]>>;
  readonly enums: Readonly<Record<string, ReadonlyArray<string>>>;
  readonly reduction: null | {
    readonly reduction_hash: string;
    readonly min: [number, number];
    readonly max: [number, number];
  };
  readonly tags: null | ReadonlyArray<string>;
};

export type SearchFilters = {
  readonly data: null | DomainSearchFilters;
  readonly annotation: null | DomainSearchFilters;
};

export type ProjectView = {
  readonly project_hash: string;
  readonly title: string;
  readonly description: string;
};

export type ProjectMetricSummary = {
  readonly metrics: {
    readonly [metric_key: string]: {
      readonly title: string;
      readonly short_desc: string;
      readonly long_desc: string;
      readonly type: "normal" | "uint" | "ufloat" | "rank" | "sfloat";
    };
  };
  readonly enums: {
    readonly [enum_key: string]:
      | {
          readonly type: "ontology";
          readonly title: string;
        }
      | {
          readonly type: "enum";
          readonly title: string;
          readonly values: Readonly<Record<string, string>>;
        };
  };
};

type OntologyObjectAttributeBase = {
  readonly id: number | string;
  readonly featureNodeHash: string;
  readonly name: string;
};

export type OntologyObjectAttribute =
  | (OntologyObjectAttributeBase & {
      type: "checklist" | "radio";
      options: OntologyObjectAttributeOptions[];
    })
  | (OntologyObjectAttributeBase & {
      type: "text";
    });

export type OntologyObjectAttributeOptions = {
  readonly id: number | string;
  readonly featureNodeHash: string;
  readonly label: string;
  readonly value: string;
  readonly options?: (
    | OntologyObjectAttribute
    | OntologyObjectAttributeOptions
  )[];
};

export type ProjectSummary = {
  readonly name: string;
  readonly description: string;
  readonly ontology: {
    readonly objects: readonly {
      readonly id: string;
      readonly name: string;
      readonly color: string;
      readonly shape: string;
      readonly featureNodeHash: string;
      readonly attributes?: readonly OntologyObjectAttribute[];
    }[];
    readonly classifications: readonly {
      readonly id: string;
      readonly name: string;
      readonly color: string;
      readonly shape: string;
      readonly featureNodeHash: string;
      readonly attributes?: readonly OntologyObjectAttribute[];
    }[];
  };
  readonly local_project: boolean;
  readonly data: ProjectMetricSummary;
  readonly annotations: ProjectMetricSummary;
  readonly global: ProjectMetricSummary;
  readonly du_count: number;
  readonly frame_count: number;
  readonly annotation_count: number;
  readonly classification_count: number;
  readonly tags: {
    readonly [tag_hash: string]: string;
  };
  readonly preview: null | {
    readonly du_hash: string;
    readonly frame: number;
  };
};

export type ProjectEmbeddingReductions = {
  results: Readonly<
    Record<
      string,
      {
        readonly name: string;
        readonly description: string;
        readonly type: "umap";
      }
    >
  >;
};

export type ProjectAnalysisDomain = "data" | "annotation";

export type ProjectAnalysisSummary = {
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

export type ProjectAnalysisScatter = {
  readonly sampling: number;
  readonly samples: Array<{
    readonly x: number;
    readonly y: number;
    readonly n: number;
    readonly du_hash: string;
    readonly frame: number;
  }>;
};

export type ProjectAnalysisDistribution = {
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

export type ProjectAnalysisCompareMetricDissimilarity = {
  readonly results: Readonly<Record<string, number>>;
};

export type ProjectSearchResult = {
  readonly truncated: boolean;
  readonly results: ReadonlyArray<string>;
};

type Point = {
  readonly x: number;
  readonly y: number;
};

export type ProjectPreviewItemResult = {
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

export type ProjectItemDetailedSummary = {
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

export type ProjectSimilarityResult = {
  readonly results: ReadonlyArray<{
    readonly du_hash: string;
    readonly frame: number;
    readonly object_hash?: string;
    readonly similarity: number;
  }>;
};

export type CreateSubsetMutationArguments = {
  readonly project_title: string;
  readonly project_description?: string | undefined;
  readonly dataset_title: string;
  readonly dataset_description?: string | undefined;
  readonly filters: Filters;
  readonly ids: string[];
};

export type CreateTagMutationArguments = {
  readonly tag_title: string;
  readonly tag_description?: string | undefined;
  readonly objects: readonly {
    readonly du_hash: string;
    readonly frame: number;
    readonly object_hash?: string | undefined;
  }[];
};

export type UploadToEncordMutationArguments = {
  readonly dataset_title: string;
  readonly dataset_description?: string | undefined;
  readonly project_title: string;
  readonly project_description?: string | undefined;
  readonly ontology_title: string;
  readonly ontology_description?: string | undefined;
};

export type PredictionView = {
  readonly name: string;
  readonly prediction_hash: string;
};

export type PaginationResult<T> = {
  readonly total: number;
  readonly results: ReadonlyArray<T>;
};

export type ProjectPredictionSummary = {
  readonly classification_only: boolean;
  readonly classification_tTN: number;
  readonly classification_accuracy: number;
  readonly num_frames: number;
  readonly mAP: number;
  readonly mAR: number;
  readonly mP: number;
  readonly mR: number;
  readonly mF1: number;
  readonly tTP: number;
  readonly tFP: number;
  readonly tFN: number;
  readonly feature_properties: Readonly<Record<string,{
    readonly ap: number;
    readonly ar: number;
    readonly p: number;
    readonly r: number;
    readonly f1: number;
    readonly tp: number;
    readonly fp: number;
    readonly fn: number;
  }>>;
  readonly prs: Readonly<
    Record<
      string,
      Array<{
        readonly p: number;
        readonly r: number;
      }>
    >
  >;
  readonly correlation: Readonly<Record<string, number>>;
  readonly importance: Readonly<Record<string, number>>;
};

export type ProjectMetricPerformance = {
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

export interface QueryAPI {
  // === Project search ===
  useListProjectViews(
    search: string,
    offset: number,
    limit: number,
  ): UseQueryResult<PaginationResult<ProjectView>>;

  // === Project summary ===
  useProjectSummary(
    projectHash: string,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectSummary>;
  useProjectListEmbeddingReductions(
    projectHash: string,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectEmbeddingReductions>;

  // === Project analytics ===
  useProjectAnalysisSummary(
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectAnalysisSummary>;
  useProjectAnalysisMetricScatter(
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    xMetric: string,
    yMetric: string,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectAnalysisScatter>;
  useProjectAnalysisDistribution(
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    metricOrEnum: string,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectAnalysisDistribution>;
  useProjectAnalysisCompareMetricDissimilarity(
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    compareProjectHash: string,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectAnalysisCompareMetricDissimilarity>;
  useProjectAnalysisSearch(
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    filters: SearchFilters,
    orderBy: null | string,
    desc: boolean,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectSearchResult>;

  // == Project visualisation ===
  useProjectItemPreview(
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash?: string | undefined | null,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectPreviewItemResult>;
  useProjectItemDetails(
    projectHash: string,
    duHash: string,
    frame: number,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectItemDetailedSummary>;
  useProjectItemSimilarity(
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash?: string | undefined | null,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectSimilarityResult>;

  // == Project actions ===
  useProjectMutationCreateSubset(
    projectHash: string,
    options?: Pick<
      UseMutationOptions<string, unknown, CreateSubsetMutationArguments>,
      "onError" | "onSuccess" | "onSettled"
    >,
  ): UseMutationResult<string, unknown, CreateSubsetMutationArguments>;
  useProjectMutationCreateTag(
    projectHash: string,
    options?: Pick<
      UseMutationOptions<string, unknown, CreateTagMutationArguments>,
      "onError" | "onSuccess" | "onSettled"
    >,
  ): UseMutationResult<string, unknown, CreateTagMutationArguments>;
  useProjectMutationUploadToEncord(
    projectHash: string,
    options?: Pick<
      UseMutationOptions<string, unknown, UploadToEncordMutationArguments>,
      "onError" | "onSuccess" | "onSettled"
    >,
  ): UseMutationResult<string, unknown, UploadToEncordMutationArguments>;

  // == Project predictions ===
  useProjectListPredictions(
    projectHash: string,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<PaginationResult<PredictionView>>;
  useProjectPredictionSummary(
    projectHash: string,
    predictionHash: string,
    iou: number,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectPredictionSummary>;
  useProjectPredictionMetricPerformance(
    projectHash: string,
    predictionHash: string,
    buckets: number,
    iou: number,
    metric: string,
    options?: Pick<UseQueryOptions, "enabled">,
  ): UseQueryResult<ProjectMetricPerformance>;
}
