import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useState } from "react";
import { z } from "zod";
import { apiUrl } from "../../constants";
import { takeDataId } from "./id";

export const PointSchema = z.object({ x: z.number(), y: z.number() });

const LabelRowObjectShapeSchema = z.union([
  z.literal("polygon"),
  z.literal("polyline"),
  z.literal("point"),
  z.literal("bounding_box"),
  z.literal("rotatable_bounding_box"),
]);

export const LabelRowObjectSchema = z.object({
  color: z.string(),
  confidence: z.number(),
  createdAt: z.string(),
  createdBy: z.string(),
  featureHash: z.string(),
  lastEditedAt: z.string().nullish(),
  lastEditedBy: z.string().nullish(),
  manualAnnotation: z.boolean(),
  name: z.string(),
  objectHash: z.string(),
  points: z.record(PointSchema).nullish(),
  boundingBoxPoints: z.record(PointSchema).nullish(),
  shape: LabelRowObjectShapeSchema,
});

export const ObjectPredictionSchema = z.object({
  color: z.string(),
  confidence: z.number(),
  featureHash: z.string(),
  name: z.string(),
  objectHash: z.string(),
  points: z.record(PointSchema).nullish(),
  boundingBoxPoints: z.record(PointSchema).nullish(),
  shape: LabelRowObjectShapeSchema,
});

export const ClassificationPredictionSchema = z.object({
  confidence: z.number(),
  featureHash: z.string(),
  name: z.string(),
});

export const LabelsSchema = z.object({
  classification: z.any().array().nullish(),
  objects: LabelRowObjectSchema.array(),
});

export const PredictionSchema = z.object({
  classification: ClassificationPredictionSchema.array().nullish(),
  objects: ObjectPredictionSchema.array(),
});

export const GroupedTagsSchema = z.object({
  data: z.string().array(),
  label: z.string().array(),
});

export const ItemSchema = z.object({
  id: z.string(),
  url: z.string(),
  videoTimestamp: z.number().nullish(),
  dataTitle: z.string().nullish(),
  editUrl: z.string().nullish(),
  metadata: z.object({
    metrics: z.record(z.coerce.string()),
    annotator: z.string().nullish(),
    labelClass: z.string().nullish(),
  }),
  tags: GroupedTagsSchema,
  labels: LabelsSchema,
  predictions: PredictionSchema,
});

const IdValueSchema = z.object({ id: z.string(), value: z.number() });
export type IdValue = z.infer<typeof IdValueSchema>;

export type GroupedTags = z.infer<typeof GroupedTagsSchema>;
export type Item = z.infer<typeof ItemSchema>;
export type ItemMetadata = Item["metadata"];
export type Point = z.infer<typeof PointSchema>;

export const RangeSchema = z.object({ min: z.number(), max: z.number() });
export const classificationsPredictionOutcomes = [
  "Correct Classifications",
  "Misclassifications",
] as const;
export type ClassificationsPredictionOutcome =
  (typeof classificationsPredictionOutcomes)[number];
export const objectPredictionOutcomes = [
  "True Positive",
  "False Positive",
  "False Negative",
] as const;
export type ObjectPredictionOutcome = (typeof objectPredictionOutcomes)[number];

const PredictionTypeSchema = z.enum(["object", "classification"]);

export const FilterSchema = z.object({
  tags: GroupedTagsSchema,
  object_classes: z.string().array().optional(),
  workflow_stages: z.string().array().optional(),
  text: z.record(z.coerce.string(), z.string()).optional(),
  categorical: z.record(z.coerce.string(), z.number().array()).optional(),
  range: z.record(z.coerce.string(), RangeSchema),
  datetime_range: z
    .record(z.coerce.string(), z.object({ start: z.date(), end: z.date() }))
    .optional(),
  prediction_filters: z
    .discriminatedUnion("type", [
      z.object({
        type: z.literal(PredictionTypeSchema.enum.object),
        outcome: z.enum(objectPredictionOutcomes).optional(),
        iou_threshold: z.number().optional(),
      }),
      z.object({
        type: z.literal(PredictionTypeSchema.enum.classification),
        outcome: z.enum(classificationsPredictionOutcomes).optional(),
        iou_threshold: z.number().optional(),
      }),
    ])
    .optional(),
});

export type Filters = z.infer<typeof FilterSchema>;
export type PredictionType = z.infer<typeof PredictionTypeSchema>;

export type PredictionOutcome =
  | ObjectPredictionOutcome
  | ClassificationsPredictionOutcome;

const EmbeddingTypeSchema = z.union([
  z.literal("classification"),
  z.literal("object"),
  z.literal("hu_moments"),
  z.literal("image"),
  z.literal("video"),
]);

export const MetricSchema = z.object({
  name: z.string(),
  // filterKey: FilterSchema.keyof(),
  embeddingType: EmbeddingTypeSchema,
  range: RangeSchema,
});

export const MetricDefinitionsSchema = z.object({
  data: MetricSchema.array(),
  annotation: MetricSchema.array(),
  prediction: MetricSchema.array(),
});

const ScopeSchema = MetricDefinitionsSchema.keyof();

export type Metric = z.infer<typeof MetricSchema>;
export type Scope = z.infer<typeof ScopeSchema>;

export const Item2DEmbeddingSchema = PointSchema.extend({
  id: z.string(),
  label: z.string(),
  score: z.number().optional(),
});

export type Item2DEmbedding = z.infer<typeof Item2DEmbeddingSchema>;

export const searchTypeOptions = {
  search: "Search",
  codegen: "Code Generation",
} as const;
export type SearchType = keyof typeof searchTypeOptions;

const searchResultSchema = z.object({
  ids: z.string().array(),
  snippet: z.string().nullish(),
});
export type SeachResult = z.infer<typeof searchResultSchema>;

export const defaultTags = { data: [], label: [] };

export const getApi = (projectHash: string, authToken?: string | null) => {
  const updateOptions = (options: Parameters<typeof fetch>[1]) => {
    if (!authToken) return options;

    return {
      ...options,
      headers: {
        ...options?.headers,
        Authorization: `Bearer ${authToken}`,
      },
    };
  };

  const fetcher: typeof fetch = (url, options) =>
    fetch(url, updateOptions(options));

  return {
    projectHash,
    fetchHasPremiumFeatures: async () =>
      z
        .boolean()
        .parse(await (await fetcher(`${apiUrl}/premium_available`)).json()),
    fetchAvailablePredictionTypes: async () =>
      PredictionTypeSchema.array().parse(
        await (
          await fetcher(`${apiUrl}/projects/${projectHash}/prediction_types`)
        ).json(),
      ),
    fetchProject2DEmbeddings: async (
      embedding_type: Metric["embeddingType"],
      filters: Filters,
    ) => {
      const url = `${apiUrl}/projects/${projectHash}/2d_embeddings`;
      try {
        const response = await (
          await fetcher(url, {
            method: "post",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ embedding_type, filters }),
          })
        ).json();
        return Item2DEmbeddingSchema.array().parse(response);
      } catch {
        return [];
      }
    },
    fetchProjectMetrics: async (
      scope: Scope,
      prediction_type?: PredictionType,
      prediction_outcome?: PredictionOutcome,
    ): Promise<z.infer<typeof MetricDefinitionsSchema>> => {
      const queryParams = new URLSearchParams({
        ...(scope ? { scope } : {}),
        ...(prediction_type ? { prediction_type } : {}),
        ...(prediction_outcome ? { prediction_outcome } : {}),
      });
      const url = `${apiUrl}/projects/${projectHash}/metrics?${queryParams}`;
      const response = await (await fetcher(url)).json();
      return MetricDefinitionsSchema.parse(response);
    },
    fetchProjectItemIds: async (
      scope: Scope,
      sort_by_metric: string,
      filters: Filters,
      itemSet: Set<string>,
    ) => {
      const result = await (
        await fetcher(`${apiUrl}/projects/${projectHash}/item_ids_by_metric`, {
          method: "post",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            scope,
            sort_by_metric,
            filters,
            ids: [...itemSet],
          }),
        })
      ).json();
      return IdValueSchema.array().parse(result);
    },
    requestResetSignedURL: async (id: string) => {
      await fetcher(
        `${apiUrl}/projects/${projectHash}/reset_signed_url/${encodeURIComponent(
          id,
        )}`,
        { method: "POST" },
      );
    },
    fetchProjectItem: async (id: string, iou?: number) => {
      const queryParams = new URLSearchParams({
        ...(iou != null ? { iou: iou.toString() } : {}),
      });

      const item = await (
        await fetcher(
          `${apiUrl}/projects/${projectHash}/items/${encodeURIComponent(
            id,
          )}?${queryParams}`,
        )
      ).json();
      return ItemSchema.parse(item) as Item;
    },
    fetchedTaggedItems: async () =>
      z
        .record(GroupedTagsSchema)
        .transform((record) => new Map(Object.entries(record)))
        .parse(
          await (
            await fetcher(`${apiUrl}/projects/${projectHash}/tagged_items`)
          ).json(),
        ),
    fetchSimilarItems: async (
      id: string,
      embeddingType: Metric["embeddingType"],
      pageSize?: number,
    ) => {
      const queryParams = new URLSearchParams({
        embedding_type: embeddingType,
        ...(pageSize ? { page_size: pageSize.toString() } : {}),
      });

      const url = `${apiUrl}/projects/${projectHash}/similarities/${encodeURIComponent(
        id,
      )}?${queryParams}`;
      const response = await fetcher(url).then((res) => res.json());
      return z.string().array().parse(response);
    },
    fetchHasSimilaritySearch: async (
      embeddingType: Metric["embeddingType"],
    ) => {
      const queryParams = new URLSearchParams({
        embedding_type: embeddingType,
      });
      const url = `${apiUrl}/projects/${projectHash}/has_similarity_search?${queryParams} `;
      const response = await fetcher(url).then((res) => res.json());
      return z.boolean().parse(response);
    },
    updateItemTags: async (
      itemTags: { id: string; groupedTags: GroupedTags }[],
    ) => {
      const url = `${apiUrl}/projects/${projectHash}/item_tags`;
      const data = itemTags.map(({ id, groupedTags }) => ({
        id,
        grouped_tags: groupedTags,
      }));

      return fetcher(url, {
        method: "put",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
    },
    searchInProject: async (
      args: { scope: Scope; query: string; type: SearchType; filters: Filters },
      signal?: AbortSignal,
    ) => {
      const formData = new FormData();
      Object.entries(args).forEach(([key, value]) =>
        formData.append(
          key,
          typeof value === "object" ? JSON.stringify(value) : value,
        ),
      );
      const response = await fetcher(
        `${apiUrl}/projects/${projectHash}/search`,
        {
          method: "post",
          headers: {},
          body: formData,
          signal,
        },
      );

      return searchResultSchema.parse(await response.json());
    },
  };
};

export const useApi = () => {
  const [invalidatedUrls, setInvalidatedUrls] = useState(
    new Map<string, number>(),
  );
  const apiContext = useContext(ApiContext);

  if (!apiContext)
    throw new Error("useApi has to be used within <ProjectContext.Provider>");

  const { projectHash, ...api } = apiContext;

  const queryClient = useQueryClient();

  return {
    itemTagsMutation: useMutation(
      ["updateItemTags"],
      (...args: Parameters<API["updateItemTags"]>) =>
        api.updateItemTags(...args),
      {
        onSettled: () => {
          queryClient.invalidateQueries({ queryKey: [projectHash, "item"] });
          queryClient.invalidateQueries({
            queryKey: [projectHash, "tagged_items"],
          });
        },
      },
    ),
    fetchTaggedItems: () =>
      useQuery([projectHash, "tagged_items"], api.fetchedTaggedItems, {
        initialData: new Map<string, GroupedTags>(),
      }),
    fetchItem: (...args: Parameters<API["fetchProjectItem"]>) =>
      useQuery([projectHash, "item", ...args], () =>
        api.fetchProjectItem(...args),
      ),
    fetch2DEmbeddings: (
      embeddingType: Parameters<API["fetchProject2DEmbeddings"]>[0],
      filters: Filters,
    ) =>
      useQuery(
        [projectHash, "2d_embeddings", embeddingType, JSON.stringify(filters)],
        () => api.fetchProject2DEmbeddings(embeddingType, filters),
        { enabled: !!embeddingType, staleTime: Infinity },
      ),
    search: (...args: Parameters<API["searchInProject"]>) =>
      api.searchInProject(...args),
    fetchAvailablePredictionTypes: (
      ...args: Parameters<API["fetchAvailablePredictionTypes"]>
    ) =>
      useQuery(
        [projectHash, "available_prediction_types"],
        () => api.fetchAvailablePredictionTypes(...args),
        { staleTime: Infinity },
      ),
    invalidateItemURL: (...args: Parameters<API["fetchProjectItem"]>) =>
      useMutation(
        [projectHash, "invalidate_query", ...args],
        () => {
          const id = args[0];

          const oldTs = invalidatedUrls.get(id);
          if (oldTs && Date.now() - oldTs < 1000 * 60 * 10)
            return new Promise(() => undefined);

          return api.requestResetSignedURL(args[0]);
        },
        {
          onSettled: () => {
            const id = args[0];
            const oldTs = invalidatedUrls.get(id);
            const ts = Date.now();
            if (oldTs && ts - oldTs < 1000 * 60 * 10) return;

            const newInvalidatedUrls = new Map(invalidatedUrls.entries());
            newInvalidatedUrls.set(id, ts);
            setInvalidatedUrls(newInvalidatedUrls);
            queryClient.refetchQueries({
              queryKey: [projectHash, "item", args[0]],
            });
          },
        },
      ),
  };
};

export type API = ReturnType<typeof getApi>;
export const ApiContext = createContext<API | null>(null);
