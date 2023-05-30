import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext } from "react";
import { z } from "zod";

export const DEFAULT_BASE_URL = "http://localhost:8000";

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
  points: z.record(PointSchema),
  shape: LabelRowObjectShapeSchema,
});

export const LabelsSchema = z.object({
  classification: z.any().array().nullish(),
  objects: LabelRowObjectSchema.array(),
});

export const GroupedTagsSchema = z.object({
  data: z.string().array(),
  label: z.string().array(),
});

export const ItemSchema = z.object({
  id: z.string(),
  url: z.string(),
  video_timestamp: z.number().nullish(),
  dataTitle: z.string().nullish(),
  editUrl: z.string(),
  metadata: z.object({
    metrics: z.record(z.coerce.string()),
    annotator: z.string().nullish(),
    labelClass: z.string().nullish(),
  }),
  tags: GroupedTagsSchema,
  labels: LabelsSchema,
});

const IdValueSchema = z.object({ id: z.string(), value: z.number() });
export type IdValue = z.infer<typeof IdValueSchema>;

export type GroupedTags = z.infer<typeof GroupedTagsSchema>;
export type Item = z.infer<typeof ItemSchema>;
export type ItemMetadata = Item["metadata"];
export type Point = z.infer<typeof PointSchema>;
export type Scope = "data_quality" | "label_quality" | "model_quality";

export const Item2DEmbeddingSchema = PointSchema.extend({
  id: z.string(),
  label: z.string(),
});

export type Item2DEmbedding = z.infer<typeof Item2DEmbeddingSchema>;

export const searchTypeOptions = {
  search: "Search",
  codegen: "Code Generation",
} as const;
type SearchType = keyof typeof searchTypeOptions;

const searchResultSchema = z.object({
  ids: z.string().array(),
  snippet: z.string().nullish(),
});
export type SeachResult = z.infer<typeof searchResultSchema>;

export const defaultTags = { data: [], label: [] };

export const getApi = (
  projectName: string,
  authToken?: string | null,
  baseUrl = DEFAULT_BASE_URL
) => {
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
    fetchHasPremiumFeatures: async () =>
      z
        .boolean()
        .parse(await (await fetcher(`${baseUrl}/premium_available`)).json()),
    fetchProject2DEmbeddings: async (selectedMetric: string) => {
      const url = `${baseUrl}/projects/${projectName}/2d_embeddings/${selectedMetric}`;
      try {
        const response = await (await fetcher(url)).json();
        return Item2DEmbeddingSchema.array().parse(response);
      } catch {
        return [];
      }
    },
    fetchProjectMetrics: async (scope: Scope) => {
      const queryParams = new URLSearchParams({
        ...(scope ? { scope } : {}),
      });
      const url = `${baseUrl}/projects/${projectName}/metrics?${queryParams}`;
      const response = await (await fetcher(url)).json();
      return z.string().array().parse(response);
    },
    fetchProjectItemIds: async (sortByMetric: string) => {
      const queryParams = new URLSearchParams({
        sort_by_metric: sortByMetric,
      });

      const url = `${baseUrl}/projects/${projectName}/items_id_by_metric?${queryParams}`;
      const result = await (await fetcher(url)).json();
      return IdValueSchema.array().parse(result);
    },
    fetchProjectItem: async (id: string) => {
      const { url, ...item } = await (
        await fetcher(
          `${baseUrl}/projects/${projectName}/items/${encodeURIComponent(id)}`
        )
      ).json();
      return ItemSchema.parse({ ...item, url }) as Item;
    },
    fetchedTaggedItems: async () =>
      z
        .record(GroupedTagsSchema)
        .transform((record) => new Map(Object.entries(record)))
        .parse(
          await (
            await fetcher(`${baseUrl}/projects/${projectName}/tagged_items`)
          ).json()
        ),
    fetchSimilarItems: async (
      id: string,
      selectedMetric: string,
      pageSize?: number
    ) => {
      const queryParams = new URLSearchParams({
        current_metric: selectedMetric,
        ...(pageSize ? { page_size: pageSize.toString() } : {}),
      });

      const url = `${baseUrl}/projects/${projectName}/similarities/${encodeURIComponent(
        id
      )}?${queryParams} `;
      const response = await fetcher(url).then((res) => res.json());
      return z.string().array().parse(response);
    },
    fetchHasSimilaritySearch: async (selectedMetric: string) => {
      const queryParams = new URLSearchParams({
        current_metric: selectedMetric,
      });
      const url = `${baseUrl}/projects/${projectName}/has_similarity_search?${queryParams} `;
      const response = await fetcher(url).then((res) => res.json());
      return z.boolean().parse(response);
    },
    fetchProjectTags: async () =>
      GroupedTagsSchema.parse(
        await (await fetcher(`${baseUrl}/projects/${projectName}/tags`)).json()
      ),
    updateItemTags: async (
      itemTags: { id: string; groupedTags: GroupedTags }[]
    ) => {
      const url = `${baseUrl}/projects/${projectName}/item_tags`;
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
      { scope, query, type }: { scope: Scope; query: string; type: SearchType },
      signal?: AbortSignal
    ) => {
      const queryParams = new URLSearchParams({ query, scope, type });

      const response = await fetcher(
        `${baseUrl}/projects/${projectName}/search?${queryParams}`,
        {
          signal,
        }
      );

      return searchResultSchema.parse(await response.json());
    },
  };
};

export const useApi = () => {
  const api = useContext(ApiContext);

  if (!api)
    throw new Error("useApi has to be used within <ProjectContext.Provider>");

  const queryClient = useQueryClient();

  return {
    itemTagsMutation: useMutation(
      ["updateItemTags"],
      (...args: Parameters<API["updateItemTags"]>) =>
        api.updateItemTags(...args),
      {
        onMutate: async (itemTagsList) => {
          Promise.all(
            itemTagsList.map(({ id }) =>
              queryClient.cancelQueries({ queryKey: ["item", id] })
            )
          );
          const previousItems = itemTagsList.map((itemTags) => {
            const itemKey = ["item", itemTags.id];
            const previousItem = queryClient.getQueryData(itemKey) as Item;
            queryClient.setQueryData(itemKey, {
              ...previousItem,
              tags: itemTags.groupedTags,
            });
            return previousItem;
          });

          await queryClient.cancelQueries(["tagged_items"]);
          const previousTaggedItems = queryClient.getQueryData([
            "tagged_items",
          ]) as Awaited<ReturnType<API["fetchedTaggedItems"]>>;
          const nextTaggedItems = new Map(previousTaggedItems);
          itemTagsList.forEach(({ id, groupedTags }) =>
            nextTaggedItems.set(id, groupedTags)
          );
          queryClient.setQueryData(["tagged_items"], nextTaggedItems);

          await queryClient.cancelQueries(["tags"]);
          const previousTags = queryClient.getQueryData([
            "tags",
          ]) as GroupedTags;
          const nextTags = itemTagsList.reduce(
            (nextTags, { groupedTags }) => {
              groupedTags.data.forEach(nextTags.data.add, nextTags.data);
              groupedTags.label.forEach(nextTags.label.add, nextTags.label);
              return nextTags;
            },
            { data: new Set<string>(), label: new Set<string>() }
          );
          queryClient.setQueryData(["tags"], {
            data: [...nextTags.data],
            label: [...nextTags.label],
          });

          return {
            previousTaggedItems,
            previousItems,
            previousTags,
          };
        },
        onError: (_, __, context) => {
          context?.previousItems.forEach((item) =>
            queryClient.setQueryData(["items", item.id], item)
          );
          queryClient.setQueryData(
            ["tagged_items"],
            context?.previousTaggedItems
          );
          queryClient.setQueryData(["tags"], context?.previousTags);
        },
        onSettled: (_, __, variables) => {
          variables.forEach(({ id }) =>
            queryClient.invalidateQueries({ queryKey: ["items", id] })
          );
          queryClient.invalidateQueries({ queryKey: ["tagged_items"] });
          queryClient.invalidateQueries({ queryKey: ["tags"] });
        },
      }
    ),
    fetchProjectTags: () =>
      useQuery(["tags"], api.fetchProjectTags, {
        initialData: defaultTags,
      }),
    fetchTaggedItems: () =>
      useQuery(["tagged_items"], api.fetchedTaggedItems, {
        initialData: new Map<string, GroupedTags>(),
      }),
    fetchItem: (...args: Parameters<API["fetchProjectItem"]>) =>
      useQuery(["item", ...args], () => api.fetchProjectItem(...args)),
    fetch2DEmbeddings: (
      selectedMetric: Parameters<API["fetchProject2DEmbeddings"]>[0]
    ) =>
      useQuery(
        ["2d_embeddings"],
        () => api.fetchProject2DEmbeddings(selectedMetric),
        { enabled: !!selectedMetric }
      ),
    search: (...args: Parameters<API["searchInProject"]>) =>
      api.searchInProject(...args),
  };
};

export type API = ReturnType<typeof getApi>;
export const ApiContext = createContext<API | null>(null);
