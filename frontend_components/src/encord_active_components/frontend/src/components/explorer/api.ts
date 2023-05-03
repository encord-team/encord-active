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

export const fetchHasPremiumFeatures = async (baseUrl = DEFAULT_BASE_URL) =>
  z.boolean().parse(await (await fetch(`${baseUrl}/premium_available`)).json());

export const fetchProject2DEmbeddings =
  (baseUrl: string, projectName: string) => async (selectedMetric: string) => {
    const url = `${baseUrl}/projects/${projectName}/2d_embeddings/${selectedMetric}`;
    try {
      const response = await (await fetch(url)).json();
      return Item2DEmbeddingSchema.array().parse(response);
    } catch {
      return [];
    }
  };

export const fetchProjectMetrics =
  (baseUrl: string, projectName: string) => async (scope: Scope) => {
    const queryParams = new URLSearchParams({
      ...(scope ? { scope } : {}),
    });
    const url = `${baseUrl}/projects/${projectName}/metrics?${queryParams}`;
    const response = await (await fetch(url)).json();
    return z.string().array().parse(response);
  };

export const fetchProjectItemIds =
  (baseUrl: string, projectName: string) => async (sortByMetric: string) => {
    const queryParams = new URLSearchParams({
      sort_by_metric: sortByMetric,
    });

    const url = `${baseUrl}/projects/${projectName}/items_id_by_metric?${queryParams}`;
    const result = await (await fetch(url)).json();
    return IdValueSchema.array().parse(result);
  };

export const fetchProjectItem =
  (baseUrl: string, projectName: string) => async (id: string) => {
    const { url, ...item } = await (
      await fetch(
        `${baseUrl}/projects/${projectName}/items/${encodeURIComponent(id)}`
      )
    ).json();
    return ItemSchema.parse({ ...item, url: `${baseUrl}/${url}` }) as Item;
  };

export const fetchedTaggedItems = async (
  baseUrl: string,
  projectName: string
) =>
  z
    .record(GroupedTagsSchema)
    .transform((record) => new Map(Object.entries(record)))
    .parse(
      await (
        await fetch(`${baseUrl}/projects/${projectName}/tagged_items`)
      ).json()
    );

export const fetchSimilarItems =
  (baseUrl: string, projectName: string) =>
  async (id: string, selectedMetric: string, pageSize?: number) => {
    const queryParams = new URLSearchParams({
      current_metric: selectedMetric,
      ...(pageSize ? { page_size: pageSize.toString() } : {}),
    });

    const url = `${baseUrl}/projects/${projectName}/similarities/${encodeURIComponent(
      id
    )}?${queryParams} `;
    const response = await fetch(url).then((res) => res.json());
    return z.string().array().parse(response);
  };

export const fetchProjectTags = async (
  baseUrl: string,
  projectName: string
) => {
  return GroupedTagsSchema.parse(
    await (await fetch(`${baseUrl}/projects/${projectName}/tags`)).json()
  );
};

export const defaultTags = { data: [], label: [] };

export const updateItemTags =
  (baseUrl: string, projectName: string) =>
  async (itemTags: { id: string; groupedTags: GroupedTags }[]) => {
    const url = `${baseUrl}/projects/${projectName}/item_tags`;
    const data = itemTags.map(({ id, groupedTags }) => ({
      id,
      grouped_tags: groupedTags,
    }));

    return fetch(url, {
      method: "put",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
  };

export const searchInProject =
  (baseUrl: string, projectName: string) =>
  async (
    { scope, query, type }: { scope: Scope; query: string; type: SearchType },
    signal?: AbortSignal
  ) => {
    const queryParams = new URLSearchParams({ query, scope, type });

    const response = await fetch(
      `${baseUrl}/projects/${projectName}/search?${queryParams}`,
      {
        signal,
      }
    );

    return searchResultSchema.parse(await response.json());
  };

export const useProjectQueries = () => {
  const projectContext = useContext(ProjectContext);

  if (!projectContext)
    throw new Error(
      "useProjectQueries has to be used within <ProjectContext.Provider>"
    );

  const queryClient = useQueryClient();
  const { projectName, baseUrl } = projectContext;

  return {
    itemTagsMutation: useMutation(
      ["updateItemTags"],
      (...args: Parameters<ReturnType<typeof updateItemTags>>) =>
        updateItemTags(baseUrl, projectName)(...args),
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
          ]) as Awaited<ReturnType<typeof fetchedTaggedItems>>;
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
      useQuery(["tags"], () => fetchProjectTags(baseUrl, projectName), {
        initialData: defaultTags,
      }),
    fetchTaggedItems: () =>
      useQuery(
        ["tagged_items"],
        () => fetchedTaggedItems(baseUrl, projectName),
        {
          initialData: new Map<string, GroupedTags>(),
        }
      ),
    fetchItem: (...args: Parameters<ReturnType<typeof fetchProjectItem>>) =>
      useQuery(["item", ...args], () =>
        fetchProjectItem(baseUrl, projectName)(...args)
      ),
    fetch2DEmbeddings: (
      selectedMetric: Parameters<ReturnType<typeof fetchProject2DEmbeddings>>[0]
    ) =>
      useQuery(
        ["2d_embeddings"],
        () => fetchProject2DEmbeddings(baseUrl, projectName)(selectedMetric),
        { enabled: !!selectedMetric }
      ),
    search: (...args: Parameters<ReturnType<typeof searchInProject>>) =>
      searchInProject(baseUrl, projectName)(...args),
  };
};

export const ProjectContext = createContext<{
  projectName: string;
  baseUrl: string;
  hasPremiumFeatures?: boolean;
} | null>(null);
