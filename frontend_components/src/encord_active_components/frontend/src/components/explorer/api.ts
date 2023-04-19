import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext } from "react";
import { z } from "zod";

export const BASE_URL = "http://localhost:8000";

export const PointSchema = z.object({ x: z.number(), y: z.number() });

export const LabelRowObjectSchema = z.object({
  color: z.string(),
  confidence: z.number(),
  createdAt: z.string(),
  createdBy: z.string(),
  featureHash: z.string(),
  lastEditedAt: z.string(),
  lastEditedBy: z.string(),
  manualAnnotation: z.boolean(),
  name: z.string(),
  objectHash: z.string(),
  polygon: z.record(PointSchema),
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

export const fetchProject2DEmbeddings =
  (projectName: string) => async (selectedMetric: string) => {
    const url = `${BASE_URL}/projects/${projectName}/2d_embeddings/${selectedMetric}`;
    const response = await (await fetch(url)).json();
    return Item2DEmbeddingSchema.array().parse(response);
  };

export const fetchProjectMetrics =
  (projectName: string) => async (scope: Scope) => {
    const queryParams = new URLSearchParams({
      ...(scope ? { scope } : {}),
    });
    const url = `${BASE_URL}/projects/${projectName}/metrics?${queryParams}`;
    const response = await (await fetch(url)).json();
    return z.string().array().parse(response);
  };

export const fetchProjectItemIds =
  (projectName: string) => async (sortByMetric: string) => {
    const queryParams = new URLSearchParams({
      sort_by_metric: sortByMetric,
    });

    const url = `${BASE_URL}/projects/${projectName}/items_id_by_metric?${queryParams}`;
    const result = await (await fetch(url)).json();
    return IdValueSchema.array().parse(result);
  };

export const fetchProjectItem = (projectName: string) => async (id: string) => {
  const { url, ...item } = await (
    await fetch(`${BASE_URL}/projects/${projectName}/items/${id}`)
  ).json();
  return ItemSchema.parse({ ...item, url: `${BASE_URL}/${url}` }) as Item;
};

export const fetchSimilarItems =
  (projectName: string) =>
  async (id: string, selectedMetric: string, pageSize?: number) => {
    const queryParams = new URLSearchParams({
      current_metric: selectedMetric,
      ...(pageSize ? { page_size: pageSize.toString() } : {}),
    });

    const url = `${BASE_URL}/projects/${projectName}/similarities/${id}?${queryParams} `;
    const response = await fetch(url).then((res) => res.json());
    return z.string().array().parse(response);
  };

export const fetchProjectTags = async (projectName: string) => {
  return GroupedTagsSchema.parse(
    await (await fetch(`${BASE_URL}/projects/${projectName}/tags`)).json()
  );
};

export const updateItemTags =
  (projectName: string) =>
  async ({ id, groupedTags }: { id: string; groupedTags: GroupedTags }) => {
    const url = `${BASE_URL}/projects/${projectName}/item_tags`;
    const data = { id, grouped_tags: groupedTags };

    return fetch(url, {
      method: "put",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
  };

export const useProjectQueries = () => {
  const projectName = useContext(ProjectContext);
  const queryClient = useQueryClient();

  if (!projectName)
    throw new Error(
      "useProjectQueries has to be used within <ProjectContext.Provider>"
    );

  return {
    itemTagsMutation: useMutation(
      ["updateItemTags"],
      (...args: Parameters<ReturnType<typeof updateItemTags>>) =>
        updateItemTags(projectName)(...args),
      {
        onMutate: async (itemTags) => {
          const queryKey = ["item", itemTags.id];
          await queryClient.cancelQueries({ queryKey });

          const previousItem = queryClient.getQueryData(queryKey) as Item;

          queryClient.setQueryData(queryKey, {
            ...previousItem,
            tags: itemTags.groupedTags,
          });

          return { previousItem };
        },
        onError: (_, __, context) => {
          queryClient.setQueryData(["todos"], context?.previousItem);
        },
        onSettled: (_, __, variables) => {
          queryClient.invalidateQueries({ queryKey: ["items", variables.id] });
        },
      }
    ),
    fetchProjectTags: () =>
      useQuery(["tags"], () => fetchProjectTags(projectName), {
        initialData: { data: [], label: [] },
      }),
    fetchItem: (...args: Parameters<ReturnType<typeof fetchProjectItem>>) =>
      useQuery(["item", ...args], () => fetchProjectItem(projectName)(...args)),
    fetch2DEmbeddings: (
      selectedMetric: Parameters<ReturnType<typeof fetchProject2DEmbeddings>>[0]
    ) =>
      useQuery(
        ["2d_embeddings"],
        () => fetchProject2DEmbeddings(projectName!)(selectedMetric),
        { enabled: !!selectedMetric }
      ),
  };
};

export const ProjectContext = createContext<string | null>(null);
