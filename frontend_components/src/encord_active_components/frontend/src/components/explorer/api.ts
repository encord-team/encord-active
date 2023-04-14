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

export const ItemSchema = z.object({
  id: z.string(),
  url: z.string(),
  editUrl: z.string(),
  metadata: z.object({
    metrics: z.record(z.coerce.string()),
    annotator: z.string().nullish(),
    labelClass: z.string().nullish(),
  }),
  tags: z.object({ data: z.string().array(), label: z.string().array() }),
  labels: LabelsSchema,
});

export type Item = z.infer<typeof ItemSchema>;
export type ItemMetadata = Item["metadata"];
export type GroupedTags = Item["tags"];
export type Point = z.infer<typeof PointSchema>;
export type EmbeddingType = "image" | "object" | "classification";
export type Scope = "data_quality" | "label_quality" | "model_quality";

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

    const url = `${BASE_URL}/projects/${projectName}/item_ids?${queryParams}`;
    const result = await (await fetch(url)).json();
    return z.string().array().parse(result);
  };

export const fetchProjectItem = (projectName: string) => async (id: string) => {
  const url = `${BASE_URL}/projects/${projectName}/items/${id}`;
  const response = await fetch(url).then((res) => res.json());
  return ItemSchema.parse(response);
};

export const getSimilarItems =
  (projectName: string) =>
  async (id: string, embeddingType: EmbeddingType, pageSize?: number) => {
    const queryParams = new URLSearchParams({
      embedding_type: embeddingType,
      ...(pageSize ? { page_size: pageSize.toString() } : {}),
    });

    const url = `${BASE_URL} /projects/${projectName} /similarities/${id}?${queryParams} `;
    const response = await fetch(url).then((res) => res.json());
    return z.string().array().parse(response);
  };

export const useProjectQueries = () => {
  const projectName = useContext(ProjectContext);

  if (!projectName)
    throw new Error(
      "useProjectQueries has to be used within <ProjectContext.Provider>"
    );

  return {
    fetchItem: fetchProjectItem(projectName!),
    getSimilarItems: getSimilarItems(projectName!),
  };
};

export const ProjectContext = createContext<string | null>(null);
