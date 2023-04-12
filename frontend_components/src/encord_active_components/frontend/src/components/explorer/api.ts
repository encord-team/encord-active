import { createContext, useContext } from "react";
import { splitId } from "./id";
import { z } from "zod";

export const BASE_URL = "http://localhost:8000";

export const ItemSchema = z.object({
  url: z.string(),
  editUrl: z.string(),
  metadata: z.object({
    metrics: z.record(z.coerce.string()),
    annotator: z.string().nullish(),
    labelClass: z.string().nullish(),
  }),
  tags: z.object({ data: z.string().array(), label: z.string().array() }),
});

export type Item = z.infer<typeof ItemSchema>;
export type ItemMetadata = Item["metadata"];
export type GroupedTags = Item["tags"];

export type EmbeddingType = "image" | "object" | "classification";

export const fetchProjectItem = (projectName: string) => async (id: string) => {
  const { labelRow, data } = splitId(id);
  const queryParams = new URLSearchParams({
    full_id: id,
  });

  const response = await fetch(
    `${BASE_URL}/projects/${projectName}/label_rows/${labelRow}/data_units/${data}?${queryParams}`
  ).then((res) => res.json());
  return ItemSchema.parse(response);
};

export const getSimilarItems =
  (projectName: string) =>
  async (id: string, embeddingType: EmbeddingType, pageSize: number) => {
    const queryParams = new URLSearchParams({
      embedding_type: embeddingType,
      page_size: pageSize.toString(),
    });

    const response = await fetch(
      `${BASE_URL}/projects/${projectName}/similarities/${id}?${queryParams}`
    ).then((res) => res.json());
    return ItemSchema.parse(response);
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
