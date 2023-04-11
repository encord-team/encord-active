import { createContext, useContext } from "react";
import { splitId } from "./id";
import { z } from "zod";

const BASE_URL = "http://localhost:8000";

export const ItemResponseSchema = z.object({
  url: z.string(),
});

export type ItemResponse = z.infer<typeof ItemResponseSchema>;

export const fetchProjectItem = (projectName: string) => async (id: string) => {
  const { labelRow, data } = splitId(id);
  const response = await fetch(
    `${BASE_URL}/projects/${projectName}/label_rows/${labelRow}/data_units/${data}`
  ).then((res) => res.json());
  return ItemResponseSchema.parse(response);
};

export const useProjectQueries = () => {
  const projectName = useContext(ProjectContext);

  if (!projectName)
    throw new Error(
      "useProjectQueries has to be used within <ProjectContext.Provider>"
    );

  return {
    fetchItem: fetchProjectItem(projectName!),
  };
};

export const ProjectContext = createContext<string | null>(null);
