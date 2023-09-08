import { createContext, useContext } from "react";
import { ProjectsV2ApiFactory } from "../openapi/api";
import { Configuration } from "../openapi/configuration";

export class Querier {
  public readonly baseUrl: string;

  public readonly usesAuth: boolean;

  private readonly projectV2: ReturnType<typeof ProjectsV2ApiFactory>;

  constructor(baseUrl: string, token: string | null) {
    this.baseUrl = baseUrl;
    this.usesAuth = false;
    let configuration: Configuration | undefined;
    if (token) {
      this.usesAuth = true;
      configuration = new Configuration({
        baseOptions: {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      });
    }
    this.projectV2 = ProjectsV2ApiFactory(configuration, this.baseUrl);
  }

  getProjectV2API(): ReturnType<typeof ProjectsV2ApiFactory> {
    return this.projectV2;
  }
}

export const QuerierContext = createContext<Querier | null>(null);


export function useQuerier(): Querier {
  const querier = useContext(QuerierContext);

  if (!querier) {
    throw new Error("useQuerier has to be used within <QuerierContext.Provider>");
  }

  return querier
}
