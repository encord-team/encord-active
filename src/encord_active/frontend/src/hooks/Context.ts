import { createContext, useContext } from "react";
import { ProjectApiFactory, PredictionApiFactory } from "../openapi/api";
import { Configuration } from "../openapi/configuration";

export class Querier {
  public readonly baseUrl: string;

  public readonly usesAuth: boolean;

  private readonly project: ReturnType<typeof ProjectApiFactory>;

  private readonly prediction: ReturnType<typeof PredictionApiFactory>;

  constructor(baseUrl: string, token: string | null) {
    this.baseUrl = baseUrl;
    this.usesAuth = false;
    let configuration: Configuration | undefined;
    if (token) {
      this.usesAuth = true;
      configuration = new Configuration({
        accessToken: token,
      });
    }
    this.project = ProjectApiFactory(configuration, this.baseUrl);
    this.prediction = PredictionApiFactory(configuration, this.baseUrl);
  }

  getProjectAPI(): ReturnType<typeof ProjectApiFactory> {
    return this.project;
  }

  getPredictionAPI(): ReturnType<typeof PredictionApiFactory> {
    return this.prediction;
  }
}

export const QuerierContext = createContext<Querier | null>(null);

export function useQuerier(): Querier {
  const querier = useContext(QuerierContext);

  if (!querier) {
    throw new Error(
      "useQuerier has to be used within <QuerierContext.Provider>"
    );
  }

  return querier;
}
