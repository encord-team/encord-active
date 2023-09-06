import { ProjectsV2ApiFactory } from "../openapi/api";
import { Configuration } from "../openapi/configuration";

export class QueryContext {
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
