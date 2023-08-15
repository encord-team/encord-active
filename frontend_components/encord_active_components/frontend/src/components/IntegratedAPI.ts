import { useMemo } from "react";
import {
  useMutation,
  UseMutationOptions,
  UseMutationResult,
  useQuery,
  useQueryClient,
  UseQueryOptions,
  UseQueryResult,
} from "@tanstack/react-query";
import axios, { AxiosError } from "axios";
import {
  CreateSubsetMutationArguments,
  CreateTagMutationArguments,
  PaginationResult,
  PredictionView,
  ProjectAnalysisCompareMetricDissimilarity,
  ProjectAnalysisDistribution,
  ProjectAnalysisDomain,
  ProjectAnalysisScatter,
  ProjectAnalysisSummary,
  ProjectEmbeddingReductions,
  ProjectItemDetailedSummary,
  ProjectMetricPerformance,
  ProjectPredictionSummary,
  ProjectPreviewItemResult,
  ProjectSearchResult,
  ProjectSimilarityResult,
  ProjectSummary,
  ProjectView,
  QueryAPI,
  SearchFilters,
  UploadToEncordMutationArguments,
} from "./Types";
import { apiUrl } from "../constants";

export type IntegratedProjectMetadata = {
  readonly title: string;
  readonly description: string;
  readonly projectHash: string;
  readonly baseProjectUrl: string;
  readonly imageUrl: string;
  readonly imageUrlTimestamp: null | number;
  readonly downloaded: boolean;
  readonly sandbox: boolean;
  readonly stats: {
    readonly dataUnits: number;
    readonly labels: number;
    readonly classes: number;
  };
};

const LONG_RUNNING_TIMEOUT: number = 1000 * 60 * 60; // 1 hour timeout!!

const SummaryQueryOptions: Pick<UseQueryOptions, "staleTime" | "cacheTime"> = {
  staleTime: 1000 * 60 * 10, // 10 minutes
  cacheTime: 1000 * 60 * 10, // 10 minutes
};

const StatisticQueryOptions: Pick<UseQueryOptions, "staleTime" | "cacheTime"> =
{
  staleTime: 1000 * 60 * 5, // 5 minutes
  cacheTime: 1000 * 60 * 5, // 5 minutes
};

class IntegratedAPI implements QueryAPI {
  private readonly projects: Readonly<
    Record<string, IntegratedProjectMetadata>
  >;

  constructor(
    token: string | null,
    projects: Readonly<Record<string, IntegratedProjectMetadata>>,
  ) {
    this.projects = projects;
    if (token)
      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  }

  private getBaseUrl(
    project_hash: string,
    enabled: boolean | undefined,
  ): string {
    if (enabled === false) {
      return `${apiUrl}/projects_v2_error_state/query_disabled`;
    }
    const base = this.projects[project_hash];
    if (base == null) {
      return `${apiUrl}/projects_v2_error_state/project_does_not_exist`;
    }
    return base.baseProjectUrl;
  }

  /// === Shared API Implementation

  // === Project Summary ===
  useListProjectViews = (
    search: string,
    offset: number,
    limit: number,
  ): UseQueryResult<PaginationResult<ProjectView>> => {
    const { projects } = this;
    const [results, total] = useMemo(() => {
      const results = Object.values(projects)
        .filter((project) => project.downloaded)
        .map((project) => ({
          project_hash: project.projectHash,
          title: project.title,
          description: project.description,
        }));
      results.sort((a, b) => (a.title < b.title ? -1 : 1));

      return [results.slice(offset, offset + limit), results.length];
    }, [projects, offset, limit]);
    return useQuery(
      [this.projects, "ACTIVE:useListProjectViews", results, total],
      () => ({
        total,
        results,
      }),
    );
  };

  useProjectSummary = (
    projectHash: string,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectSummary", projectHash],
      // eslint-disable-next-line
      () => axios.get(`${baseURL}/summary`).then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions },
    );
  };

  useProjectListEmbeddingReductions = (
    projectHash: string,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectEmbeddingReductions> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectListEmbeddingReductions", projectHash],
      // eslint-disable-next-line
      () => axios.get(`${baseURL}/reductions`).then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions },
    );
  };

  useProjectAnalysisSummary = (
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectAnalysisSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectAnalysisSummary", projectHash, analysisDomain],
      () =>
        axios
          .get(`${baseURL}/analysis/${analysisDomain}/summary`)
          // eslint-disable-next-line
          .then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions },
    );
  };

  useProjectAnalysisMetricScatter = (
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    xMetric: string,
    yMetric: string,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectAnalysisScatter> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      [
        "ACTIVE:useProjectAnalysisMetricScatter",
        projectHash,
        analysisDomain,
        xMetric,
        yMetric,
      ],
      () =>
        axios
          .get(`${baseURL}/analysis/${analysisDomain}/scatter`, {
            params: {
              x_metric: xMetric,
              y_metric: yMetric,
            },
          })
          // eslint-disable-next-line
          .then((res) => res.data as any),
      { ...options, ...StatisticQueryOptions },
    );
  };

  useProjectAnalysisDistribution = (
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    metricOrEnum: string,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectAnalysisDistribution> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      [
        "ACTIVE:useProjectAnalysisDistribution",
        projectHash,
        analysisDomain,
        metricOrEnum,
      ],
      () =>
        axios
          .get(`${baseURL}/analysis/${analysisDomain}/distribution`, {
            params: {
              group: metricOrEnum,
            },
          })
          // eslint-disable-next-line
          .then((res) => res.data as any),
      { ...options, ...StatisticQueryOptions },
    );
  };

  useProjectAnalysisCompareMetricDissimilarity = (
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    compareProjectHash: string,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectAnalysisCompareMetricDissimilarity> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      [
        "ACTIVE:useProjectAnalysisCompareMetricDissimilarity",
        projectHash,
        analysisDomain,
        compareProjectHash,
      ],
      () =>
        axios
          .get(
            `${baseURL}/analysis/${analysisDomain}/project_compare/metric_dissimilarity`,
            {
              params: {
                compare_project_hash: compareProjectHash,
              },
            },
          )
          // eslint-disable-next-line
          .then((res) => res.data as any),
      {
        ...options,
        staleTime: 15 * 60 * 1000, // 15 minutes, do not refetch.
      },
    );
  };

  useProjectAnalysisSearch = (
    projectHash: string,
    analysisDomain: ProjectAnalysisDomain,
    filters: SearchFilters,
    orderBy: null | string,
    desc: boolean,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectSearchResult> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      [
        "ACTIVE:useProjectAnalysisSearch",
        projectHash,
        analysisDomain,
        filters,
        orderBy,
        desc,
      ],
      () =>
        axios
          .get(`${baseURL}/analysis/${analysisDomain}/search`, {
            params: {
              filters,
              order_by: orderBy,
              desc,
            },
          })
          // eslint-disable-next-line
          .then((res) => res.data as any),
      {
        ...options,
        staleTime: 15 * 60 * 1000, // 15 minutes, do not refetch.
      },
    );
  };

  // == Project visualisation ===
  useProjectItemPreview = (
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash?: string | undefined | null,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectPreviewItemResult> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectItemPreview", projectHash, duHash, frame, objectHash],
      () =>
        axios
          .get(`${baseURL}/preview/${duHash}/${frame}/${objectHash ?? ""}`)
          // eslint-disable-next-line
          .then((res) => res.data as any),
      {
        ...options,
        staleTime: 1000 * 60 * 5, // 5 minutes
        cacheTime: 1000 * 60 * 5, // 5 minutes
      },
    );
  };

  useProjectItemDetails = (
    projectHash: string,
    duHash: string,
    frame: number,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectItemDetailedSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectItemDetails", projectHash, duHash, frame],
      () =>
        axios
          .get(`${baseURL}/item/${duHash}/${frame}/`)
          // eslint-disable-next-line
          .then((res) => res.data as any),
      options,
    );
  };

  useProjectItemSimilarity = (
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash: string | undefined | null = null,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectSimilarityResult> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      [
        "ACTIVE:useProjectItemSimilarity",
        projectHash,
        duHash,
        frame,
        objectHash,
      ],
      () =>
        axios
          .get(
            `${baseURL}/analysis/${objectHash == null ? "data" : "annotations"
            }/similarity/${duHash}/${frame}/${objectHash == null ? "" : objectHash
            }?embedding=embedding_clip`,
          )
          // eslint-disable-next-line
          .then((res) => res.data as any),
      options,
    );
  };

  // == Project actions ===
  useProjectMutationCreateSubset = (
    projectHash: string,
    options: Pick<
      UseMutationOptions<string, unknown, CreateSubsetMutationArguments>,
      "onError" | "onSuccess" | "onSettled"
    > = {},
  ): UseMutationResult<string, unknown, CreateSubsetMutationArguments> => {
    const baseURL = this.getBaseUrl(projectHash, true).replace(
      "/projects_v2/",
      "/projects/",
    );
    const queryClient = useQueryClient();
    return useMutation(
      ["ACTIVE:useProjectMutationCreateTag", projectHash],
      async (args: CreateSubsetMutationArguments) => {
        const params = {
          filters: args.filters ?? {},
          ids: args.ids ?? [],
          project_title: args.project_title,
          project_description: args.project_description ?? "",
          dataset_title: args.dataset_title,
          dataset_description: args.dataset_description ?? "",
        };
        const headers = {
          Accept: "application/json",
          "Content-Type": "application/json",
        };
        const r = await axios.post(
          `${baseURL}/create_subset`,
          JSON.stringify(params),
          { headers, timeout: LONG_RUNNING_TIMEOUT },
        );
        await queryClient.invalidateQueries({
          queryKey: ["IntegratedAPI:useLookupProjectsFromUrlList"],
        });
        return r.data;
      },
      options,
    );
  };

  useProjectMutationCreateTag = (
    projectHash: string,
    options: Pick<
      UseMutationOptions<string, unknown, CreateTagMutationArguments>,
      "onError" | "onSuccess" | "onSettled"
    > = {},
  ): UseMutationResult<string, unknown, CreateTagMutationArguments> =>
    useMutation(
      ["ACTIVE:useProjectMutationCreateTag", projectHash],
      async (args: CreateTagMutationArguments) => "FIXME: impl",
      options,
    );

  useProjectMutationUploadToEncord = (
    projectHash: string,
    options?: Pick<
      UseMutationOptions<string, unknown, UploadToEncordMutationArguments>,
      "onError" | "onSuccess" | "onSettled"
    >,
  ): UseMutationResult<string, unknown, UploadToEncordMutationArguments> => {
    const baseURL = this.getBaseUrl(projectHash, true).replace(
      "/projects_v2/",
      "/projects/",
    );
    const queryClient = useQueryClient();
    return useMutation(
      ["ACTIVE:useProjectMutationUploadToEncord", projectHash],
      async (args: UploadToEncordMutationArguments) => {
        const params = {
          project_title: args.project_title,
          project_description: args.project_description ?? "",
          dataset_title: args.dataset_title,
          dataset_description: args.dataset_description ?? "",
          ontology_title: args.ontology_title,
          ontology_description: args.ontology_description ?? "",
        };
        const headers = {
          Accept: "application/json",
          "Content-Type": "application/json",
        };
        const r = await axios.post(
          `${baseURL}/upload_to_encord`,
          JSON.stringify(params),
          { headers, timeout: LONG_RUNNING_TIMEOUT },
        );
        await queryClient.invalidateQueries({
          queryKey: ["IntegratedAPI:useLookupProjectsFromUrlList"],
        });
        return r.data.project_hash;
      },
      { ...options },
    );
  };

  // == Project predictions ===
  useProjectListPredictions = (
    projectHash: string,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<PaginationResult<PredictionView>> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectListPredictions", projectHash],
      () =>
        // eslint-disable-next-line
        axios.get(`${baseURL}/predictions`).then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions },
    );
  };

  useProjectPredictionSummary = (
    projectHash: string,
    predictionHash: string,
    iou: number,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectPredictionSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectPredictionSummary", projectHash, predictionHash, iou],
      () =>
        axios
          .get(`${baseURL}/predictions/${predictionHash}/summary`, {
            params: { iou },
          }) // eslint-disable-next-line
          .then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions },
    );
  };

  useProjectPredictionMetricPerformance = (
    projectHash: string,
    predictionHash: string,
    buckets: number,
    iou: number,
    metric: string,
    options: Pick<UseQueryOptions, "enabled"> = {},
  ): UseQueryResult<ProjectMetricPerformance> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      [
        "ACTIVE:useProjectPredictionMetricPerformance",
        projectHash,
        predictionHash,
        buckets,
        iou,
        metric,
      ],
      () =>
        axios
          .get(`${baseURL}/predictions/${predictionHash}/metric_performance`, {
            params: { iou, metric_name: metric, buckets },
          }) // eslint-disable-next-line
          .then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions },
    );
  };
}

export function useProjectsList(): UseQueryResult<
  Readonly<Record<string, IntegratedProjectMetadata>>,
  AxiosError<{ details: string }>
> {
  return useQuery(
    ["IntegratedAPI:useLookupProjectsFromUrlList", apiUrl],
    async () => {
      const allData: Record<string, IntegratedProjectMetadata> = {};
      // eslint-disable-next-line no-await-in-loop
      const res = await axios.get(`${apiUrl}/projects_v2`);
      // eslint-disable-next-line
      const data: Record<
        string,
        Omit<IntegratedProjectMetadata, "baseProjectUrl">
      > = res.data as any;
      Object.entries(data).forEach(([projectHash, projectMeta]) => {
        allData[projectHash] = {
          ...projectMeta,
          imageUrl: projectMeta.imageUrl.startsWith("http")
            ? projectMeta.imageUrl
            : `${apiUrl}${projectMeta.imageUrl}`,
          baseProjectUrl: `${apiUrl}/projects_v2/${projectHash}`,
        };
      });
      return allData;
    },
    {
      retry: 0,
      // 30 minutes
      staleTime: 1000 * 60 * 30,
      cacheTime: 1000 * 60 * 30,
    },
  );
}

export function useIntegratedAPI(
  token: string | null,
  projects: Readonly<Record<string, IntegratedProjectMetadata>>,
): QueryAPI {
  return useMemo(() => new IntegratedAPI(token, projects), [token, projects]);
}
