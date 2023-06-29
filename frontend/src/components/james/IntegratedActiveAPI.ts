import { useMemo } from "react";
import {
  useQuery,
  UseQueryOptions,
  UseQueryResult,
} from "@tanstack/react-query";
import axios from "axios";
import {
  ActivePaginationResult,
  ActivePredictionView,
  ActiveProjectAnalysisDistribution,
  ActiveProjectAnalysisDomain,
  ActiveProjectAnalysisScatter,
  ActiveProjectAnalysisSummary,
  ActiveProjectItemDetailedSummary,
  ActiveProjectMetricPerformance,
  ActiveProjectPredictionSummary,
  ActiveProjectPreviewItemResult,
  ActiveProjectSearchResult,
  ActiveProjectSimilarityResult,
  ActiveProjectSummary,
  ActiveProjectView,
  ActiveQueryAPI,
} from "./oss/ActiveTypes";

export type IntegratedProjectMetadata = {
  readonly title: string;
  readonly description: string;
  readonly projectHash: string;
  readonly baseProjectUrl: string;
  readonly sandbox: boolean;
  readonly downloaded: boolean;
  readonly imageUrl: string;
  readonly stats: {
    readonly dataUnits: number;
    readonly labels: number;
    readonly classes: number;
  };
};

const SummaryQueryOptions: Pick<UseQueryOptions, "staleTime" | "cacheTime"> = {
  staleTime: 1000 * 60 * 10, // 10 minutes
  cacheTime: 1000 * 60 * 10, // 10 minutes
};

const StatisticQueryOptions: Pick<UseQueryOptions, "staleTime" | "cacheTime"> =
  {
    staleTime: 1000 * 60 * 5, // 5 minutes
    cacheTime: 1000 * 60 * 5, // 5 minutes
  };

class IntegratedActiveAPI implements ActiveQueryAPI {
  private readonly projects: Readonly<
    Record<string, IntegratedProjectMetadata>
  >;

  constructor(projects: Readonly<Record<string, IntegratedProjectMetadata>>) {
    this.projects = projects;
  }

  private getBaseUrl(
    project_hash: string,
    enabled: boolean | undefined
  ): string {
    if (enabled === false) {
      return "/ACTIVE:DISABLED";
    }
    const base = this.projects[project_hash];
    if (base == null) {
      throw Error(`Active - cannot find project with hash: "${project_hash}"`);
    }
    return base.baseProjectUrl;
  }

  /// === Shared API Implementation

  // === Project Summary ===
  useListProjectViews = (
    search?: string,
    offset?: number,
    limit?: number
  ): UseQueryResult<ActivePaginationResult<ActiveProjectView>> => {
    const { projects } = this;
    const [results, total] = useMemo(() => {
      const results = Object.values(projects).filter(
        ({ downloaded }) => downloaded
      );
      results.sort((a, b) => (a.title < b.title ? -1 : 1));
      return [
        offset == null || limit == null
          ? results
          : results.slice(offset, offset + limit),
        results.length,
      ];
    }, [projects, offset, limit]);
    return useQuery(
      [this.projects, "ACTIVE:useListProjectViews", results, total],
      () => ({
        total,
        results,
      })
    );
  };

  useProjectSummary = (
    projectHash: string,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectSummary", projectHash],
      () => axios.get(`${baseURL}/summary`).then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions }
    );
  };

  useProjectAnalysisSummary = (
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectAnalysisSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectAnalysisSummary", projectHash, analysisDomain],
      () =>
        axios
          .get(`${baseURL}/analysis/${analysisDomain}/summary`)
          .then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions }
    );
  };

  useProjectAnalysisMetricScatter = (
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    xMetric: string,
    yMetric: string,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectAnalysisScatter> => {
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
          .then((res) => res.data as any),
      { ...options, ...StatisticQueryOptions }
    );
  };

  useProjectAnalysisDistribution = (
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    metricOrEnum: string,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectAnalysisDistribution> => {
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
          .get(`${baseURL}/analysis/${analysisDomain}/dist`, {
            params: {
              group: metricOrEnum,
            },
          })
          .then((res) => res.data as any),
      { ...options, ...StatisticQueryOptions }
    );
  };

  useProjectAnalysisSearch = (
    projectHash: string,
    analysisDomain: ActiveProjectAnalysisDomain,
    metricFilters: null | Readonly<Record<string, [number, number]>>,
    metricOutliers: null | Readonly<Record<string, "warning" | "severe">>,
    enumFilters: null | Readonly<Record<string, ReadonlyArray<string>>>,
    orderBy: null | string,
    desc: boolean,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectSearchResult> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      [
        "ACTIVE:useProjectAnalysisSearch",
        projectHash,
        analysisDomain,
        metricFilters,
        metricOutliers,
        enumFilters,
        orderBy,
        desc,
      ],
      () =>
        axios
          .get(`${baseURL}/analysis/${analysisDomain}/search`, {
            params: {
              metric_filters: metricFilters,
              metric_outliers: metricOutliers,
              enum_filters: enumFilters,
              order_by: orderBy,
              desc,
            },
          })
          .then((res) => res.data as any),
      {
        ...options,
        staleTime: 15 * 60 * 1000, // 15 minutes, do not refetch.
      }
    );
  };

  // == Project visualisation ===
  useProjectItemPreview = (
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash?: string | undefined | null,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectPreviewItemResult> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectItemPreview", projectHash, duHash, frame, objectHash],
      () =>
        axios
          .get(`${baseURL}/preview/${duHash}/${frame}/${objectHash ?? ""}`)
          .then((res) => res.data as any),
      {
        ...options,
        staleTime: 1000 * 60 * 5, // 5 minutes
        cacheTime: 1000 * 60 * 5, // 5 minutes
      }
    );
  };

  useProjectItemDetails = (
    projectHash: string,
    duHash: string,
    frame: number,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectItemDetailedSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectItemDetails", projectHash, duHash, frame],
      () =>
        axios
          .get(`${baseURL}/item/${duHash}/${frame}/`)
          .then((res) => res.data as any),
      options
    );
  };

  useProjectItemSimilarity = (
    projectHash: string,
    duHash: string,
    frame: number,
    objectHash: string | undefined | null = null,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectSimilarityResult> => {
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
            `${baseURL}/analysis/${
              objectHash == null ? "data" : "annotations"
            }/similarity/${duHash}/${frame}/${
              objectHash == null ? "" : objectHash
            }?embedding=embedding_clip`
          )
          .then((res) => res.data as any),
      options
    );
  };

  // == Project predictions ===
  useProjectListPredictions = (
    projectHash: string,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActivePaginationResult<ActivePredictionView>> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectListPredictions", projectHash],
      () =>
        axios.get(`${baseURL}/predictions/list`).then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions }
    );
  };

  useProjectPredictionSummary = (
    projectHash: string,
    predictionHash: string,
    iou: number,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectPredictionSummary> => {
    const baseURL = this.getBaseUrl(projectHash, options.enabled);
    return useQuery(
      ["ACTIVE:useProjectPredictionSummary", projectHash, predictionHash, iou],
      () =>
        axios
          .get(`${baseURL}/predictions/get/${predictionHash}/summary`, {
            params: { iou },
          })
          .then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions }
    );
  };

  useProjectPredictionMetricPerformance = (
    projectHash: string,
    predictionHash: string,
    buckets: number,
    iou: number,
    metric: string,
    options: Pick<UseQueryOptions, "enabled"> = {}
  ): UseQueryResult<ActiveProjectMetricPerformance> => {
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
          .get(
            `${baseURL}/predictions/get/${predictionHash}/metric_performance`,
            {
              params: { iou, buckets, metric_name: metric },
            }
          )
          .then((res) => res.data as any),
      { ...options, ...SummaryQueryOptions }
    );
  };
}

export function useLookupProjectsFromUrlList(
  urls: string[]
): UseQueryResult<Readonly<Record<string, IntegratedProjectMetadata>>> {
  return useQuery(
    ["IntegratedActiveAPI:useLookupProjectsFromUrlList", urls],
    async () => {
      const allData: Record<string, IntegratedProjectMetadata> = {};
      for (const url of urls) {
        // eslint-disable-next-line no-await-in-loop
        const res = await axios.get(`${url}/projects_v2/list`);
        const data: Record<
          string,
          Omit<IntegratedProjectMetadata, "baseProjectUrl" | "projectHash">
        > = res.data as any;

        Object.entries(data).forEach(([projectHash, projectMeta]) => {
          allData[projectHash] = {
            ...projectMeta,
            projectHash,
            imageUrl: projectMeta.imageUrl.startsWith("http")
              ? projectMeta.imageUrl
              : `${url}/${projectMeta.imageUrl}`,
            baseProjectUrl: `${url}/projects_v2/get/${projectHash}`,
          };
        });
      }
      return allData;
    },
    {
      // 30 minutes
      staleTime: 1000 * 60 * 30,
      cacheTime: 1000 * 60 * 30,
    }
  );
}

export function useIntegratedActiveAPI(
  projects: Readonly<Record<string, IntegratedProjectMetadata>>
): ActiveQueryAPI {
  return useMemo(() => new IntegratedActiveAPI(projects), [projects]);
}
