import { Button, Select } from "antd";
import { MdFilterAltOff } from "react-icons/md";
import { Dispatch, SetStateAction } from "react";
import {
  FilterState,
  MetricFilter,
  addNewEntry,
} from "../../util/MetricFilter";
import { FeatureHashMap } from "../../Types";
import { useProjectListCollaborators } from "../../../hooks/queries/useProjectListCollaborators";
import { useProjectListTags } from "../../../hooks/queries/useProjectListTags";
import { useProjectAnalysisSummary } from "../../../hooks/queries/useProjectAnalysisSummary";
import { ProjectDomainSummary } from "../../../openapi/api";

type Props = {
  projectHash: string;
  dataMetricsSummary: ProjectDomainSummary;
  annotationMetricsSummary: ProjectDomainSummary;
  annotationFilters: FilterState;
  setAnnotationFilters: Dispatch<SetStateAction<FilterState>>;
  dataFilters: FilterState;
  setDataFilters: Dispatch<SetStateAction<FilterState>>;
  canResetFilters: string | boolean | File;
  reset: () => void;
  featureHashMap: FeatureHashMap;
};
export function Filters({
  projectHash,
  dataMetricsSummary,
  annotationMetricsSummary,
  annotationFilters,
  setAnnotationFilters,
  canResetFilters,
  reset,
  featureHashMap,
  dataFilters,
  setDataFilters,
}: Props) {
  // Load metric ranges
  const { data: dataMetricRanges } = useProjectAnalysisSummary(
    projectHash,
    "data"
  );
  const { data: annotationMetricRanges } = useProjectAnalysisSummary(
    projectHash,
    "annotation"
  );

  // Load all collaborators & tags -> needed to support filters
  const { data: collaborators } = useProjectListCollaborators(projectHash);
  const { data: tags } = useProjectListTags(projectHash);

  return (
    <div className="flex w-full flex-col gap-2">
      <Select
        showSearch
        onChange={(metricKey) => {
          const type =
            metricKey.split("-")[0] === "data" ? "data" : "annotation";
          const key = metricKey.split("-")[1];
          setDataFilters(
            addNewEntry(
              type === "data" ? dataMetricsSummary : annotationMetricsSummary,
              type === "data"
                ? dataMetricRanges?.metrics
                : annotationMetricRanges?.metrics,
              featureHashMap,
              collaborators ?? [],
              tags ?? [],
              key
            )
          );
        }}
        placeholder="Add a new filter"
        options={[
          ...Object.entries(dataMetricsSummary.metrics).map(
            ([metricKey, metric]) => ({
              label: `D: ${metric?.title ?? metricKey}`,
              value: `data-${metricKey}`,
              type: "data",
            })
          ),
          ...Object.entries(annotationMetricsSummary.metrics).map(
            ([metricKey, metric]) => ({
              label: `A: ${metric?.title ?? metricKey}`,
              value: `annotation-${metricKey}`,
              type: "annotation",
            })
          ),
          ...Object.entries(dataMetricsSummary.enums).map(
            ([metricKey, metric]) => ({
              label: `D: ${metric?.title ?? metricKey}`,
              value: `data-${metricKey}`,
              type: "data",
            })
          ),
          ...Object.entries(annotationMetricsSummary.enums).map(
            ([metricKey, metric]) => ({
              label: `A: ${metric?.title ?? metricKey}`,
              value: `annotation-${metricKey}`,
              type: "annotation",
            })
          ),
        ]}
      />

      <MetricFilter
        filters={dataFilters}
        setFilters={setDataFilters}
        metricsSummary={dataMetricsSummary}
        metricRanges={dataMetricRanges?.metrics}
        featureHashMap={featureHashMap}
        projectHash={projectHash}
        tags={tags ?? []}
        collaborators={collaborators ?? []}
      />
      <MetricFilter
        filters={annotationFilters}
        setFilters={setAnnotationFilters}
        metricsSummary={annotationMetricsSummary}
        metricRanges={annotationMetricRanges?.metrics}
        featureHashMap={featureHashMap}
        projectHash={projectHash}
        tags={tags ?? []}
        collaborators={collaborators ?? []}
      />
      <Button
        disabled={!canResetFilters}
        onClick={() => reset()}
        icon={<MdFilterAltOff />}
      >
        Reset filters
      </Button>
    </div>
  );
}
