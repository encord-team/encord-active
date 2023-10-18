import { Button, Select } from "antd";
import { MdFilterAltOff } from "react-icons/md";
import { Dispatch, SetStateAction } from "react";
import { PlusOutlined } from "@ant-design/icons";
import {
  FilterState,
  MetricFilter,
  addNewEntry,
} from "../../util/MetricFilter";
import { FeatureHashMap } from "../../Types";
import { useProjectListCollaborators } from "../../../hooks/queries/useProjectListCollaborators";
import { useProjectListTags } from "../../../hooks/queries/useProjectListTags";
import {
  AnalysisDomain,
  ProjectDomainSummary,
  QuerySummary,
} from "../../../openapi/api";

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
  dataMetricRanges: QuerySummary["metrics"] | undefined;
  annotationMetricRanges: QuerySummary["metrics"] | undefined;
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
  dataMetricRanges,
  annotationMetricRanges,
}: Props) {
  // Load all collaborators & tags -> needed to support filters
  const { data: collaborators } = useProjectListCollaborators(projectHash);
  const { data: tags } = useProjectListTags(projectHash);

  return (
    <div className="flex w-full flex-col gap-2">
      <Button onClick={() => reset()} icon={<MdFilterAltOff />}>
        Reset filters
      </Button>
      <Select
        suffixIcon={<PlusOutlined />}
        value="Add a new filter"
        className="w-full"
        showSearch
        onChange={(metricKey) => {
          const type =
            metricKey.split("-")[0] === "data" ? "data" : "annotation";
          const key = metricKey.split("-")[1];
          if (type === "data") {
            setDataFilters(
              addNewEntry(
                dataMetricsSummary,
                dataMetricRanges,
                featureHashMap,
                collaborators ?? [],
                tags ?? [],
                key
              )
            );
          } else if (type === "annotation") {
            setAnnotationFilters(
              addNewEntry(
                annotationMetricsSummary,
                annotationMetricRanges,
                featureHashMap,
                collaborators ?? [],
                tags ?? [],
                key
              )
            );
          }
        }}
        options={[
          ...Object.entries(dataMetricsSummary.metrics).map(
            ([metricKey, metric]) => ({
              label: `D: ${metric?.title ?? metricKey}`,
              value: `data-${metricKey}`,
              type: "data",
            })
          ),
          ...Object.entries(dataMetricsSummary.enums).map(
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
        metricRanges={dataMetricRanges}
        featureHashMap={featureHashMap}
        projectHash={projectHash}
        tags={tags ?? []}
        collaborators={collaborators ?? []}
        analysisDomain={AnalysisDomain.Data}
      />
      <MetricFilter
        filters={annotationFilters}
        setFilters={setAnnotationFilters}
        metricsSummary={annotationMetricsSummary}
        metricRanges={annotationMetricRanges}
        featureHashMap={featureHashMap}
        projectHash={projectHash}
        tags={tags ?? []}
        collaborators={collaborators ?? []}
        analysisDomain={AnalysisDomain.Annotation}
      />
    </div>
  );
}
