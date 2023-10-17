import { useEffect, useMemo, useState } from "react";
import { Badge, Spin, Tabs } from "antd";
import { useNavigate, useParams } from "react-router";
import {
  FeatureHashMap,
  ModalName,
  OntologyObjectAttribute,
  OntologyObjectAttributeOptions,
  ProjectOntology,
} from "./Types";
import { ProjectSelector } from "./ProjectSelector";
import { Explorer } from "./explorer";
import { SummaryView } from "./tabs/AnalyticsView";
import { loadingIndicator } from "./Spin";
import { useProjectSummary } from "../hooks/queries/useProjectSummary";
import { useProjectHash } from "../hooks/useProjectHash";
import { Colors } from "../constants";
import { PredictionsTab } from "./tabs/predictions/PredictionsTab";
import { Collections } from "./tabs/collections/Collections";
import { DefaultFilters, FilterState } from "./util/MetricFilter";
import { useProjectListTagsMeta } from "../hooks/queries/useProjectListTagsMeta";
import { env } from "../constants";
import { classy } from "../helpers/classy";
import { CustomTooltip } from "./util/CustomTooltip";

export function ProjectPage(props: {
  encordDomain: string;
  setSelectedProjectHash: (projectHash?: string) => void;
}) {
  const projectHash = useProjectHash();
  const { setSelectedProjectHash, encordDomain } = props;

  const navigate = useNavigate();
  const { tab } = useParams();

  if (!tab) {
    throw Error("Missing `tab` path parameter");
  }

  const { data: projectTagsMeta = [] } = useProjectListTagsMeta(projectHash);

  const { data: projectSummary, isError } = useProjectSummary(projectHash);

  const editUrl =
    projectSummary === undefined || projectSummary.local_project
      ? undefined
      : (dataHash: string, projectHash: string, frame: number): string =>
        `${encordDomain}/label_editor/${dataHash}&${projectHash}/${frame}`;

  // Go to parent in the error case (project does not exist).
  useEffect(() => {
    if (isError) {
      setSelectedProjectHash(undefined);
    }
  }, [isError, setSelectedProjectHash]);

  // selection
  const [selectedItems, setSelectedItems] = useState<
    ReadonlySet<string> | "ALL"
  >(new Set<string>());

  const hasSelectedItems = selectedItems === "ALL" || selectedItems.size > 0;

  const featureHashMap: FeatureHashMap = useMemo(() => {
    const featureHashMap: Record<
      string,
      { readonly color: string; readonly name: string }
    > = {};
    if (projectSummary == null) {
      return {};
    }
    const procAttribute = (
      a: OntologyObjectAttribute | OntologyObjectAttributeOptions,
      color: string
    ) => {
      if ("name" in a) {
        featureHashMap[a.featureNodeHash] = {
          color,
          name: a.name ?? a.featureNodeHash,
        };
        if (a.type === "checklist" || a.type === "radio") {
          a.options.forEach((o) => procAttribute(o, color));
        }
      } else {
        featureHashMap[a.featureNodeHash] = {
          color,
          name: a.label ?? a.featureNodeHash,
        };
        if (a.options != null) {
          a.options.forEach((o) => procAttribute(o, color));
        }
      }
    };
    const ontology: ProjectOntology =
      projectSummary.ontology as ProjectOntology;
    ontology.objects.forEach((o) => {
      featureHashMap[o.featureNodeHash] = {
        color: o.color,
        name: o.name ?? o.featureNodeHash,
      };
      if (o.attributes != null) {
        o.attributes.forEach((a) => procAttribute(a, o.color));
      }
    });
    ontology.classifications.forEach((o) => {
      featureHashMap[o.featureNodeHash] = {
        color: o.color,
        name: o.name ?? o.featureNodeHash,
      };
      if (o.attributes != null) {
        o.attributes.forEach((a) => procAttribute(a, o.color));
      }
    });
    return featureHashMap;
  }, [projectSummary]);

  // Modal state
  const [openModal, setOpenModal] = useState<ModalName | undefined>();

  const [dataFilters, setDataFilters] = useState<FilterState>(DefaultFilters);
  // Loading screen while waiting for full summary of project metrics.
  if (projectSummary == null) {
    return <Spin indicator={loadingIndicator} />;
  }

  const remoteProject = !projectSummary.local_project;

  return (
    <Tabs
      className="h-full"
      size="large"
      tabBarStyle={{
        background: "#FAFAFA",
        margin: 0,
        padding: "0px 10px",
      }}
      centered
      tabBarExtraContent={{
        left: (
          <ProjectSelector
            selectedProjectHash={projectHash}
            setSelectedProjectHash={setSelectedProjectHash}
          />
        ),
      }}
      items={[
        {
          label: "Explorer",
          key: "explorer",
          children: (
            <Explorer
              projectHash={projectHash}
              predictionHash={undefined}
              dataMetricsSummary={projectSummary.data}
              annotationMetricsSummary={projectSummary.annotation}
              editUrl={editUrl}
              featureHashMap={featureHashMap}
              setSelectedProjectHash={setSelectedProjectHash}
              remoteProject={remoteProject}
              openModal={openModal}
              setOpenModal={setOpenModal}
              selectedItems={selectedItems}
              setSelectedItems={setSelectedItems}
              hasSelectedItems={hasSelectedItems}
              dataFilters={dataFilters}
              setDataFilters={setDataFilters}
            />
          ),
        },
        {
          label: "Analytics",
          key: "analytics",
          children: (
            <SummaryView
              projectHash={projectHash}
              projectSummary={projectSummary}
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: (
            <CustomTooltip title="Only available on the hoseted version">
              <div>Model Evaluation</div>
            </CustomTooltip>),
          key: "predictions",
          disabled: true
        },
        {
          label: (
            <div className="flex items-center gap-1">
              <div>Collections</div>
              <Badge
                count={projectTagsMeta.length}
                color={Colors.lightestGray}
              />
            </div>
          ),
          key: "collections",
          children: (
            <Collections
              projectHash={projectHash}
              dataFilters={dataFilters}
              setDataFilters={setDataFilters}
              openModal={openModal}
              setOpenModal={setOpenModal}
            />
          )
        },
      ]}
      activeKey={tab}
      onChange={(key) => navigate(`../${key}`, { relative: "path" })}
    />
  );
}
