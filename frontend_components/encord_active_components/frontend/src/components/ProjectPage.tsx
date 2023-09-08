import * as React from "react";
import { useEffect, useMemo, useState } from "react";
import { Spin, Tabs } from "antd";
import {
  OntologyObjectAttribute,
  OntologyObjectAttributeOptions,
  ProjectOntology,
} from "./Types";
import { PredictionsTab } from "./tabs/predictions/PredictionsTab";
import { ProjectSelector } from "./ProjectSelector";
import { ProjectComparisonTab } from "./tabs/ProjectComparisonTab";
import { Explorer } from "./explorer";
import { SummaryView } from "./tabs/SummaryView";
import { getApi, ApiContext } from "./explorer/api";
import { useAuth } from "../authContext";
import { loadingIndicator } from "./Spin";
import { useQuerier } from "../hooks/Context";
import { useProjectSummary } from "../hooks/queries/useProjectSummary";

export function ProjectPage(props: {
  projectHash: string;
  encordDomain: string;
  setSelectedProjectHash: (projectHash?: string) => void;
}) {
  const { projectHash, setSelectedProjectHash, encordDomain } =
    props;

  const querier = useQuerier()

  const [activeTab, setActiveTab] = useState<string>("1");
  const { data: projectSummary, isError } = useProjectSummary(
    projectHash
  );

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

  const featureHashMap = useMemo(() => {
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

  const { token } = useAuth();
  const api = getApi(projectHash, token);

  // Loading screen while waiting for full summary of project metrics.
  if (projectSummary == null) {
    return <Spin indicator={loadingIndicator} />;
  }

  const remoteProject = !projectSummary.local_project;

  return (
    <Tabs
      tabBarExtraContent={
        <ProjectSelector
          selectedProjectHash={projectHash}
          setSelectedProjectHash={setSelectedProjectHash}
        />
      }
      items={[
        {
          label: "Summary",
          key: "1",
          children: (
            <SummaryView
              projectHash={projectHash}
              projectSummary={projectSummary}
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: "Explorer",
          key: "2",
          children: (
            <ApiContext.Provider value={api}>
              <Explorer
                projectHash={projectHash}
                predictionHash={undefined}
                dataMetricsSummary={projectSummary.data}
                annotationMetricsSummary={projectSummary.annotation}
                editUrl={editUrl}
                featureHashMap={featureHashMap}
                setSelectedProjectHash={setSelectedProjectHash}
                remoteProject={remoteProject}
              />
            </ApiContext.Provider>
          ),
        },
        {
          label: "Predictions",
          key: "3",
          children: (
            <ApiContext.Provider value={api}>
              <PredictionsTab
                projectHash={projectHash}
                annotationMetricsSummary={projectSummary.annotation}
                dataMetricsSummary={projectSummary.data}
                featureHashMap={featureHashMap}
                setSelectedProjectHash={setSelectedProjectHash}
                remoteProject={remoteProject}
              />
            </ApiContext.Provider>
          ),
        },
        {
          label: "Project Comparison",
          key: "4",
          children: (
            <ProjectComparisonTab
              projectHash={projectHash}
              dataMetricsSummary={projectSummary.data}
              annotationMetricsSummary={projectSummary.annotation}
            />
          ),
        },
      ]}
      activeKey={activeTab}
      onChange={setActiveTab}
    />
  );
}
