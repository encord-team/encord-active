import { useEffect, useMemo } from "react";
import { Spin, Tabs } from "antd";
import { useNavigate, useParams } from "react-router";
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
import { loadingIndicator } from "./Spin";
import { useProjectSummary } from "../hooks/queries/useProjectSummary";
import { useProjectHash } from "../hooks/useProjectHash";

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
          key: "summary",
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
            />
          ),
        },
        {
          label: "Predictions",
          key: "predictions",
          children: (
            <PredictionsTab
              projectHash={projectHash}
              annotationMetricsSummary={projectSummary.annotation}
              dataMetricsSummary={projectSummary.data}
              featureHashMap={featureHashMap}
              setSelectedProjectHash={setSelectedProjectHash}
              remoteProject={remoteProject}
            />
          ),
        },
        {
          label: "Project Comparison",
          key: "comparison",
          children: (
            <ProjectComparisonTab
              projectHash={projectHash}
              dataMetricsSummary={projectSummary.data}
              annotationMetricsSummary={projectSummary.annotation}
            />
          ),
        },
      ]}
      activeKey={tab}
      onChange={(key) => navigate(`../${key}`, { relative: "path" })}
    />
  );
}
