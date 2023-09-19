import { useEffect, useMemo, useState } from "react";
import { Tabs } from "antd";
import {
  OntologyObjectAttribute,
  OntologyObjectAttributeOptions,
  QueryAPI,
} from "./Types";
import { PredictionsTab } from "./tabs/predictions/PredictionsTab";
import { ProjectSelector } from "./ProjectSelector";
import { IntegratedProjectMetadata } from "./IntegratedAPI";
import { ProjectComparisonTab } from "./tabs/ProjectComparisonTab";
import { Explorer } from "./explorer";
import { SummaryView } from "./tabs/SummaryView";
import { getApi, ApiContext } from "./explorer/api";
import { useAuth } from "../authContext";
import { Spinner } from "./explorer/Spinner";

export function ProjectPage(props: {
  queryAPI: QueryAPI;
  projectHash: string;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  setSelectedProjectHash: (projectHash?: string) => void;
  projects: readonly IntegratedProjectMetadata[];
}) {
  const {
    queryAPI,
    projectHash,
    projects,
    setSelectedProjectHash: setSelectedProject,
  } = props;
  const [activeTab, setActiveTab] = useState<string>("1");
  const { data: projectSummary, isError } =
    queryAPI.useProjectSummary(projectHash);

  // Go to parent in the error case (project does not exist).
  useEffect(() => {
    if (isError) {
      setSelectedProject(undefined);
    }
  }, [isError]);

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
      color: string,
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
    projectSummary.ontology.objects.forEach((o) => {
      featureHashMap[o.featureNodeHash] = {
        color: o.color,
        name: o.name ?? o.featureNodeHash,
      };
      if (o.attributes != null) {
        o.attributes.forEach((a) => procAttribute(a, o.color));
      }
    });
    projectSummary.ontology.classifications.forEach((o) => {
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
    return <Spinner />;
  }

  const remoteProject = !projectSummary.local_project;

  return (
    <Tabs
      tabBarExtraContent={
        <ProjectSelector
          projects={projects}
          selectedProjectHash={projectHash}
          onViewAllProjects={() => setSelectedProject(undefined)}
          onSelectedProjectChange={setSelectedProject}
        />
      }
      items={[
        {
          label: "Summary",
          key: "1",
          children: (
            <SummaryView
              projectHash={projectHash}
              queryAPI={queryAPI}
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
                metricsSummary={projectSummary.global}
                scope={"data"}
                queryAPI={queryAPI}
                featureHashMap={featureHashMap}
                setSelectedProjectHash={setSelectedProject}
                remoteProject={remoteProject}
                /* metricRanges={projectSummary.data?.metrics} */
              />
            </ApiContext.Provider>
          ),
        },
        /* { */
        /*   label: "Predictions", */
        /*   key: "3", */
        /*   children: ( */
        /*     <ApiContext.Provider value={api}> */
        /*       <PredictionsTab */
        /*         projectHash={projectHash} */
        /*         queryAPI={queryAPI} */
        /*         metricsSummary={projectSummary.annotations} */
        /*         featureHashMap={featureHashMap} */
        /*         setSelectedProjectHash={setSelectedProject} */
        /*         remoteProject={remoteProject} */
        /*       /> */
        /*     </ApiContext.Provider> */
        /*   ), */
        /* }, */
        /* { */
        /*   label: "Project Comparison", */
        /*   key: "4", */
        /*   children: ( */
        /*     <ProjectComparisonTab */
        /*       projectHash={projectHash} */
        /*       queryAPI={queryAPI} */
        /*       dataMetricsSummary={projectSummary.data} */
        /*       annotationMetricsSummary={projectSummary.annotations} */
        /*     /> */
        /*   ), */
        /* }, */
      ]}
      activeKey={activeTab}
      onChange={setActiveTab}
    />
  );
}
