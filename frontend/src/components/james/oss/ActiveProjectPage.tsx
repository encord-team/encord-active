import * as React from "react";
import { useMemo, useState } from "react";
import { Spin, Tabs } from "antd";
import { ActiveQueryAPI } from "./ActiveTypes";
import ActivePredictionsTab from "./tabs/predictions/ActivePredictionsTab";
import { ProjectSelector } from "../../ProjectSelector";
import { IntegratedProjectMetadata } from "../IntegratedActiveAPI";
import ActiveProjectComparisonTab from "./tabs/ActiveProjectComparisonTab";
import { Explorer } from "../../explorer";
import ActiveSummaryView from "./tabs/ActiveSummaryView";
import { apiUrl } from "../../../constants";

function ActiveProjectPage(props: {
  queryAPI: ActiveQueryAPI;
  projectHash: string;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  setSelectedProject: (projectHash?: string) => void;
  projects: readonly IntegratedProjectMetadata[];
}) {
  const { queryAPI, projectHash, editUrl, projects, setSelectedProject } =
    props;
  const [activeTab, setActiveTab] = useState<string>("1");
  const { data: projectSummary } = queryAPI.useProjectSummary(projectHash);

  const featureHashMap = useMemo(() => {
    const featureHashMap: Record<
      string,
      { readonly color: string; readonly name: string }
    > = {};
    if (projectSummary == null) {
      return {};
    }
    projectSummary.ontology.objects.forEach((o) => {
      featureHashMap[o.featureNodeHash] = o;
    });
    projectSummary.ontology.classifications.forEach((o) => {
      featureHashMap[o.featureNodeHash] = o;
    });
    return featureHashMap;
  }, [projectSummary]);

  // Loading screen while waiting for full summary of project metrics.
  if (projectSummary == null) {
    return <Spin />;
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
            <ActiveSummaryView
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
            <Explorer
              projectHash={projectHash}
              metricsSummary={projectSummary.data}
              scope={"data"}
              queryAPI={queryAPI}
              featureHashMap={featureHashMap}
              setSelectedProjectHash={setSelectedProject}
              remoteProject={remoteProject}
              /* metricRanges={projectSummary.data?.metrics} */
            />
          ),
        },
        {
          label: "Predictions",
          key: "3",
          children: (
            <ActivePredictionsTab
              projectHash={projectHash}
              queryAPI={queryAPI}
              metricsSummary={projectSummary.annotations}
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: "Project Comparison",
          key: "4",
          children: (
            <ActiveProjectComparisonTab
              projectHash={projectHash}
              queryAPI={queryAPI}
              dataMetricsSummary={projectSummary.data}
              annotationMetricsSummary={projectSummary.annotations}
            />
          ),
        },
      ]}
      activeKey={activeTab}
      onChange={setActiveTab}
    />
  );
}

export default ActiveProjectPage;
