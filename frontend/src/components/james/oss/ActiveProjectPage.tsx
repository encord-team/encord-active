import * as React from "react";
import { useMemo, useState } from "react";
import { Spin, Tabs } from "antd";
import { ActiveQueryAPI } from "./ActiveTypes";
import ActiveAnalysisDomainTab from "./tabs/ActiveAnalysisDomainTab";
import ActivePredictionsTab from "./tabs/predictions/ActivePredictionsTab";
import { ProjectSelector } from "../../ProjectSelector";
import { IntegratedProjectMetadata } from "../IntegratedActiveAPI";
import ActiveProjectComparisonTab from "./tabs/ActiveProjectComparisonTab";

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
          label: "Data",
          key: "1",
          children: (
            <ActiveAnalysisDomainTab
              projectHash={projectHash}
              editUrl={editUrl}
              queryAPI={queryAPI}
              metricsSummary={projectSummary.data}
              analysisDomain="data"
              featureHashMap={featureHashMap}
            />
          ),
        },
        {
          label: "Annotations",
          key: "2",
          children: (
            <ActiveAnalysisDomainTab
              projectHash={projectHash}
              editUrl={editUrl}
              queryAPI={queryAPI}
              metricsSummary={projectSummary.annotations}
              analysisDomain="annotation"
              featureHashMap={featureHashMap}
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
          )
        }
      ]}
      activeKey={activeTab}
      onChange={setActiveTab}
    />
  );
}

export default ActiveProjectPage;
