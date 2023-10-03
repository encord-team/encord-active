import { CSSProperties, useEffect, useMemo, useState } from "react";
import { Button, ConfigProvider, Spin, Tabs } from "antd";
import { useNavigate, useParams } from "react-router";
import { PlusOutlined } from "@ant-design/icons";
import {
  FeatureHashMap,
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
import { env } from "../constants";

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
  const [openModal, setOpenModal] = useState<undefined | "subset" | "upload">();

  // Loading screen while waiting for full summary of project metrics.
  if (projectSummary == null) {
    return <Spin indicator={loadingIndicator} />;
  }

  const remoteProject = !projectSummary.local_project;
  const tabBarStyle: CSSProperties = {
    background: "#FAFAFA",
    margin: 0,
    padding: "0px 10px",
  };

  return (
    <Tabs
      className="h-full"
      size="large"
      tabBarStyle={tabBarStyle}
      centered
      tabBarExtraContent={{
        left: (
          <ProjectSelector
            selectedProjectHash={projectHash}
            setSelectedProjectHash={setSelectedProjectHash}
          />
        ),
        right: (
          <ConfigProvider
            theme={{
              components: {
                Button: {
                  defaultBg: "#434343",
                },
              },
            }}
          >
            <Button
              className="border-none bg-gray-9 text-white"
              onClick={() => setOpenModal("subset")}
              // disabled={!canResetFilters}
              disabled
              hidden={env === "sandbox"}
              icon={<PlusOutlined />}
              size="large"
            >
              Create Annotate Project
            </Button>
          </ConfigProvider>
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

        // Not to be displayed. Commented to work upon later.
        // {
        //   label: "Project Comparison",
        //   key: "comparison",
        //   children: (
        //     <ProjectComparisonTab
        //       projectHash={projectHash}
        //       dataMetricsSummary={projectSummary.data}
        //       annotationMetricsSummary={projectSummary.annotation}
        //     />
        //   ),
        // },
      ]}
      activeKey={tab}
      onChange={(key) => navigate(`../${key}`, { relative: "path" })}
    />
  );
}
