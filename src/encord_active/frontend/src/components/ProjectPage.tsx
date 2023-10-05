import { useEffect, useMemo, useState } from "react";
import { Button, Spin, Tabs } from "antd";
import { useNavigate, useParams } from "react-router";
import { PlusOutlined } from "@ant-design/icons";
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
import { env } from "../constants";
import { classy } from "../helpers/classy";
import { PredictionsTab } from "./tabs/predictions/PredictionsTab";

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
        right: (
          <Button
            className={classy(
              "text-white disabled:bg-gray-9 disabled:bg-opacity-40 disabled:text-white",
              {
                "border-none bg-gray-9": hasSelectedItems,
              }
            )}
            onClick={() => setOpenModal("subset")}
            disabled={!hasSelectedItems}
            hidden={env === "sandbox"}
            icon={<PlusOutlined />}
            size="large"
          >
            Create Annotate Project
          </Button>
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
          label: "Model Evaluation",
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
