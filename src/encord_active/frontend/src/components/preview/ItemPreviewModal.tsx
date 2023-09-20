import { Button, Col, Modal, Row, Spin, Table } from "antd";

import { useMemo } from "react";
import { MdImageSearch } from "react-icons/md";
import { EditOutlined } from "@ant-design/icons";
import { useProjectItem } from "../../hooks/queries/useProjectItem";
import { loadingIndicator } from "../Spin";
import { useProjectSummary } from "../../hooks/queries/useProjectSummary";
import { AnnotatedImage } from "./AnnotatedImage";
import { ItemTags } from "../explorer/Tagging";
import { usePredictionItem } from "../../hooks/queries/usePredictionItem";

export function ItemPreviewModal(props: {
  projectHash: string;
  predictionHash: string | undefined;
  previewItem: string | undefined;
  domain: "annotation" | "data";
  onClose: () => void;
  onShowSimilar: () => void;
  editUrl:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
}) {
  const {
    previewItem,
    domain,
    projectHash,
    predictionHash,
    onClose,
    onShowSimilar,
    editUrl,
  } = props;
  const dataId =
    previewItem === undefined
      ? undefined
      : previewItem.split("_").slice(0, 2).join("_");
  const frame =
    previewItem === undefined ? undefined : Number(previewItem.split("_")[1]);
  const annotationHash: string | undefined =
    domain === "annotation" && previewItem !== undefined
      ? previewItem.split("_")[2]
      : undefined;
  const { data: previewProject } = useProjectItem(projectHash, dataId ?? "", {
    enabled: dataId !== undefined,
  });
  const { data: previewPrediction } = usePredictionItem(
    projectHash,
    predictionHash ?? "",
    dataId ?? "",
    {
      enabled: dataId !== undefined,
    }
  );
  const preview = useMemo(() => {
    if (predictionHash === undefined) {
      return previewProject;
    } else if (
      previewProject !== undefined &&
      previewPrediction !== undefined
    ) {
      // Set values not set by prediction using values from preview project.
      return {
        ...previewProject,
        ...previewPrediction,
      };
    } else {
      return undefined;
    }
  }, [previewProject, previewPrediction, predictionHash]);
  const { data: projectSummary } = useProjectSummary(projectHash);

  const metricsList = useMemo(() => {
    const projectSummaryForDomain =
      projectSummary === undefined ? {} : projectSummary[domain].metrics;

    const metricsDict =
      domain === "data" || preview === undefined
        ? preview?.data_metrics ?? {}
        : preview.annotation_metrics[annotationHash ?? ""] ?? {};

    const metricsList = Object.entries(metricsDict).map(([name, value]) => ({
      name: projectSummaryForDomain[name]?.title ?? name,
      value,
    }));
    const sortByName = (a: { name: string }, b: { name: string }) => {
      if (a === b) {
        return 0;
      }
      return a < b ? 1 : -1;
    };

    return metricsList.sort(sortByName);
  }, [preview, projectSummary, annotationHash, domain]);

  const columns = [
    {
      title: domain === "data" ? "Data Metric" : "Annotation Metric",
      dataIndex: "name",
      key: "name",
    },
    {
      title: "Value",
      dataIndex: "value",
      key: "value",
    },
  ];

  return (
    <Modal
      title={preview?.data_title ?? ""}
      open={dataId !== undefined}
      onCancel={onClose}
      width="95vw"
      footer={
        <>
          <Button onClick={onShowSimilar} icon={<MdImageSearch />}>
            Similarity search
          </Button>
          <Button
            disabled={editUrl === undefined}
            icon={<EditOutlined />}
            onClick={() =>
              editUrl !== undefined &&
              preview !== undefined &&
              frame !== undefined
                ? window.open(
                    editUrl(preview.data_hash, projectHash, frame),
                    "_blank"
                  )
                : undefined
            }
          >
            Edit in Encord
          </Button>
        </>
      }
    >
      {preview === undefined ? (
        <Spin indicator={loadingIndicator} />
      ) : (
        <Row className="vh-100 vw-100">
          <Col span={12}>
            <Row className="[&>*]:w-full">
              <Table
                dataSource={metricsList}
                columns={columns}
                pagination={{ pageSize: 5 }}
                id="key"
              />
            </Row>
            <ItemTags tags={preview.tags} annotationHash={annotationHash} />
          </Col>
          <Col span={12}>
            <AnnotatedImage
              key={`preview-image-${previewItem ?? ""}`}
              item={preview}
              annotationHash={annotationHash}
              mode="large"
              predictionTruePositive={undefined}
            />
          </Col>
        </Row>
      )}
    </Modal>
  );
  /*
  const dataId = id.split("_").slice(0, 3).join("_");
  const { data: preview, isLoading } = useProjectDataItem(
    projectHash,
    dataId
  );
  const mutate = () => console.log("fixme");

  if (isLoading || !preview) {
    return <Spin indicator={loadingIndicator} />;
  }

  // const { description, ...metrics } = preview.metadata.metrics;
  const { editUrl } = data;
  const editUrl = "FIXME";
  const description = "";

  return (
    <div className="flex w-full flex-col items-center gap-3 p-1">
      <div className="flex w-full justify-between">
        <div className="flex gap-3">
          <button
            className="btn btn-ghost gap-2"
            disabled={similaritySearchDisabled}
            onClick={onShowSimilar}
          >
            <MdImageSearch className="text-base" />
            Similar
          </button>
          <button
            className="btn btn-ghost gap-2"
            onClick={() =>
              editUrl ? window.open(editUrl, "_blank") : undefined
            }
            disabled={editUrl == null}
          >
            <FaEdit />
            Edit
          </button>
          <TaggingDropdown
            disabledReason={scope === "prediction" ? scope : undefined}
          >
            <TaggingForm
              onChange={(groupedTags) => mutate([{ id, groupedTags }])}
              selectedTags={{ data: [], label: [] }} // FIXME:
              tabIndex={0}
              allowTaggingAnnotations={allowTaggingAnnotations}
            />
          </TaggingDropdown>
        </div>
        <button onClick={onClose} className="btn btn-outline btn-square">
          <MdClose className="text-base" />
        </button>
      </div>
      <div className="flex w-full justify-between">
        <div className="flex flex-col gap-5">
          <div className="flex flex-col">
            <div>
              <span>Title: </span>
              <span>{preview?.data_title ?? "unknown"}</span>
            </div>
            {description && (
              <div>
                <span>Description: </span>
                <span>{description}</span>
              </div>
            )}
          </div>
          {/ <MetadataMetrics metrics={metrics} />
          <TagList tags={data.tags} /> /}
        </div>
        <div className="relative inline-block w-fit">
          <ImageWithPolygons className="" preview={preview} />
        </div>
      </div>
    </div>
  );
  */
}
