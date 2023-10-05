import {
  Button,
  Col,
  List,
  Modal,
  Row,
  Spin,
  Table,
  Tabs,
  Typography,
} from "antd";

import { useMemo } from "react";
import { MdImageSearch } from "react-icons/md";
import { EditOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { useProjectItem } from "../../hooks/queries/useProjectItem";
import { loadingIndicator } from "../Spin";
import { useProjectSummary } from "../../hooks/queries/useProjectSummary";
import { AnnotatedImage } from "./AnnotatedImage";
import { ItemTags } from "../explorer/Tagging";
import { usePredictionItem } from "../../hooks/queries/usePredictionItem";
import { AnnotationType } from "../../openapi/api";
import { AnnotationShapeIcon } from "../icons/AnnotationShapeIcon";
import { VscSymbolClass } from "react-icons/vsc";
import { FeatureHashMap } from "../Types";

export function ItemPreviewModal(props: {
  featureHashMap: FeatureHashMap;
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
    featureHashMap,
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
    predictionHash === undefined ? "" : dataId ?? "",
    {
      enabled: dataId !== undefined && predictionHash !== undefined,
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

  const metadataList: { name: string; value: string }[] = useMemo(() => {
    const metadataList: { name: string; value: string }[] = [
      { name: "Data title", value: preview?.data_title ?? "" },
      { name: "Dataset", value: preview?.dataset_title ?? "" },
    ];
    return metadataList;
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

  const objectsAndClassiciations: {
    readonly confidence: number;
    readonly createdAt: string;
    readonly createdBy: string;
    readonly featureHash: string;
    readonly lastEditedAt: string;
    readonly lastEditedBy: string;
    readonly manualAnnotation: boolean;
    readonly objectHash?: string;
    readonly classificationHash?: string;
    readonly name?: string;
    readonly color?: string;
    readonly shape?: AnnotationType;
  }[] = useMemo(() => {
    console.log(annotationHash, preview);
    if (preview === undefined) {
      return [];
    }
    const objOrClassList = [
      ...preview.objects,
      ...preview.classifications,
    ] as readonly {
      readonly confidence: number;
      readonly createdAt: string;
      readonly createdBy: string;
      readonly featureHash: string;
      readonly lastEditedAt: string;
      readonly lastEditedBy: string;
      readonly manualAnnotation: boolean;
      readonly objectHash?: string;
      readonly classificationHash?: string;
      readonly name?: string;
      readonly shape?: AnnotationType;
    }[];

    return objOrClassList.filter(
      (elem: { objectHash?: string; classificationHash?: string }) =>
        elem.objectHash === annotationHash ||
        elem.classificationHash === annotationHash
    );
  }, [annotationHash, preview]);

  const objects = objectsAndClassiciations.filter(
    (elem) => elem.objectHash !== undefined
  );

  const classifications = objectsAndClassiciations.filter(
    (elem) => elem.classificationHash !== undefined
  );

  const annotationShape =
    preview === undefined
      ? null
      : preview.annotation_enums[annotationHash ?? ""]?.annotation_type;

  const objectLabels = useMemo((): {
    labelObjectName: string | null;
    labelObjectColor: string;
    labelShape: AnnotationType | null | undefined;
  }[] => {
    return objects.map((labelObject) => {
      if (labelObject == null || preview == null) {
        return {
          labelObjectName: null,
          labelObjectColor: "#00ffffff",
          labelShape: null,
        };
      }
      let { featureHash } = labelObject;
      const classificationAnswer = preview.classification_answers[
        labelObject.classificationHash ?? ""
      ] as {
        readonly classifications?: {
          readonly featureHash: string;
          readonly answers?: readonly { readonly featureHash: string }[];
        }[];
      };
      if (classificationAnswer !== undefined) {
        const { classifications } = classificationAnswer;
        if (classifications !== undefined && classifications.length > 0) {
          const { answers } = classifications[0];
          if (answers !== undefined && answers.length > 0) {
            const { featureHash: classificationFeatureHash } = answers[0];
            featureHash = classificationFeatureHash;
          }
        }
      }
      const featureMeta = featureHashMap[featureHash];
      if (featureMeta == null) {
        const name = labelObject?.name ?? null;

        return name === null
          ? {
              labelObjectName: null,
              labelObjectColor: "#00ffffff",
              labelShape: labelObject.shape,
            }
          : {
              labelObjectName: name,
              labelObjectColor: labelObject.color ?? "#00ffffff",
              labelShape: null,
            };
      }
      return {
        labelObjectName: featureMeta.name,
        labelObjectColor: featureMeta.color,
        labelShape:
          preview.annotation_enums[labelObject.objectHash ?? ""]
            ?.annotation_type,
      };
    });
  }, [featureHashMap, objects, preview]);

  const classificationLabels = useMemo((): {
    labelObjectName: string | null;
    labelObjectColor: string;
    labelShape: AnnotationType | null | undefined;
  }[] => {
    return classifications.map((labelObject) => {
      if (labelObject == null || preview == null) {
        return {
          labelObjectName: null,
          labelObjectColor: "#00ffffff",
          labelShape: null,
        };
      }
      let { featureHash } = labelObject;
      const classificationAnswer = preview.classification_answers[
        labelObject.classificationHash ?? ""
      ] as {
        readonly classifications?: {
          readonly featureHash: string;
          readonly answers?: readonly { readonly featureHash: string }[];
        }[];
      };
      if (classificationAnswer !== undefined) {
        const { classifications } = classificationAnswer;
        if (classifications !== undefined && classifications.length > 0) {
          const { answers } = classifications[0];
          if (answers !== undefined && answers.length > 0) {
            const { featureHash: classificationFeatureHash } = answers[0];
            featureHash = classificationFeatureHash;
          }
        }
      }
      const featureMeta = featureHashMap[featureHash];
      if (featureMeta == null) {
        const name = labelObject?.name ?? null;

        return name === null
          ? {
              labelObjectName: null,
              labelObjectColor: "#00ffffff",
              labelShape: labelObject.shape,
            }
          : {
              labelObjectName: name,
              labelObjectColor: labelObject.color ?? "#00ffffff",
              labelShape: null,
            };
      }
      return {
        labelObjectName: featureMeta.name,
        labelObjectColor: featureMeta.color,
        labelShape:
          preview.annotation_enums[labelObject.objectHash ?? ""]
            ?.annotation_type,
      };
    });
  }, [featureHashMap, classifications, preview]);

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
          <Col span={16}>
            <AnnotatedImage
              key={`preview-image-${previewItem ?? ""}`}
              item={preview}
              annotationHash={annotationHash}
              mode="large"
              predictionTruePositive={undefined}
            />
          </Col>
          <Col span={8}>
            <Tabs
              className="h-full px-2"
              items={[
                {
                  label: "MetaData",
                  key: "metadata",
                  children: (
                    <div className="relative h-full overflow-y-auto ">
                      <div className="absolute">
                        <List
                          dataSource={metadataList}
                          renderItem={(item) => (
                            <List.Item>
                              <div className="flex flex-col">
                                <div className="text-xs text-gray-7">
                                  {item.name} <InfoCircleOutlined />
                                </div>
                                <div className="text-xs text-gray-9">
                                  {item.value}
                                </div>
                              </div>
                            </List.Item>
                          )}
                        />
                        <ItemTags
                          tags={preview.tags}
                          annotationHash={annotationHash}
                        />
                      </div>
                    </div>
                  ),
                },
                {
                  label: "Metrics",
                  key: "metrics",
                  children: (
                    <div className="relative h-full overflow-y-auto ">
                      <List
                        className="absolute"
                        dataSource={metricsList}
                        renderItem={(item) => (
                          <List.Item>
                            <div className="flex flex-col">
                              <div className="text-xs text-gray-7">
                                {item.name} <InfoCircleOutlined />
                              </div>
                              <div className="text-xs text-gray-9">
                                {item.value}
                              </div>
                            </div>
                          </List.Item>
                        )}
                      />
                    </div>
                  ),
                },
                {
                  label: "Labels & Predictions",
                  key: "labels",
                  children: (
                    <div className="relative h-full overflow-y-auto ">
                      <div className="absolute">
                        {objectLabels && objectLabels.length > 0 && (
                          <>
                            <div className="font-semibold">Objects</div>
                            <List
                              dataSource={objectLabels}
                              renderItem={(item) => (
                                <Row>
                                  {item.labelShape && (
                                    <AnnotationShapeIcon
                                      shape={item.labelShape}
                                      color={item.labelObjectColor}
                                    />
                                  )}

                                  <Typography.Text
                                    className="ml-1"
                                    color={item.labelObjectColor}
                                  >
                                    {item.labelObjectName}
                                  </Typography.Text>
                                </Row>
                              )}
                            />
                          </>
                        )}

                        {classificationLabels &&
                          classificationLabels.length > 0 && (
                            <>
                              <div className="font-semibold">
                                Classifications
                              </div>
                              <List
                                dataSource={classificationLabels}
                                renderItem={(item) => (
                                  <Row>
                                    {item.labelShape && (
                                      <AnnotationShapeIcon
                                        shape={item.labelShape}
                                        color={item.labelObjectColor}
                                      />
                                    )}

                                    <Typography.Text
                                      className="ml-1"
                                      color={item.labelObjectColor}
                                    >
                                      {item.labelObjectName}
                                    </Typography.Text>
                                  </Row>
                                )}
                              />
                            </>
                          )}

                        <ItemTags
                          tags={preview.tags}
                          annotationHash={annotationHash}
                        />
                      </div>
                    </div>
                  ),
                },
              ]}
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
