import Icon, {
  CheckOutlined,
  CloseOutlined,
  FontSizeOutlined,
  FullscreenOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { MdImageSearch } from "react-icons/md";
import { VscSymbolClass } from "react-icons/vsc";
import { RiUserLine } from "react-icons/ri";
import { Button, Card, Checkbox, Row, Tag, Typography } from "antd";
import React, { memo, useMemo } from "react";
import { useProjectSummary } from "../../hooks/queries/useProjectSummary";
import { useProjectItem } from "../../hooks/queries/useProjectItem";
import { AnnotatedImage } from "./AnnotatedImage";
import { usePredictionItem } from "../../hooks/queries/usePredictionItem";
import { classy } from "../../helpers/classy";
import { ItemTags } from "../explorer/Tagging";
import { FeatureHashMap } from "../Types";
import { AnalysisDomain, PredictionItem, ProjectItem } from "../../openapi/api";
import { AnnotationShapeIcon } from "../icons/AnnotationShapeIcon";
import {
  toAnnotationHash,
  toDataItemID,
  toPredictionTy,
} from "../util/ItemIdUtil";
import { calculateTruePositiveSet, getIOU } from "../util/PredictionUtil";

export const GalleryCard = memo(GalleryCardRaw);

function getPredictionIcon(ty: "TP" | "FP" | "FN"): React.ReactNode {
  if (ty === "TP") {
    return <CheckOutlined />;
  } else if (ty === "FP") {
    return <CloseOutlined />;
  } else {
    return <StopOutlined />;
  }
}

function getPredictionName(ty: "TP" | "FP" | "FN"): string {
  if (ty === "TP") {
    return "True Positive";
  } else if (ty === "FP") {
    return "False Positive";
  } else {
    return "False Negative";
  }
}

function getPredictionIOUPostfix(
  preview: PredictionItem | undefined,
  annotationHash: string,
  ty: "TP" | "FP" | "FN"
): string {
  if (preview === undefined || ty === "FN") {
    return "";
  }
  const iou = (getIOU(preview as any, annotationHash) ?? NaN).toFixed(2);

  // const featureMismatchSet = new Set(preview.annotation_feature_mismatch);
  // const featureMismatch = featureMismatchSet.has(annotationHash);
  // console.log("featureMismatch", featureMismatch);
  // if (featureMismatch) {
  //   return `: ${iou} (& wrong class)`;
  // }
  return `: ${iou}`;
}

function GalleryCardRaw(props: {
  projectHash: string;
  predictionHash: string | undefined;
  itemId: string;
  itemSimilarity: number | undefined;
  similaritySearchCard: boolean;
  selected: boolean;
  selectedMetric: string;
  analysisDomain: AnalysisDomain;
  setItemPreview: (itemId: string) => void;
  setSelectedToggle: (itemId: string) => void;
  setSimilaritySearch: (itemId: string | undefined) => void;
  hideExtraAnnotations: boolean;
  iou: number;
  featureHashMap: FeatureHashMap;
  gridCount: number | undefined;
}) {
  const {
    projectHash,
    predictionHash,
    itemId,
    itemSimilarity,
    similaritySearchCard,
    selected,
    selectedMetric,
    analysisDomain,
    setItemPreview,
    setSelectedToggle,
    setSimilaritySearch,
    hideExtraAnnotations,
    featureHashMap,
    iou,
    gridCount,
  } = props;
  // Conditionally extract annotation hash
  const dataId = toDataItemID(itemId);
  const annotationItem =
    analysisDomain === AnalysisDomain.Annotation ||
    predictionHash !== undefined;
  const annotationHash: string | undefined = annotationItem
    ? toAnnotationHash(itemId, predictionHash !== undefined)
    : undefined;

  // Conditionally extract prediction type
  const predictionTy: "TP" | "FP" | "FN" | undefined =
    predictionHash !== undefined ? toPredictionTy(itemId) : undefined;

  // Conditionally fetch the correct dataId from project or prediction.
  const projectItem = predictionTy === undefined || predictionTy === "FN";
  const { data: previewProject, isLoading: isLoadingProject } = useProjectItem(
    projectHash,
    dataId
  );
  const { data: previewPrediction, isLoading: isLoadingPrediction } =
    usePredictionItem(
      projectHash,
      predictionHash ?? "",
      predictionHash === undefined ? "" : dataId ?? "",
      {
        enabled: !projectItem,
      }
    );
  const preview: (PredictionItem & ProjectItem) | ProjectItem | undefined =
    useMemo(() => {
      if (projectItem) {
        return previewProject;
      } else if (
        previewProject !== undefined &&
        previewPrediction !== undefined
      ) {
        // Set values not set by prediction using values from preview project.
        return {
          ...previewProject,
          ...previewPrediction,
        } satisfies PredictionItem & ProjectItem;
      } else {
        return undefined;
      }
    }, [previewProject, previewPrediction, projectItem]);

  // Load project summary state for extra metadata
  const { data: projectSummary } = useProjectSummary(projectHash);
  const projectSummaryForDomain =
    projectSummary === undefined ? {} : projectSummary[analysisDomain].metrics;

  // Metric name for order by metric
  const metricName =
    projectSummaryForDomain === undefined
      ? "unknown"
      : projectSummaryForDomain[selectedMetric]?.title ?? "unknown";

  // Metric value for order by metric
  const displayValueData =
    preview === undefined ? NaN : preview.data_metrics[selectedMetric] ?? NaN;
  const displayValueAnnotationMetricsDict =
    preview === undefined
      ? {}
      : preview.annotation_metrics[annotationHash ?? ""] ?? {};
  const displayValueAnnotation =
    displayValueAnnotationMetricsDict[selectedMetric] ?? NaN;
  const displayValue = annotationItem
    ? displayValueAnnotation
    : displayValueData;

  // FIXME: const description = "FIXME_DESCRIPTION";
  const labelObject:
    | undefined
    | {
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
      } = useMemo(() => {
    if (annotationHash === undefined || preview === undefined) {
      return undefined;
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
    }[];

    return objOrClassList.find(
      (elem: { objectHash?: string; classificationHash?: string }) =>
        elem.objectHash === annotationHash ||
        elem.classificationHash === annotationHash
    );
  }, [annotationHash, preview]);

  const labelAnswers:
    | undefined
    | {
        classifications?: readonly {
          readonly answers?: string | any[];
          readonly featureHash: string;
        }[];
      } =
    preview === undefined || annotationHash === undefined
      ? undefined
      : preview.classification_answers[annotationHash] ??
        (preview.object_answers[annotationHash] as any);

  const isLoading = projectItem
    ? isLoadingProject
    : isLoadingProject && isLoadingPrediction;

  const [labelObjectName, labelObjectColor] = useMemo((): [
    string | null,
    string
  ] => {
    if (labelObject == null || preview == null) {
      return [null, "#00ffffff"];
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
        ? [null, "#00ffffff"]
        : [name, labelObject.color ?? "#00ffffff"];
    }
    return [featureMeta.name, featureMeta.color];
  }, [featureHashMap, labelObject, preview]);

  const annotationShape =
    preview === undefined
      ? null
      : preview.annotation_enums[annotationHash ?? ""]?.annotation_type;

  const predictionTruePositive: ReadonlySet<string> | undefined = useMemo(():
    | ReadonlySet<string>
    | undefined => {
    if (preview !== undefined && "annotation_iou_bounds" in preview) {
      return calculateTruePositiveSet(preview as any, iou); // FIXME: type properly?
    }
    return undefined;
  }, [preview, iou]);

  return (
    <Card
      hoverable
      onClick={() => setSelectedToggle(itemId)}
      loading={isLoading}
      bodyStyle={{ padding: 4, marginTop: "auto" }}
      className={classy(
        "z-1 annotated-image-cover flex h-full w-full flex-col p-1",
        {
          "border border-blue-300": selected,
        }
      )}
      cover={
        <div className="!flex items-center justify-center">
          {preview != null && (
            <AnnotatedImage
              item={preview}
              className={classy("relative", {
                "h-56 w-56": gridCount === 0,
              })}
              annotationHash={annotationHash}
              hideExtraAnnotations={hideExtraAnnotations}
              predictionTruePositive={predictionTruePositive}
              mode="full"
            >
              {similaritySearchCard ? (
                <>
                  <Icon
                    component={MdImageSearch}
                    className="top-50 left-50 absolute z-50 text-5xl opacity-70"
                  />
                  <div className="absolute z-40 h-20 w-20 rounded-md bg-gray-100 bg-opacity-70" />
                  <div
                    className={classy(
                      "group absolute z-30 h-full w-full bg-gray-100 bg-opacity-0 hover:bg-opacity-40 hover:opacity-100"
                    )}
                  >
                    <Checkbox
                      className={classy(
                        "absolute left-2 top-2 opacity-0 group-hover:opacity-100",
                        {
                          "!opacity-100": selected,
                        }
                      )}
                      checked={selected}
                    />
                    <div className="absolute top-2 right-2 flex flex-col gap-1 opacity-0 group-hover:opacity-100">
                      <Button
                        className="bg-white"
                        icon={<FullscreenOutlined />}
                        onClick={(e) => {
                          e.stopPropagation();
                          setItemPreview(itemId);
                        }}
                      />
                      <Button
                        className="bg-white"
                        shape="circle"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSimilaritySearch(undefined);
                        }}
                        icon={<CloseOutlined />}
                      />
                    </div>
                  </div>
                </>
              ) : (
                <div
                  className={classy(
                    "group absolute z-30 h-full w-full bg-gray-100 bg-opacity-0 hover:bg-opacity-40 hover:opacity-100"
                  )}
                >
                  <Checkbox
                    className={classy(
                      "absolute left-2 top-2 opacity-0 group-hover:opacity-100",
                      {
                        "!opacity-100": selected,
                      }
                    )}
                    checked={selected}
                  />
                  <div className="absolute top-2 right-2 flex flex-col gap-1 opacity-0 group-hover:opacity-100">
                    <Button
                      className="bg-white"
                      icon={<FullscreenOutlined />}
                      onClick={(e) => {
                        e.stopPropagation();
                        setItemPreview(itemId);
                      }}
                    />
                    <Button
                      className="bg-white"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSimilaritySearch(itemId);
                      }}
                      icon={<Icon component={MdImageSearch} />}
                    />
                  </div>
                  {
                    /* <div className="absolute top-7 flex h-5/6 w-full flex-col gap-3 overflow-y-auto p-2 pb-8 group-hover:opacity-100">
                    {description && (
                      <div className="flex flex-col">
                        <div className="inline-flex items-center gap-1">
                          <BsCardText className="text-base" />
                          <span>Description:</span>
                        </div>
                        <span>{description}</span>
                      </div>
                    )}
                  </div> */ null
                  }
                </div>
              )}
            </AnnotatedImage>
          )}
        </div>
      }
    >
      {itemSimilarity != null ? (
        <Row className="mt-1">
          <Tag bordered={false} color="red" className="rounded-xl">
            Distance:{" "}
            <span className="font-bold">{itemSimilarity.toFixed(5)}</span>
          </Tag>
        </Row>
      ) : null}
      <Row className="mt-1">
        <Tag bordered={false} color="gold" className="rounded-xl">
          {metricName}:{" "}
          <span className="font-bold">{displayValue.toFixed(5)}</span>
        </Tag>
      </Row>
      {preview?.tags &&
      (((labelObject == null
        ? 0
        : (preview?.tags?.label ?? {})[
            labelObject?.objectHash ?? labelObject?.classificationHash ?? ""
          ]?.length) ?? 0) > 0 ||
        (preview?.tags?.data?.length ?? 0) > 0) ? (
        <Row className="mt-1">
          <ItemTags
            tags={preview?.tags}
            annotationHash={
              labelObject?.objectHash ?? labelObject?.classificationHash
            }
            limit={4}
          />
        </Row>
      ) : null}
      {labelObject != null ? (
        <>
          <Row className="mt-1">
            <Tag bordered={false} color="magenta" className="rounded-xl">
              <Row>
                {annotationShape != null ? (
                  <AnnotationShapeIcon
                    shape={annotationShape}
                    color={labelObjectColor}
                  />
                ) : (
                  <VscSymbolClass />
                )}
                <Typography.Text className="ml-1" color={labelObjectColor}>
                  {labelObjectName}
                </Typography.Text>
              </Row>
            </Tag>
          </Row>
          {(labelAnswers?.classifications ?? [])
            .filter((v) => typeof v.answers === "string")
            .map((v) => (
              <Row className="mt-1" key={v.featureHash}>
                <Tag bordered={false} color="geekblue" className="rounded-xl">
                  <Row>
                    <FontSizeOutlined />
                    <Typography.Text className="ml-1">
                      {v.answers}
                    </Typography.Text>
                  </Row>
                </Tag>
              </Row>
            ))}
          {predictionTy != null ? (
            <Row className="mt-1">
              <Tag bordered={false} color="volcano" className="rounded-xl">
                {getPredictionIcon(predictionTy)}
                <Typography.Text className="ml-1" color={labelObjectColor}>
                  {getPredictionName(predictionTy)}
                  {getPredictionIOUPostfix(
                    preview as any, // FIXME: any?
                    annotationHash ?? "",
                    predictionTy
                  )}
                </Typography.Text>
              </Tag>
            </Row>
          ) : null}
          <Row className="mt-1">
            <Tag bordered={false} color="cyan" className="rounded-xl">
              <Icon component={RiUserLine} />
              <Typography.Text className="ml-1">
                {labelObject.lastEditedBy}
              </Typography.Text>
            </Tag>
          </Row>
        </>
      ) : null}
    </Card>
  );
}
