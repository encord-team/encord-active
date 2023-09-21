import Icon, { FullscreenOutlined } from "@ant-design/icons";
import { MdImageSearch } from "react-icons/md";
import { VscSymbolClass } from "react-icons/vsc";
import { RiUserLine } from "react-icons/ri";
import { Button, Card, Checkbox, Row, Tag, Typography } from "antd";
import { ReactNode, memo, useMemo } from "react";
import { useProjectSummary } from "../../hooks/queries/useProjectSummary";
import { useProjectItem } from "../../hooks/queries/useProjectItem";
import { AnnotatedImage } from "./AnnotatedImage";
import { usePredictionItem } from "../../hooks/queries/usePredictionItem";
import { classy } from "../../helpers/classy";
import { ItemTags } from "../explorer/Tagging";
import { FeatureHashMap } from "../Types";
import { PredictionItem, ProjectItem } from "../../openapi/api";
import { AnnotationShapeIcon } from "../icons/AnnotationShapeIcon";
import {
  toAnnotationHash,
  toDataItemID,
  toPredictionTy,
} from "../util/ItemIdUtil";

export const GalleryCard = memo(GalleryCardRaw);

function GalleryCardRaw(props: {
  projectHash: string;
  predictionHash: string | undefined;
  itemId: string;
  selected: boolean;
  selectedMetric: { domain: "annotation" | "data"; metric_key: string };
  onExpand: (itemId: string) => void;
  onClick: (itemId: string) => void;
  onShowSimilar: (itemId: string) => void;
  hideExtraAnnotations: boolean;
  customTags?: ReactNode | undefined;
  iou: number;
  featureHashMap: FeatureHashMap;
}) {
  const {
    projectHash,
    predictionHash,
    itemId,
    selected,
    selectedMetric,
    onExpand,
    onClick,
    onShowSimilar,
    hideExtraAnnotations,
    customTags,
    featureHashMap,
    iou,
  } = props;
  // Conditionally extract annotation hash
  const dataId = toDataItemID(itemId);
  const annotationItem =
    selectedMetric.domain === "annotation" || predictionHash !== undefined;
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
    usePredictionItem(projectHash, predictionHash ?? "", dataId, {
      enabled: !projectItem,
    });
  const preview: PredictionItem | ProjectItem | undefined = useMemo(() => {
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
      } satisfies PredictionItem;
    } else {
      return undefined;
    }
  }, [previewProject, previewPrediction, projectItem]);

  // Load project summary state for extra metadata
  const { data: projectSummary } = useProjectSummary(projectHash);
  const projectSummaryForDomain =
    projectSummary === undefined
      ? {}
      : projectSummary[selectedMetric.domain].metrics;

  // Metric name for order by metric
  const metricName =
    projectSummaryForDomain === undefined
      ? "unknown"
      : projectSummaryForDomain[selectedMetric.metric_key]?.title ?? "unknown";

  // Metric value for order by metric
  const displayValueData =
    preview === undefined
      ? NaN
      : preview.data_metrics[selectedMetric.metric_key] ?? NaN;
  const displayValueAnnotationMetricsDict =
    preview === undefined
      ? {}
      : preview.annotation_metrics[annotationHash ?? ""] ?? {};
  const displayValueAnnotation =
    displayValueAnnotationMetricsDict[selectedMetric.metric_key] ?? NaN;
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
    if (preview !== undefined && "annotation_tp_bounds" in preview) {
      const tpSet = new Set();
      Object.entries(preview.annotation_tp_bounds ?? {}).forEach(
        ([key, [keyIOU, keyIOUBound]]) => {
          if (keyIOU >= iou && keyIOUBound < iou) {
            tpSet.add(key);
          }
        }
      );
      return new Set(tpSet) as ReadonlySet<string>;
    }
    return undefined;
  }, [preview, iou]);

  return (
    <Card
      hoverable
      style={{ width: 240, margin: 10 }}
      onClick={() => onClick(itemId)}
      loading={isLoading}
      bodyStyle={{ padding: 4 }}
      className={classy("group m-2.5 w-60 overflow-clip", {
        "border-blue-300": selected,
      })}
      cover={
        <div className="!flex items-center justify-center">
          {preview != null && (
            <AnnotatedImage
              item={preview}
              className="h-56"
              annotationHash={annotationHash}
              hideExtraAnnotations={hideExtraAnnotations}
              predictionTruePositive={predictionTruePositive}
              mode="full"
            >
              <div
                className={classy(
                  "absolute z-10 h-full w-full bg-gray-100 bg-opacity-0 group-hover:bg-opacity-70",
                  "[&>*]:opacity-0 [&>*]:group-hover:opacity-100"
                )}
              >
                <Checkbox
                  className={classy("absolute left-2 top-2", {
                    "!opacity-100": selected,
                  })}
                  checked={selected}
                />
                <div className="absolute top-2 right-2 flex flex-col gap-1">
                  <Button
                    className="bg-white"
                    icon={<FullscreenOutlined />}
                    onClick={(e) => {
                      e.stopPropagation();
                      onExpand(itemId);
                    }}
                  />
                  <Button
                    className="bg-white"
                    onClick={(e) => {
                      e.stopPropagation();
                      onShowSimilar(itemId);
                    }}
                    type="text"
                    key="similarity-search"
                    icon={
                      <span
                        role="img"
                        aria-label="fullscreen"
                        className="anticon anticon-fullscreen"
                      >
                        <MdImageSearch />
                      </span>
                    }
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
            </AnnotatedImage>
          )}
        </div>
      }
    >
      {customTags != null ? <Row className="mt-1">{customTags}</Row> : null}
      <Row className="mt-1">
        <Tag bordered={false} color="gold" className="rounded-xl">
          {metricName}:{" "}
          <span className="font-bold">{displayValue.toFixed(5)}</span>
        </Tag>
      </Row>
      {preview?.tags &&
      ((labelObject == null
        ? preview?.tags?.data?.length
        : (preview?.tags?.label ?? {})[
            labelObject?.objectHash ?? labelObject?.classificationHash ?? ""
          ]?.length) ?? 0) > 0 ? (
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
