import * as React from "react";
import { FullscreenOutlined, EditOutlined } from "@ant-design/icons";
import { MdImageSearch } from "react-icons/md";
import { VscSymbolClass } from "react-icons/vsc";
import { RiUserLine } from "react-icons/ri";
import { Button, Card, Checkbox, Row, Typography } from "antd";
import { useMemo } from "react";
import { QueryContext } from "../../hooks/Context";
import { useProjectSummary } from "../../hooks/queries/useProjectSummary";
import { useProjectItem } from "../../hooks/queries/useProjectItem";
import { AnnotatedImage } from "./AnnotatedImage";
import { usePredictionItem } from "../../hooks/queries/usePredictionItem";

export function GalleryCard(props: {
  projectHash: string;
  predictionHash: string | undefined;
  queryContext: QueryContext;
  itemId: string;
  selected: boolean;
  selectedMetric: { domain: "annotation" | "data"; metric_key: string };
  similaritySearchDisabled: boolean;
  onExpand: () => void;
  onClick: () => void;
  onShowSimilar: () => void;
  editUrl:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  hideExtraAnnotations: boolean;
}) {
  const {
    projectHash,
    predictionHash,
    queryContext,
    itemId,
    selected,
    selectedMetric,
    similaritySearchDisabled,
    onExpand,
    onClick,
    onShowSimilar,
    editUrl,
    hideExtraAnnotations,
  } = props;
  // Conditionally extract annotation hash
  const dataId = itemId.split("_").slice(0, 2).join("_");
  const annotationItem =
    selectedMetric.domain === "annotation" || predictionHash !== undefined;
  const annotationHash: string | undefined = annotationItem
    ? itemId.split("_")[2]
    : undefined;

  // Conditionally extract prediction type
  const predictionTy: "TP" | "FP" | "FN" | string | undefined =
    predictionHash !== undefined ? itemId.split("_")[3] : undefined;

  // Conditionally fetch the correct dataId from project or prediction.
  const projectItem = predictionTy === undefined || predictionTy === "FN";
  const { data: previewProject, isLoading: isLoadingProject } = useProjectItem(
    queryContext,
    projectHash,
    dataId
  );
  const { data: previewPrediction, isLoading: isLoadingPrediction } =
    usePredictionItem(queryContext, projectHash, predictionHash ?? "", dataId, {
      enabled: !projectItem,
    });
  const preview = useMemo(() => {
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
      };
    } else {
      return undefined;
    }
  }, [previewProject, previewPrediction, projectItem]);
  const isLoading = projectItem ? isLoadingProject : true;

  // Load project summary state for extra metadata
  const { data: projectSummary } = useProjectSummary(queryContext, projectHash);
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

  const description = "FIXME_DESCRIPTION";
  const labelObject = useMemo(() => {
    if (annotationHash === undefined || preview === undefined) {
      return undefined;
    }
    const objOrClass = preview.objects
      .concat(preview.classifications)
      .find(
        (elem: { objectHash?: string; classificationHash?: string }) =>
          elem.objectHash === annotationHash ||
          elem.classificationHash === annotationHash
      );

    return objOrClass as {
      readonly objectHash: string;
      readonly color: string;
      readonly confidence: number;
      readonly createdAt: string;
      readonly createdBy: string;
      readonly featureHash: string;
      readonly lastEditedAt: string;
      readonly lastEditedBy: string;
      readonly manualAnnotation: boolean;
      readonly name: string;
      readonly value: string;
    };
  }, [annotationHash, preview]);

  return (
    <Card
      hoverable
      style={{ width: 240, margin: 10 }}
      onClick={onClick}
      loading={isLoading}
      bodyStyle={{ padding: 0 }}
      cover={
        <div>
          {preview != null && (
            <AnnotatedImage
              queryContext={queryContext}
              item={preview}
              className="group-hover:opacity-30"
              annotationHash={annotationHash}
              hideExtraAnnotations={hideExtraAnnotations}
              mode="full"
            >
              <div className="absolute z-10 h-full w-full bg-gray-100 bg-opacity-70 opacity-0 hover:opacity-100">
                <Checkbox
                  className="absolute left-2 top-2"
                  checked={selected}
                  onChange={() => {}}
                />
                <Typography.Text className="absolute top-2">
                  {metricName}: {displayValue.toFixed(5)}
                </Typography.Text>
                <Button
                  className="absolute top-2 right-2 bg-white"
                  icon={<FullscreenOutlined />}
                  shape="circle"
                  onClick={onExpand}
                />
                {
                  /* <div className="absolute top-7 flex h-5/6 w-full flex-col gap-3 overflow-y-auto p-2 pb-8 group-hover:opacity-100">
                  {<TagList tags={data.tags} />}
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
      actions={[
        <Button
          disabled={similaritySearchDisabled}
          onClick={onShowSimilar}
          type="text"
          key="similarity-search"
          icon={<MdImageSearch />}
        />,
        <Button
          disabled={editUrl == null || preview == null}
          onClick={() =>
            editUrl != null && preview != null
              ? window.open(
                  editUrl(preview.data_hash, projectHash, 0).toString(),
                  "_blank"
                )
              : null
          }
          type="text"
          key="edit"
          icon={<EditOutlined />}
        />,
      ]}
    >
      {labelObject != null ? (
        <>
          <Row>
            <VscSymbolClass />
            {labelObject.name}
          </Row>
          <Row>
            <RiUserLine />
            {labelObject.lastEditedBy}
          </Row>
        </>
      ) : null}
    </Card>
  );
}
