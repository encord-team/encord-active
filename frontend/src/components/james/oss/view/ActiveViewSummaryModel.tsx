import * as React from "react";
import useResizeObserver from "use-resize-observer";
import { Col, Divider, Row, Typography } from "antd";
import ActiveViewLabelledImage from "./ActiveViewLabelledImage";
import { ActiveProjectPreviewItemResult, ActiveQueryAPI } from "../ActiveTypes";

function ActiveViewSummaryModel(props: {
  queryAPI: ActiveQueryAPI;
  projectHash: string;
  duHash: string;
  frame: number;
  objectHash?: string | undefined;
  visualization: ActiveProjectPreviewItemResult;
}) {
  const { queryAPI, projectHash, duHash, frame, objectHash, visualization } =
    props;
  const { ref, width } = useResizeObserver();
  const detailedResult = queryAPI.useProjectItemDetails(
    projectHash,
    duHash,
    frame
  );
  return (
    <>
      <Row>
        <Typography.Title level={2}>
          Data Title: {detailedResult.data?.data_title ?? ""}
        </Typography.Title>
      </Row>
      <Divider />
      <Row>
        <Col span={12} style={{ padding: "10px" }}>
          <Row>
            <Typography.Title level={2}>Metadata: </Typography.Title>
          </Row>
          <Row>
            <Typography.Text strong>Dataset:</Typography.Text>
            <Typography.Text>
              {detailedResult.data?.dataset_title}
            </Typography.Text>
          </Row>
          <Row>
            <Typography.Text strong>Data Type:</Typography.Text>
            <Typography.Text>{detailedResult.data?.data_type}</Typography.Text>
          </Row>
          {detailedResult.data?.data_type === "video" ? (
            <>
              <Row>
                <Typography.Text strong>Frame:</Typography.Text>
                <Typography.Text>
                  {frame}/{detailedResult.data?.num_frames}
                </Typography.Text>
              </Row>
              <Row>
                <Typography.Text strong>FPS:</Typography.Text>
                <Typography.Text>
                  {detailedResult.data?.frames_per_second}
                </Typography.Text>
              </Row>
            </>
          ) : null}
          <Divider />
          <Row>
            <Typography.Title level={2}>
              Metrics (FIXME Data only currently):{" "}
            </Typography.Title>
          </Row>
          {detailedResult.data == null
            ? null
            : Object.entries(detailedResult.data.metrics).map(
                ([metricName, metricValue]) => (
                  <Row>
                    <Typography.Text strong>{metricName}:</Typography.Text>
                    <Typography.Text>{metricValue}</Typography.Text>
                  </Row>
                )
              )}
        </Col>
        <Col span={12} style={{ padding: "10px" }}>
          <div ref={ref} style={{ width: "100%", height: "100%" }}>
            <ActiveViewLabelledImage
              visualization={visualization}
              width={width}
            />
          </div>
        </Col>
      </Row>
    </>
  );
}

export default ActiveViewSummaryModel;
