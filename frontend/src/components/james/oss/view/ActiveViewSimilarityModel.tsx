import * as React from "react";
import { Col, Row, Space, Spin, Typography } from "antd";
import ActiveViewLabelledImage from "./ActiveViewLabelledImage";
import { ActiveProjectPreviewItemResult, ActiveQueryAPI } from "../ActiveTypes";

type ActiveViewImageCardProps = {
  queryAPI: ActiveQueryAPI;
  projectHash: string;
  duHash: string;
  frame: number;
  objectHash?: string | undefined;
  width?: number;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  selectedKey?: string;
  getSelected?: ((selectedKey: string) => boolean) | undefined;
  setSelected?: ((selectedKey: string, selected: boolean) => void) | undefined;
};

function ActiveViewSummaryModel(props: {
  queryAPI: ActiveQueryAPI;
  projectHash: string;
  duHash: string;
  frame: number;
  objectHash?: string | undefined;
  visualization: ActiveProjectPreviewItemResult;
  ImageCardClass: (props: ActiveViewImageCardProps) => React.ReactElement;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  getSelected?: ((selectedKey: string) => boolean) | undefined;
  setSelected?: ((selectedKey: string, selected: boolean) => void) | undefined;
}) {
  const {
    queryAPI,
    projectHash,
    duHash,
    frame,
    objectHash,
    visualization,
    ImageCardClass,
    editUrl,
    setSelected,
    getSelected,
  } = props;
  const similarity = queryAPI.useProjectItemSimilarity(
    projectHash,
    duHash,
    frame,
    objectHash
  );
  return (
    <>
      <Typography.Title level={2}>Similarity search</Typography.Title>
      <Row>
        <Col span={12} style={{ padding: "10px" }}>
          <Typography.Title level={3}>Source image</Typography.Title>
          <ActiveViewLabelledImage visualization={visualization} width={200} />
        </Col>
        <Col span={12}>
          <Space wrap>
            {similarity.data == null ? (
              <Spin />
            ) : (
              similarity.data.results.map((result) => {
                const key = `${result.du_hash}/${result.frame}/${
                  result.object_hash ?? ""
                }`;
                return (
                  <ImageCardClass
                    key={key}
                    queryAPI={queryAPI}
                    projectHash={projectHash}
                    duHash={result.du_hash}
                    frame={result.frame}
                    objectHash={result.object_hash}
                    width={100}
                    editUrl={editUrl}
                    selectedKey={key}
                    setSelected={setSelected}
                    getSelected={getSelected}
                  />
                );
              })
            )}
          </Space>
        </Col>
      </Row>
    </>
  );
}

export default ActiveViewSummaryModel;
