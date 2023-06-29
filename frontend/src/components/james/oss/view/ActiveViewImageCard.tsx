import * as React from "react";
import { Button, Card, Modal } from "antd";
import {
  EditOutlined,
  FullscreenOutlined,
  SearchOutlined,
  TagOutlined,
  TagTwoTone,
} from "@ant-design/icons";
import Meta from "antd/es/card/Meta";
import { CSSProperties, useState } from "react";
import ActiveViewLabelledImage from "./ActiveViewLabelledImage";
import { ActiveQueryAPI } from "../ActiveTypes";
import ActiveViewSummaryModel from "./ActiveViewSummaryModel";
import ActiveViewSimilarityModel from "./ActiveViewSimilarityModel";

function ActiveViewImageCard(props: {
  queryAPI: ActiveQueryAPI;
  projectHash: string;
  duHash: string;
  frame: number;
  objectHash?: string | undefined;
  editUrl?:
    | ((dataHash: string, projectHash: string, frame: number) => string)
    | undefined;
  selectedKey?: string;
  getSelected?: ((selectedKey: string) => boolean) | undefined;
  setSelected?: ((selectedKey: string, selected: boolean) => void) | undefined;
  width?: number | undefined;
  iconStyle?: CSSProperties | undefined;
  title?: React.ReactNode | undefined;
  description?: React.ReactNode | undefined;
  style?: React.CSSProperties | undefined;
}) {
  const {
    queryAPI,
    projectHash,
    duHash,
    frame,
    objectHash,
    editUrl,
    selectedKey,
    getSelected,
    setSelected,
    width,
    iconStyle,
    title,
    description,
    style,
  } = props;
  const [modelOpen, setModelOpen] = useState<null | "detail" | "similarity">(
    null
  );
  const { data: visualization } = queryAPI.useProjectItemPreview(
    projectHash,
    duHash,
    frame,
    objectHash
  );

  const getTag = () => {
    if (
      getSelected === undefined ||
      selectedKey === undefined ||
      setSelected === undefined
    ) {
      return null;
    }
    if (getSelected(selectedKey)) {
      return (
        <TagTwoTone key="tag" onClick={() => setSelected(selectedKey, false)} />
      );
    } else {
      return (
        <TagOutlined key="tag" onClick={() => setSelected(selectedKey, true)} />
      );
    }
  };

  const getModel = () => {
    if (modelOpen == null || visualization == null) {
      return null;
    }
    if (modelOpen === "detail") {
      return (
        <ActiveViewSummaryModel
          visualization={visualization}
          queryAPI={queryAPI}
          projectHash={projectHash}
          duHash={duHash}
          frame={frame}
          objectHash={objectHash}
        />
      );
    } else if (modelOpen === "similarity") {
      return (
        <ActiveViewSimilarityModel
          visualization={visualization}
          queryAPI={queryAPI}
          projectHash={projectHash}
          duHash={duHash}
          frame={frame}
          objectHash={objectHash}
          ImageCardClass={ActiveViewImageCard}
          setSelected={setSelected}
          getSelected={getSelected}
        />
      );
    } else {
      return null;
    }
  };

  return (
    <Card
      style={{ width: "fit-content", ...style }}
      headStyle={{ padding: "0" }}
      loading={visualization == null}
      hoverable
      cover={
        visualization == null ? null : (
          <>
            <ActiveViewLabelledImage
              visualization={visualization}
              width={width}
            />
            <Modal
              open={modelOpen != null}
              title={title}
              onCancel={() => setModelOpen(null)}
              width="90vw"
              footer={
                modelOpen === "detail"
                  ? [
                      <Button
                        key="edit"
                        href={
                          editUrl
                            ? editUrl(duHash, projectHash, frame)
                            : "_blank"
                        }
                        disabled={editUrl == null}
                        type="primary"
                        icon={<EditOutlined key="edit" />}
                      >
                        Edit
                      </Button>,
                    ]
                  : null
              }
            >
              {getModel()}
            </Modal>
          </>
        )
      }
      actions={[
        <FullscreenOutlined
          key="fullscreen"
          style={iconStyle}
          onClick={() => setModelOpen("detail")}
        />,
        <EditOutlined
          key="edit"
          style={editUrl ? iconStyle : { ...iconStyle, color: "lightgrey" }}
          href={editUrl ? editUrl(duHash, projectHash, frame) : "_blank"}
          type="small"
        />,
        <SearchOutlined
          key="similar"
          style={iconStyle}
          onClick={() => setModelOpen("similarity")}
        />,
        getTag(),
      ]}
      bodyStyle={{ padding: "0" }}
    >
      {title != null || description != null ? (
        <Meta title={title} description={description} />
      ) : null}
    </Card>
  );
}

export default ActiveViewImageCard;
