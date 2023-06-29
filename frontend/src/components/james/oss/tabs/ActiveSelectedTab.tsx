import * as React from "react";
import {
  Button,
  Input,
  Modal,
  Pagination,
  Select,
  Space,
  Typography,
} from "antd";
import { useCallback, useMemo, useState } from "react";
import { Actions } from "usehooks-ts/dist/esm/useMap/useMap";
import {
  CloseCircleOutlined,
  DownloadOutlined,
  NodeIndexOutlined,
  PartitionOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import ActiveViewImageCard from "../view/ActiveViewImageCard";
import { ActiveQueryAPI } from "../ActiveTypes";

function getModalTitle(
  actionModelOpen: "subset" | "tag" | "workflow" | "download" | undefined
): string {
  if (actionModelOpen === "subset") {
    return "Create Project Subset";
  } else if (actionModelOpen === "tag") {
    return "Create Tag";
  } else if (actionModelOpen === "workflow") {
    return "Update Workflow State";
  } else if (actionModelOpen === "download") {
    return "Download Selection";
  } else {
    return "";
  }
}

function ActiveSelectedTab(props: {
  projectHash: string;
  queryAPI: ActiveQueryAPI;
  editUrl?:
    | undefined
    | ((dataHash: string, projectHash: string, frame: number) => string);
  itemWidth: number;
  selectedItems: Omit<Map<string, null>, "set" | "clear" | "delete">;
  setSelectedItems: Actions<string, null>;
}) {
  const {
    projectHash,
    editUrl,
    queryAPI,
    itemWidth,
    selectedItems,
    setSelectedItems,
  } = props;
  const [pageSize, setPageSize] = useState<number>(20);
  const [pageIdx, setPageIdx] = useState<number>(1);
  const sortedItems = useMemo(() => {
    const keys = Array.from(selectedItems.keys());
    keys.sort();
    return keys;
  }, [selectedItems]);
  const pagedItems = useMemo(
    () => ({
      total: sortedItems.length,
      results: sortedItems.slice(
        (pageIdx - 1) * pageSize,
        Math.min(sortedItems.length, pageIdx * pageSize)
      ),
    }),
    [sortedItems, pageIdx, pageSize]
  );

  const setSelectedItemBool = useCallback(
    (selectedKey: string, selected: boolean) => {
      if (selected) {
        setSelectedItems.set(selectedKey, null);
      } else {
        setSelectedItems.remove(selectedKey);
      }
    },
    [setSelectedItems]
  );
  const hasSelectedItemBool = useCallback(
    (selectedKey: string) => selectedItems.has(selectedKey),
    [selectedItems]
  );
  const [actionModelOpen, setActionModelOpen] = useState<
    "subset" | "tag" | "workflow" | "download" | undefined
  >();

  const [modalName, setModalName] = useState("");
  const [modalDescription, setModalDescription] = useState("");

  const getModalContent = () => {
    if (actionModelOpen === "subset" || actionModelOpen === "tag") {
      return (
        <>
          <Typography.Text strong>
            {actionModelOpen === "subset" ? "Project" : "Tag"} Name:
          </Typography.Text>
          <Input value={modalName} />
          <Typography.Text strong>
            {actionModelOpen === "subset" ? "Project" : "Tag"} Description:
          </Typography.Text>
          <Input />
          {actionModelOpen === "subset" ? (
            <>
              <Typography.Text strong>Dataset Name:</Typography.Text>
              <Input />
              <Typography.Text strong>Dataset Description:</Typography.Text>
              <Input />
            </>
          ) : null}
        </>
      );
    } else if (actionModelOpen === "workflow") {
      return null;
    } else if (actionModelOpen === "download") {
      return (
        <>
          <Typography.Text strong>Format:</Typography.Text>
          <Select
            options={[
              { label: "CSV", value: "csv" },
              { label: "COCO", value: "coco" },
            ]}
            style={{ width: 100 }}
          />
        </>
      );
    } else {
      return null;
    }
  };

  return (
    <>
      <Modal
        open={actionModelOpen != null}
        title={getModalTitle(actionModelOpen)}
        onCancel={() => setActionModelOpen(undefined)}
        onOk={() => {
          // setActionModelOpen(undefined);
          // FIXME: actually do something.
        }}
      >
        {getModalContent()}
      </Modal>
      <Space.Compact block style={{ marginBottom: 15 }}>
        <Button
          onClick={() => setSelectedItems.reset()}
          type="primary"
          icon={<CloseCircleOutlined />}
          disabled={selectedItems.size === 0}
        >
          Deselect all
        </Button>
        <Button
          onClick={() => setActionModelOpen("subset")}
          type="primary"
          icon={<PartitionOutlined />}
          disabled={selectedItems.size === 0}
        >
          Create project subset
        </Button>
        <Button
          onClick={() => setActionModelOpen("tag")}
          type="primary"
          icon={<SaveOutlined />}
          disabled={selectedItems.size === 0}
        >
          Save as tag
        </Button>
        <Button
          onClick={() => setActionModelOpen("workflow")}
          type="primary"
          icon={<NodeIndexOutlined />}
          disabled={selectedItems.size === 0}
        >
          Update workflow
        </Button>
        <Button
          onClick={() => setActionModelOpen("download")}
          type="primary"
          icon={<DownloadOutlined />}
          disabled={selectedItems.size === 0}
        >
          Download
        </Button>
      </Space.Compact>
      <Space wrap>
        {pagedItems.results.map((exploreItem) => {
          const exploreItemSegments = exploreItem.split("/");
          const duHash = exploreItemSegments[0] ?? "";
          const frame = parseInt(exploreItemSegments[1] ?? "");
          const objectHash = exploreItemSegments[2];
          return (
            <ActiveViewImageCard
              key={exploreItem}
              queryAPI={queryAPI}
              projectHash={projectHash}
              duHash={duHash}
              frame={frame}
              objectHash={objectHash}
              width={itemWidth}
              editUrl={editUrl}
              selectedKey={exploreItem}
              getSelected={hasSelectedItemBool}
              setSelected={setSelectedItemBool}
            />
          );
        })}
      </Space>
      <Pagination
        total={pagedItems.total}
        showSizeChanger
        showQuickJumper
        pageSize={pageSize}
        current={pageIdx}
        onChange={(page, pageSize) => {
          setPageIdx(page);
          setPageSize(pageSize);
        }}
      />
    </>
  );
}

export default ActiveSelectedTab;
