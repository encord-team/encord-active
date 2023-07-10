import * as React from "react";
import { Button, Pagination, Space } from "antd";
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
import ActiveCreateSubsetModal from "./modals/ActiveCreateSubsetModal";
import ActiveCreateTagModal from "./modals/ActiveCreateTagModal";

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
  const [actionModalOpen, setActionModalOpen] = useState<
    "subset" | "tag" | "workflow" | "download" | undefined
  >();

  const close = () => setActionModalOpen(undefined);

  return (
    <>
      <ActiveCreateSubsetModal
        open={actionModalOpen === "subset"}
        close={close}
        projectHash={projectHash}
        queryAPI={queryAPI}
      />
      <ActiveCreateTagModal
        open={actionModalOpen === "tag"}
        close={close}
        projectHash={projectHash}
        queryAPI={queryAPI}
      />
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
          onClick={() => setActionModalOpen("subset")}
          type="primary"
          icon={<PartitionOutlined />}
          disabled={selectedItems.size === 0}
        >
          Create project subset
        </Button>
        <Button
          onClick={() => setActionModalOpen("tag")}
          type="primary"
          icon={<SaveOutlined />}
          disabled={selectedItems.size === 0}
        >
          Save as tag
        </Button>
        <Button
          onClick={() => setActionModalOpen("workflow")}
          type="primary"
          icon={<NodeIndexOutlined />}
          disabled={selectedItems.size === 0}
        >
          Update workflow
        </Button>
        <Button
          onClick={() => setActionModalOpen("download")}
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
