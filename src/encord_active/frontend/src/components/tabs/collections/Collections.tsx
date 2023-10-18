import { Button, Popover, Table } from "antd";
import { Key, TableRowSelection } from "antd/es/table/interface";
import {
  Dispatch,
  SetStateAction,
  useCallback,
  useMemo,
  useState,
} from "react";
import { DatabaseOutlined, MoreOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router";
import { GrMultiple } from "react-icons/gr";
import { useProjectListTagsMeta } from "../../../hooks/queries/useProjectListTagsMeta";
import { getReadableDatetime } from "../../../utils/time";
import { FilterState } from "../../util/MetricFilter";
import { CreateSubsetModal } from "../modals/CreateSubsetModal";
import { DeleteCollectionModal } from "../modals/DeleteCollectionModal";
import { ProjectTagEntryMeta } from "../../../openapi/api";
import { ModalName } from "../../Types";
import { EditCollectionModal } from "../modals/EditCollectionModal";

const columns = [
  {
    title: "Name",
    dataIndex: "name",
  },
  {
    title: "Data Units",
    dataIndex: "dataUnits",
  },
  {
    title: "Label Units",
    dataIndex: "labelUnits",
  },
  {
    title: "Created at",
    dataIndex: "createdAt",
  },
  {
    title: "Updated at",
    dataIndex: "updatedAt",
  },
];

type Collection = {
  key: string;
  name: JSX.Element;
  dataUnits: number;
  labelUnits: number;
  createdAt?: string;
  updatedAt?: JSX.Element;
};

type props = {
  projectHash: string;
  dataFilters: FilterState;
  setDataFilters: (filterState: FilterState) => void;
  openModal: ModalName | undefined;
  setOpenModal: Dispatch<SetStateAction<ModalName | undefined>>;
};

export function Collections({
  projectHash,
  dataFilters,
  setDataFilters,
  openModal,
  setOpenModal,
}: props) {
  // existing tags
  const { data: projectTagsMeta = [] } = useProjectListTagsMeta(projectHash);
  const [tagsToDelete, setTagsToDelete] = useState<ProjectTagEntryMeta[]>([]);
  const [createSubsetModal, setCreateSubsetModal] = useState(false);
  const [deleteCollectionModal, setDeleteCollectionModal] = useState(false);

  // Modal state
  const close = () => setOpenModal(undefined);

  const handleDelete = useCallback(
    (tagHash: string) => {
      setTagsToDelete([projectTagsMeta.filter((p) => p.hash === tagHash)[0]]);
      setDeleteCollectionModal(true);
    },
    [projectTagsMeta, setTagsToDelete, setDeleteCollectionModal]
  );

  const handleDeleteMultiple = useCallback(
    (tagHashList: string[]) => {
      setTagsToDelete(
        projectTagsMeta.filter((p) => tagHashList.includes(p.hash))
      );
      setDeleteCollectionModal(true);
    },
    [projectTagsMeta, setTagsToDelete, setDeleteCollectionModal]
  );

  const [openCollectionMenu, setOpenCollectionMenu] = useState<
    string | undefined
  >(undefined);

  const [collectionToEdit, setCollectionToEdit] =
    useState<ProjectTagEntryMeta>();
  const navigate = useNavigate();
  const handleViewInExplorer = useCallback(
    (tags: string[]) => {
      setDataFilters({
        ordering: ["tags"],
        metricFilters: {},
        enumFilters: {
          tags,
        },
      });
      setOpenCollectionMenu(undefined);
      navigate(`../projects/${projectHash}/explorer`, {});
    },
    [setDataFilters, setOpenCollectionMenu, navigate, projectHash]
  );

  const collections: Collection[] = useMemo<Collection[]>(
    () =>
      projectTagsMeta.map(
        ({
          hash,
          name,
          description,
          dataCount,
          labelCount,
          createdAt,
          updatedAt,
        }) => ({
          name: (
            <div>
              <div className="text-xs font-medium text-gray-9">
                <GrMultiple className="mr-1 inline" />
                {name}
              </div>
              <div className="text-xs font-medium text-gray-7">
                {description}
              </div>
            </div>
          ),
          key: hash,
          dataUnits: dataCount,
          labelUnits: labelCount,
          createdAt: createdAt
            ? getReadableDatetime(createdAt?.toLocaleString())
            : "",
          updatedAt: (
            <div className="flex justify-between">
              <div>
                {updatedAt
                  ? getReadableDatetime(updatedAt?.toLocaleString())
                  : ""}
              </div>
              <Button
                className="border-0 shadow-none"
                onClick={(e) => {
                  e.stopPropagation();
                }}
              >
                <Popover
                  open={openCollectionMenu === hash}
                  onOpenChange={(open) => {
                    setOpenCollectionMenu(open ? hash : undefined);
                  }}
                  trigger={["click"]}
                  placement="bottomLeft"
                  arrow={false}
                  content={
                    <div className="flex flex-col gap-1">
                      <Button
                        className="border-0 text-left shadow-none hover:bg-gray-100 hover:text-black"
                        onClick={() => {
                          handleViewInExplorer([hash]);
                        }}
                      >
                        View in Explorer
                      </Button>
                      <Button
                        className="border-0 p-0 text-left shadow-none"
                        onClick={(e) => {
                          e.stopPropagation();
                        }}
                      >
                        <Button
                          className="border-0 text-left shadow-none hover:bg-gray-100 hover:text-black"
                          onClick={() => {
                            const entry = projectTagsMeta.filter(
                              (p) => p.hash === hash
                            )[0];
                            setCollectionToEdit(entry);
                            setOpenModal("editCollection");
                            setOpenCollectionMenu(undefined);
                          }}
                        >
                          Edit Details
                        </Button>
                      </Button>

                      <Button
                        className="border-0 text-left shadow-none hover:bg-gray-100 hover:text-black"
                        onClick={(e) => {
                          handleDelete(hash);
                          e.stopPropagation();
                        }}
                      >
                        Delete
                      </Button>
                    </div>
                  }
                >
                  <MoreOutlined onClick={() => setOpenCollectionMenu(hash)} />
                </Popover>
              </Button>
            </div>
          ),
        })
      ),
    [
      projectTagsMeta,
      openCollectionMenu,
      handleDelete,
      handleViewInExplorer,
      setOpenModal,
    ]
  );

  // table selection
  const [selectedRowKeys, setSelectedRowKeys] = useState<Key[]>([]);
  const onSelectChange = (
    newSelectedRowKeys: Key[],
    selectedRows: Collection[]
  ) => {
    setSelectedRowKeys(newSelectedRowKeys);
  };
  const rowSelection: TableRowSelection<Collection> = {
    selectedRowKeys,
    onChange: onSelectChange,
  };

  const hasSelected = selectedRowKeys.length > 0;

  const firstSelectedItem = hasSelected
    ? projectTagsMeta.filter((p) => p.hash === selectedRowKeys[0])[0]
    : undefined;

  const prefill = firstSelectedItem
    ? {
        project_title: firstSelectedItem.name,
        project_description: firstSelectedItem.description,
        dataset_title: firstSelectedItem.name,
        dataset_description: firstSelectedItem.description,
      }
    : {};

  return (
    <div className="">
      {collectionToEdit !== undefined && (
        <EditCollectionModal
          projectHash={projectHash}
          open={openModal === "editCollection"}
          close={close}
          tag={collectionToEdit}
        />
      )}
      <CreateSubsetModal
        prefill={prefill}
        open={createSubsetModal}
        close={() => setCreateSubsetModal(false)}
        projectHash={projectHash}
        filters={{
          data: {
            tags: selectedRowKeys.map((r) => r.toString()),
            metrics: {},
            enums: {},
          },
          annotation: { metrics: {}, enums: {} },
        }}
      />
      <DeleteCollectionModal
        open={deleteCollectionModal}
        close={() => setDeleteCollectionModal(false)}
        projectHash={projectHash}
        tags={tagsToDelete}
      />
      <div className="flex w-full justify-end gap-2 px-4 py-2">
        <Button
          disabled={!hasSelected}
          className="text-primary"
          onClick={() => {
            handleViewInExplorer(selectedRowKeys.map((r) => r.toString()));
          }}
        >
          View in Explorer
        </Button>
        <Button
          disabled={!hasSelected || selectedRowKeys.length > 1}
          icon={<DatabaseOutlined />}
          className="text-primary"
          onClick={() => {
            setCreateSubsetModal(true);
          }}
        >
          Create Dataset
        </Button>
        <Button
          danger
          disabled={!hasSelected}
          className="bg-transparent text-white"
          onClick={() => {
            handleDeleteMultiple(selectedRowKeys.map((r) => r.toString()));
          }}
        >
          Delete
        </Button>
      </div>
      <Table
        rowSelection={rowSelection}
        columns={columns}
        dataSource={collections}
        onRow={(record, rowIndex) => ({
          onClick: (event) => {
            if (selectedRowKeys.includes(record.key)) {
              setSelectedRowKeys(
                selectedRowKeys.filter((r) => r !== record.key)
              );
            } else {
              setSelectedRowKeys([...selectedRowKeys, record.key]);
            }
          }, // click row
          onContextMenu: (event) => {
            event.preventDefault();
            setOpenCollectionMenu(record.key);
          }, // right button click row
        })}
      />
    </div>
  );
}
