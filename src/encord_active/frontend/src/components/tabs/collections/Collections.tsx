import { Button, Table, TableProps } from "antd";
import { Key, TableRowSelection } from "antd/es/table/interface";
import { useMemo, useState } from "react";
import { DatabaseOutlined } from "@ant-design/icons";
import { useProjectListTagsMeta } from "../../../hooks/queries/useProjectListTagsMeta";
import { getReadableDatetime } from "../../../utils/time";
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
  updatedAt?: string;
};

type props = {
  projectHash: string;
  selectedItems: ReadonlySet<string> | "ALL";
};

export function Collections({ projectHash, selectedItems }: props) {
  //existing tags
  const { data: projectTagsMeta = [] } = useProjectListTagsMeta(projectHash);
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
          lastEditedAt,
        }) => ({
          name: (
            <div>
              <div className="text-xs font-medium text-gray-9">{name}</div>
              <div className="text-xs font-medium text-gray-7">
                {description}
              </div>
            </div>
          ),
          key: hash,
          dataUnits: dataCount,
          labelUnits: labelCount,
          createdAt: getReadableDatetime(createdAt?.toLocaleString()),
          updatedAt: getReadableDatetime(lastEditedAt?.toLocaleString()),
        })
      ),
    [projectTagsMeta]
  );

  //table selection
  const [selectedRowKeys, setSelectedRowKeys] = useState<Key[]>([]);
  const onSelectChange = (
    newSelectedRowKeys: Key[],
    selectedRows: Collection[]
  ) => {
    console.log("selectedRowKeys changed: ", newSelectedRowKeys);
    setSelectedRowKeys(newSelectedRowKeys);
  };
  const rowSelection: TableRowSelection<Collection> = {
    selectedRowKeys,
    onChange: onSelectChange,
  };
  const hasSelected = selectedRowKeys.length > 0;
  return (
    <div className="">
      <div className="flex w-full justify-end gap-2 px-4 py-2">
        <Button
          disabled={!hasSelected}
          icon={<DatabaseOutlined />}
          className="bg-primary text-white"
          type="primary"
        >
          Create Dataset
        </Button>
        <Button disabled={!hasSelected} icon={<DatabaseOutlined />}>
          Delete
        </Button>
      </div>
      <Table
        rowSelection={rowSelection}
        columns={columns}
        dataSource={collections}
      />
    </div>
  );
}
