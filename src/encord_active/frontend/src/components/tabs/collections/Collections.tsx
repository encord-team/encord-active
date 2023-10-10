import { Button, Table, TableProps } from "antd";
import { Key, TableRowSelection } from "antd/es/table/interface";
import { useMemo, useState } from "react";
import { AnalysisDomain, AnnotationType } from "../../../openapi/api";
import { DatabaseOutlined } from "@ant-design/icons";
import { useProjectItemsListTags } from "../../../hooks/queries/useProjectItemsListTags";
import { useProjectListTags } from "../../../hooks/queries/useProjectListTags";
import { useProjectListTagsMeta } from "../../../hooks/queries/useProjectListTagsMeta";
const columns = [
  {
    title: "Name",
    dataIndex: "name",
  },
  {
    title: "Type",
    dataIndex: "type",
  },
  {
    title: "Data Units",
    dataIndex: "dataUnits",
  },
  {
    title: "Label Units",
    dataIndex: "labelUnits",
  },
];

type Collection = {
  key: string;
  name: string;
  type: AnalysisDomain;
  dataUnits: number;
  labelUnits: number;
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
      projectTagsMeta.map(({ hash, name, dataCount, labelCount }) => ({
        name: name,
        key: hash,
        type: AnalysisDomain.Annotation,
        dataUnits: dataCount,
        labelUnits: labelCount,
      })),
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
