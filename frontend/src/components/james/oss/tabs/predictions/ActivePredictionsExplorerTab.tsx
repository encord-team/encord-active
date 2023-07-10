import {
  Button,
  Checkbox,
  Divider,
  Pagination,
  Popover,
  Row,
  Space,
  Typography,
} from "antd";
import * as React from "react";
import { useState } from "react";
import { FilterOutlined } from "@ant-design/icons";
import ActiveMetricFilter, {
  ActiveFilterState,
} from "../../util/ActiveMetricFilter";

function ActivePredictionsExplorerTab(props: {
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { featureHashMap } = props;
  const [pageSize, setPageSize] = useState<number>(20);
  const [pageIdx, setPageIdx] = useState<number>(1);
  const [showTruePositive, setShowTruePositive] = useState(true);
  const [showFalsePositive, setShowFalsePositive] = useState(true);
  const [showFalseNegative, setShowFalseNegative] = useState(false);
  const [filters, setFilters] = useState<ActiveFilterState>({
    metricFilters: {},
    enumFilters: {},
    ordering: [],
  });
  const metricsSummary = { metrics: {}, enums: {} };
  const metricRanges = {};

  return (
    <>
      <Space.Compact block style={{ marginBottom: 15 }}>
        <Popover
          placement="bottomLeft"
          content={
            <ActiveMetricFilter
              filters={filters}
              setFilters={setFilters}
              metricsSummary={metricsSummary}
              metricRanges={metricRanges}
              featureHashMap={featureHashMap}
            />
          }
          trigger="click"
        >
          <Button type="primary" icon={<FilterOutlined />}>
            Filters
          </Button>
        </Popover>
      </Space.Compact>
      <Row>
        <Typography.Text strong>True positive: </Typography.Text>
        <Checkbox
          value={showTruePositive}
          onChange={() => setShowTruePositive(!showTruePositive)}
        />
        <Typography.Text strong>False positive: </Typography.Text>
        <Checkbox
          value={showFalsePositive}
          onChange={() => setShowFalsePositive(!showFalsePositive)}
        />
        <Typography.Text strong>False negative: </Typography.Text>
        <Checkbox
          value={showFalseNegative}
          onChange={() => setShowFalseNegative(!showFalseNegative)}
        />
      </Row>
      <Divider />
      <Space wrap>FIXME: actually implement search results here.</Space>
      <Pagination
        style={{ marginTop: 10 }}
        total={1}
        showSizeChanger
        showQuickJumper
        pageSize={pageSize}
        current={pageIdx}
        onChange={(page, pageSize) => {
          setPageIdx(page);
          setPageSize(pageSize);
        }}
        // showTotal={(total, range) =>
        // `${total} results ${7 === 1 ? " (Truncated)" : ""}`
        // }
      />
    </>
  );
}

export default ActivePredictionsExplorerTab;
