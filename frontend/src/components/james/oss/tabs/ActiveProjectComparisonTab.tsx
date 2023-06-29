import * as React from "react";
import { Divider, Row, Select, Space, Typography } from "antd";
import { useState } from "react";
import {
  ActiveProjectAnalysisDomain,
  ActiveProjectMetricSummary,
  ActiveQueryAPI,
} from "../ActiveTypes";
import ActiveChartMetricCompareScatter from "../charts/ActiveChartMetricCompareScatter";

function ActiveProjectComparisonTab(props: {
  projectHash: string;
  queryAPI: ActiveQueryAPI;
  metricsSummary: ActiveProjectMetricSummary;
  analysisDomain: ActiveProjectAnalysisDomain;
}) {
  const { projectHash, queryAPI, metricsSummary, analysisDomain } = props;
  const allProjects = queryAPI.useListProjectViews("", 0, 100000);
  const [compareProjectHash, setCompareProjectHash] = useState<
    undefined | string
  >();
  return (
    <>
      <Row>
        <Space align="center" wrap>
          <Typography.Text strong>Comparison Project: </Typography.Text>
          <Select
            onChange={setCompareProjectHash}
            value={compareProjectHash}
            options={
              allProjects.data?.results?.map((project) => ({
                label: project.title,
                value: project.project_hash,
              })) ?? []
            }
          />
        </Space>
      </Row>
      {compareProjectHash === undefined ? null : (
        <>
          <Divider>
            <Typography.Title level={3}>Metric Comparison</Typography.Title>
          </Divider>
          <ActiveChartMetricCompareScatter
            metricsSummary={metricsSummary}
            analysisDomain={analysisDomain}
            projectHash={projectHash}
            queryAPI={queryAPI}
            compareProjectHash={compareProjectHash}
          />
          <Divider>
            <Typography.Title level={3}>Metric Dissimilarity</Typography.Title>
          </Divider>
          TODO: implement this!!!!
        </>
      )}
    </>
  );
}

export default ActiveProjectComparisonTab;
