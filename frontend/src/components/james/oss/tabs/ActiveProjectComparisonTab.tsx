import * as React from "react";
import { Col, Divider, Row, Select, Typography } from "antd";
import { useState } from "react";
import { ActiveProjectMetricSummary, ActiveQueryAPI } from "../ActiveTypes";
import ActiveChartMetricCompareScatter from "../charts/ActiveChartMetricCompareScatter";
import ActiveChartMetricDissimilarity from "../charts/ActiveChartMetricDissimilarity";

function ActiveProjectComparisonTab(props: {
  projectHash: string;
  queryAPI: ActiveQueryAPI;
  dataMetricsSummary: ActiveProjectMetricSummary;
  annotationMetricsSummary: ActiveProjectMetricSummary;
}) {
  const {
    projectHash,
    queryAPI,
    dataMetricsSummary,
    annotationMetricsSummary,
  } = props;
  const allProjects = queryAPI.useListProjectViews("", 0, 100000);
  const [compareProjectHash, setCompareProjectHash] = useState<
    undefined | string
  >();

  const [domain, setDomain] = useState<"annotation" | "data">("data");
  const metricsSummary =
    domain === "data" ? dataMetricsSummary : annotationMetricsSummary;

  return (
    <>
      <Row align="middle">
        <Col span={12}>
          <Row align="middle">
            <Typography.Text strong>Comparison Project: </Typography.Text>
            <Select
              onChange={setCompareProjectHash}
              value={compareProjectHash}
              style={{ width: 300 }}
              options={
                allProjects.data?.results?.map((project) => ({
                  label: project.title,
                  value: project.project_hash,
                })) ?? []
              }
            />
          </Row>
        </Col>
        <Col span={12}>
          <Typography.Text strong>Domain: </Typography.Text>
          <Select
            onChange={setDomain}
            value={domain}
            style={{ width: 200 }}
            options={[
              { value: "data", label: "Data" },
              { value: "annotation", label: "Annotation" },
            ]}
          />
        </Col>
      </Row>
      {compareProjectHash === undefined ? null : (
        <>
          <Divider>
            <Typography.Title level={3}>Metric Comparison</Typography.Title>
          </Divider>
          <ActiveChartMetricCompareScatter
            metricsSummary={metricsSummary}
            analysisDomain={domain}
            projectHash={projectHash}
            queryAPI={queryAPI}
            compareProjectHash={compareProjectHash}
          />
          <Divider>
            <Typography.Title level={3}>Metric Dissimilarity</Typography.Title>
          </Divider>
          <ActiveChartMetricDissimilarity
            projectHash={projectHash}
            analysisDomain={domain}
            queryAPI={queryAPI}
            metricsSummary={metricsSummary}
            compareProjectHash={compareProjectHash}
          />
        </>
      )}
    </>
  );
}

export default ActiveProjectComparisonTab;
