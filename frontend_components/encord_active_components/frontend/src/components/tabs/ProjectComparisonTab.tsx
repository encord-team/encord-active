import * as React from "react";
import { Col, Divider, Row, Select, Typography } from "antd";
import { useState } from "react";
import { ChartMetricCompareScatter } from "../charts/ChartMetricCompareScatter";
import { ChartMetricDissimilarity } from "../charts/ChartMetricDissimilarity";
import { ProjectDomainSummary } from "../../openapi/api";
import { useProjectList } from "../../hooks/queries/useListProjects";

export function ProjectComparisonTab(props: {
  projectHash: string;
  dataMetricsSummary: ProjectDomainSummary;
  annotationMetricsSummary: ProjectDomainSummary;
}) {
  const { projectHash, dataMetricsSummary, annotationMetricsSummary } = props;
  const allProjects = useProjectList();
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
                allProjects.data?.projects
                  ?.filter((project) => project.project_hash !== projectHash)
                  ?.map((project) => ({
                    label: project.title,
                    value: project.project_hash,
                  })) ?? []
              }
            />
          </Row>
        </Col>
        <Col span={12}>
          <Typography.Text strong>Scope: </Typography.Text>
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
          <ChartMetricCompareScatter
            metricsSummary={metricsSummary}
            analysisDomain={domain}
            projectHash={projectHash}
            compareProjectHash={compareProjectHash}
          />
          <Divider>
            <Typography.Title level={3}>Metric Dissimilarity</Typography.Title>
          </Divider>
          <ChartMetricDissimilarity
            projectHash={projectHash}
            analysisDomain={domain}
            metricsSummary={metricsSummary}
            compareProjectHash={compareProjectHash}
          />
        </>
      )}
    </>
  );
}
