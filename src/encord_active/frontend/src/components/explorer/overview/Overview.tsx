import { InfoCircleOutlined, WarningOutlined } from "@ant-design/icons";
import { Space } from "antd";
import { useMemo } from "react";
import { isEmpty } from "radash";
import { useProjectAnalysisSummary } from "../../../hooks/queries/useProjectAnalysisSummary";
import { AnalysisDomain } from "../../../openapi/api";

type Props = {
  projectHash: string;
  analysisDomain: AnalysisDomain;
};
export function Overview({ projectHash, analysisDomain }: Props) {
  const summary = useProjectAnalysisSummary(projectHash, analysisDomain);
  const { data } = summary;
  // Derived: Total outliers
  const [totalSevereOutlier, totalFrames] = useMemo(() => {
    if (data == null || isEmpty(data.metrics)) {
      return [0, 0, 0];
    }

    const frames = data.count;
    const severe = Object.values(data.metrics)
      .map((metric) => metric?.severe ?? 0)
      .reduce((a, b) => a + b);

    return [severe, frames];
  }, [data]);

  return (
    <Space direction="vertical" size="large" className="p-4">
      <div className="gap-f flex flex-col">
        <div className="text-base text-gray-500">Total No. of frames</div>
        <div className="text-2xl">{totalFrames}</div>
      </div>
      <Space size="small" direction="vertical">
        <div className="text-base text-gray-500">
          Severe Outliers <InfoCircleOutlined />
        </div>
        <div className="text-2xl">
          <WarningOutlined className="text-severe" /> {totalSevereOutlier}
        </div>
      </Space>
    </Space>
  );
}
