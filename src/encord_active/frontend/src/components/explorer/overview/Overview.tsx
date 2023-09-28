import { InfoCircleOutlined } from "@ant-design/icons";
import { Space } from "antd";

type Props = {
  totalFrames: number;
};
export function Overview({ totalFrames }: Props) {
  return (
    <Space direction="vertical" size="large" className="p-4">
      <div className="gap-f flex flex-col">
        <div className="text-base text-gray-500">Total No. of frames</div>
        <div className="text-2xl">{totalFrames}</div>
      </div>
      <Space size="small" direction="vertical">
        <div className="text-base text-gray-500">
          Data Quality Score <InfoCircleOutlined />
        </div>
        <div className="text-2xl">50</div>
      </Space>
      <Space size="small" direction="vertical">
        <div className="text-base text-gray-500">
          Issue Types <InfoCircleOutlined />
        </div>
        <div className="text-sm">Duplicate</div>
        <div className="text-sm">Blur</div>
        <div className="text-sm">Dark</div>
        <div className="text-sm">Bright</div>
      </Space>
    </Space>
  );
}
