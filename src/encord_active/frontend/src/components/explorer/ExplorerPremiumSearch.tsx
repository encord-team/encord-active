import { UploadOutlined } from "@ant-design/icons";
import { Button, Tooltip } from "antd";

export function ExplorerPremiumSearch() {

  return (
    <Tooltip overlay="Text Search">
      <div className="bg-[#F5F5F5] mr-2 flex h-full items-center gap-2 border-r border-gray-200 px-4">
        <input
          placeholder="Search Anything"
          disabled
        />
        <Button
          disabled
          className="border-none shadow-none disabled:bg-[#F5F5F5]"
          icon={<UploadOutlined />}
        />
      </div>
    </Tooltip>
  );
}
