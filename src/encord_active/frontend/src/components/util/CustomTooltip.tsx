import React from "react";
import { Tooltip } from "antd";

export function CustomTooltip(props: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  const { title, description, children } = props;

  return (
    <Tooltip
      overlay={
        <div className="flex flex-col divide-y divide-gray-7">
          <div className="p-1 text-sm font-semibold">{title}</div>
          <div className="p-1 text-sm">{description}</div>
        </div>
      }
    >
      {children}
    </Tooltip>
  );
}
