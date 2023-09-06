import { LoadingOutlined } from "@ant-design/icons";

export const Spinner = (props: Parameters<typeof LoadingOutlined>[0]) => (
  <LoadingOutlined {...props} />
);
