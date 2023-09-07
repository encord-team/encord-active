import * as React from "react";
import { Modal } from "antd";
import { QueryContext } from "../../hooks/Context";

export function SimilarityModal(props: {
  queryContext: QueryContext;
  projectHash: string;
  predictionHash: string;
  similarityItem: string | undefined;
  onClose: () => void;
}) {
  const { onClose, similarityItem } = props;
  return (
    <Modal
      width="95vw"
      open={similarityItem !== undefined}
      onCancel={onClose}
    ></Modal>
  );
}
