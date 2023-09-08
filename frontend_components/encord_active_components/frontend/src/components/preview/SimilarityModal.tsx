import * as React from "react";
import { Modal } from "antd";

export function SimilarityModal(props: {
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
