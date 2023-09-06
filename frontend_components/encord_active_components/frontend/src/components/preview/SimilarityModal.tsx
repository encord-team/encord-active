import { Modal } from "antd";

function SimilarityModal(props: {
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
