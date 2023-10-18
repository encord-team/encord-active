import { Modal, notification } from "antd";
import { useMemo } from "react";
import { ProjectTagEntryMeta } from "../../../openapi/api";
import { useProjectMutationDeleteCollection } from "../../../hooks/mutation/useProjectMutationDeleteCollection";

export function DeleteCollectionModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  tags: ProjectTagEntryMeta[];
}) {
  const { open, close, projectHash, tags } = props;

  const {
    mutateAsync: deleteCollection,
    isLoading: isMutatingDeleteCollection,
  } = useProjectMutationDeleteCollection(projectHash);

  const handleDelete = useMemo(
    () => async () => {
      try {
        await deleteCollection(tags.map((tag) => tag.hash));
        notification.success({
          message: "Deleted successfully",
          description: `Collection deleted successfully`,
          placement: "bottomRight",
          duration: 5,
        });
      } catch (e) {
        notification.error({
          message: "Error",
          description: "Error deleting collection",
          placement: "bottomRight",
          duration: 5,
        });
      }
    },
    [tags, deleteCollection]
  );

  return (
    <Modal
      open={open}
      title={`Delete ${tags.length} Collection${tags.length > 1 ? "s" : ""}`}
      okText="Delete"
      onCancel={close}
      okButtonProps={{
        loading: isMutatingDeleteCollection,
        style: { backgroundColor: "#5555ff" },
      }}
      cancelButtonProps={{ disabled: isMutatingDeleteCollection }}
      onOk={() => {
        handleDelete();
        close();
      }}
    >
      <div className="text-xs text-gray-8">
        This will permanently delete the curation of data in collecitons.
        Don&#39;t worry, this will not delete the data.
      </div>
    </Modal>
  );
}
