import { Form, Input, Modal, notification } from "antd";
import TextArea from "antd/es/input/TextArea";
import { useCallback } from "react";
import { ProjectTagEntryMeta } from "../../../openapi/api";
import { useProjectMutationUpdateTag } from "../../../hooks/mutation/useProjectMutationUpdateTag";

export function EditCollectionModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  tag: ProjectTagEntryMeta;
}) {
  const { open, close, projectHash, tag } = props;

  const [editCollectionForm] = Form.useForm<{
    collection_title: string;
    collection_description?: string | undefined;
  }>();

  editCollectionForm.setFieldsValue({
    collection_title: tag.name,
    collection_description: tag.description,
  });

  const { mutateAsync: updateTag, isLoading: isMutatingUpdateTag } =
    useProjectMutationUpdateTag(projectHash, tag.hash);

  const isMutating = isMutatingUpdateTag;

  const handleUpdate = useCallback(async () => {
    try {
      await updateTag({
        name: editCollectionForm.getFieldValue("collection_title"),
        description: editCollectionForm.getFieldValue("collection_description"),
      });

      notification.success({
        message: "Updated Successfully",
        description: "Collection updated successfully",
        placement: "bottomRight",
        duration: 5,
      });
      close();
    } catch (e) {
      notification.error({
        message: "Error",
        description: "Error updating collection details",
        placement: "bottomRight",
        duration: 5,
      });
    }
  }, [editCollectionForm, updateTag, close]);

  return (
    <div>
      <Modal
        open={open}
        title="Add items to Collection"
        okText="Submit"
        onCancel={close}
        okButtonProps={{
          loading: isMutating,
          style: { backgroundColor: "#5555ff" },
        }}
        cancelButtonProps={{ disabled: isMutating }}
        onOk={handleUpdate}
      >
        <div className="flex flex-col gap-4">
          <div>
            <Form
              form={editCollectionForm}
              layout="vertical"
              name="create_project_subset_form"
            >
              <Form.Item
                name="collection_title"
                label="Collection Title"
                rules={[{ required: true }]}
              >
                <Input />
              </Form.Item>
              <Form.Item
                name="collection_description"
                label="Collection Description"
              >
                <TextArea autoSize={{ minRows: 2, maxRows: 6 }} />
              </Form.Item>
            </Form>
          </div>
        </div>
      </Modal>
    </div>
  );
}
