import * as React from "react";
import { Form, Input, Modal } from "antd";
import { QueryAPI } from "../../Types";

export function CreateTagModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  queryAPI: QueryAPI;
}) {
  const { open, close, projectHash, queryAPI } = props;
  const [form] = Form.useForm<{
    tag_title: string;
    tag_description?: string | undefined;
  }>();

  const mutateCreatTag = queryAPI.useProjectMutationCreateTag(projectHash, {
    onSuccess: close,
  });

  return (
    <Modal
      open={open}
      title="Save Selection As Tag"
      okText="Create"
      onCancel={close}
      okButtonProps={{ loading: mutateCreatTag.isLoading }}
      onOk={() => {
        form
          .validateFields()
          .then((fields) =>
            mutateCreatTag.mutate({
              ...fields,
              objects: [],
            })
          )
          .catch(() => undefined);
      }}
    >
      <Form
        form={form}
        layout="vertical"
        name="create_selection_tag_form"
        initialValues={{ modifier: "public" }}
      >
        <Form.Item
          name="tag_title"
          label="Tag Title"
          rules={[{ required: true }]}
        >
          <Input />
        </Form.Item>
        <Form.Item name="tag_description" label="Tag Description">
          <Input type="textarea" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
