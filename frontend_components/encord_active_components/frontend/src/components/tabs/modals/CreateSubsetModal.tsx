import * as React from "react";
import { Form, Input, Modal } from "antd";
import { useProjectMutationCreateSubset } from "../../../hooks/mutation/useProjectMutationCreateSubset";
import { SearchFilters } from "../../../openapi/api";

export function CreateSubsetModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  filters: SearchFilters;
}) {
  const { open, close, projectHash, filters } = props;
  const [form] = Form.useForm<{
    project_title: string;
    project_description?: string | undefined;
    dataset_title: string;
    dataset_description?: string | undefined;
  }>();

  const mutateCreateSubset = useProjectMutationCreateSubset(
    projectHash
  );

  return (
    <Modal
      open={open}
      title="Create Subset Project"
      okText="Create"
      onCancel={close}
      okButtonProps={{
        loading: mutateCreateSubset.isLoading,
        style: { backgroundColor: "#5555ff" },
      }}
      cancelButtonProps={{ disabled: mutateCreateSubset.isLoading }}
      onOk={() => {
        form
          .validateFields()
          .then((fields) =>
            mutateCreateSubset.mutate({
              ...fields,
              filters,
            })
          )
          .then(close)
          .catch(() => undefined);
      }}
    >
      <Form
        form={form}
        layout="vertical"
        name="create_project_subset_form"
        initialValues={{ modifier: "public" }}
      >
        <Form.Item
          name="project_title"
          label="Project Title"
          rules={[{ required: true }]}
        >
          <Input />
        </Form.Item>
        <Form.Item name="project_description" label="Project Description">
          <Input type="textarea" />
        </Form.Item>
        <Form.Item
          name="dataset_title"
          label="Dataset Title"
          rules={[{ required: true }]}
        >
          <Input />
        </Form.Item>
        <Form.Item name="dataset_description" label="Dataset Description">
          <Input type="textarea" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
