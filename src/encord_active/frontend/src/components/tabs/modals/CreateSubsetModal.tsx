import { useState } from "react";
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
  const [newProjectHash, setNewProjectHash] = useState<string>();

  const mutateCreateSubset = useProjectMutationCreateSubset(projectHash);

  if (newProjectHash) {
    return (
      <Modal
        open={open}
        title="Subset created successfully"
        okText="Close"
        okButtonProps={{ style: { backgroundColor: "#5555ff" } }}
        cancelButtonProps={{ hidden: true }}
        onOk={close}
      >
        Click{" "}
        <a
          className="link"
          href={`https://app.encord.com/projects/view/${newProjectHash}/summary`}
        >
          here
        </a>{" "}
        to go to the newly created annotation project
      </Modal>
    );
  }

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
            mutateCreateSubset.mutateAsync({
              ...fields,
              filters,
            })
          )
          .then((f) => setNewProjectHash(f.data))
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
