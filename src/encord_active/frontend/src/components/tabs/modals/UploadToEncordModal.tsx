import { Form, Input, Modal } from "antd";
import { useProjectMutationUploadToEncord } from "../../../hooks/mutation/useProjectMutationUploadToEncord";

export function UploadToEncordModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  setSelectedProjectHash: (key: string | undefined) => void;
}) {
  const { open, close, projectHash, setSelectedProjectHash } = props;
  const [form] = Form.useForm<{
    dataset_title: string;
    dataset_description?: string | undefined;
    project_title: string;
    project_description?: string | undefined;
    ontology_title: string;
    ontology_description?: string | undefined;
  }>();

  const mutateCreateSubset = useProjectMutationUploadToEncord(projectHash);

  return (
    <Modal
      open={open}
      title="Upload Project to Encord"
      okText="Upload"
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
            })
          )
          .then((a) => setSelectedProjectHash(a.data))
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
        <Form.Item
          name="ontology_title"
          label="Ontology Title"
          rules={[{ required: true }]}
        >
          <Input />
        </Form.Item>
        <Form.Item name="ontology_description" label="Ontology Description">
          <Input type="textarea" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
