import { Checkbox, Form, Input, Modal, Segmented, notification } from "antd";
import { useCallback, useMemo, useState } from "react";
import { CheckboxValueType } from "antd/es/checkbox/Group";
import TextArea from "antd/es/input/TextArea";
import { toDataItemID } from "../../util/ItemIdUtil";
import {
  AnalysisDomain,
  ProjectTagEntry,
  SearchFilters,
} from "../../../openapi/api";
import { useProjectListTags } from "../../../hooks/queries/useProjectListTags";
import { useProjectMutationItemsAddTag } from "../../../hooks/mutation/useProjectMutationItemsAddTag";
import { useProjectMutationCreateTag } from "../../../hooks/mutation/useProjectMutationCreateTag";
import { useProjectMutationFiltersAddTag } from "../../../hooks/mutation/useProjectMutationFiltersAddTag";

export function AddToCollectionModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  selectedItems: ReadonlySet<string> | "ALL";
  filtersDomain: AnalysisDomain;
  filters: SearchFilters;
}) {
  const { open, close, projectHash, selectedItems, filters, filtersDomain } =
    props;
  const { data: projectTags = [] } = useProjectListTags(projectHash);
  const CheckboxGroup = Checkbox.Group;
  const [checkedList, setCheckedList] = useState<ProjectTagEntry[]>([]);
  const onChange = (list: CheckboxValueType[]) => {
    const newList = projectTags.filter((item) => list.includes(item.hash));
    setCheckedList(newList);
  };

  const { mutateAsync: itemsAddTags, isLoading: isMutatingItemsAdd } =
    useProjectMutationItemsAddTag(projectHash);

  const { mutateAsync: filtersAddTags, isLoading: isMutatingFiltersAdd } =
    useProjectMutationFiltersAddTag(projectHash);

  const getDomainItems = useCallback((items: ReadonlySet<string>): string[] => {
    const dataItems = new Set([...items].map(toDataItemID));

    return [...dataItems];
  }, []);

  const [newCollectionForm] = Form.useForm<{
    collection_title: string;
    collection_description?: string | undefined;
  }>();

  // Reset logic
  const closeModal = useCallback(() => {
    newCollectionForm.resetFields();
    checkedList.length = 0;
    setExistingOrNew("existing");
    close();
  }, [newCollectionForm, checkedList, close]);

  const addTags = useCallback(
    async (tagList: string[]) => {
      if (selectedItems === "ALL") {
        await filtersAddTags({ domain: filtersDomain, filters, tags: tagList });
      } else {
        await itemsAddTags({
          items: getDomainItems(selectedItems),
          tags: tagList,
        });
      }
      closeModal();
      notification.success({
        message: "Items added",
        description: "Items added to collection successfully.",
        placement: "bottomRight",
        duration: 5,
      });
    },
    [
      selectedItems,
      filtersAddTags,
      filtersDomain,
      filters,
      itemsAddTags,
      getDomainItems,
      closeModal,
    ]
  );

  type ExistingOrNew = "existing" | "new";
  const [existingOrNew, setExistingOrNew] = useState<ExistingOrNew>("existing");

  const activeTab: ExistingOrNew =
    projectTags.length > 0 ? existingOrNew : "new";

  const collectionOptions = useMemo(
    () => projectTags.map(({ hash, name }) => ({ label: name, value: hash })),
    [projectTags]
  );

  const { mutateAsync: createTag, isLoading: isMutatingCreateTag } =
    useProjectMutationCreateTag(projectHash);

  const isMutating =
    isMutatingItemsAdd || isMutatingCreateTag || isMutatingFiltersAdd;

  const handleAdd = useMemo(
    () => async () => {
      try {
        if (activeTab === "existing") {
          await addTags(checkedList.map((tag) => tag.hash));
        } else {
          const fields = await newCollectionForm.validateFields();
          const newTagsDict = await createTag([
            {
              name: fields.collection_title,
              description: fields.collection_description ?? "",
            },
          ]);
          const newTagHash =
            newTagsDict[newCollectionForm.getFieldValue("collection_title")];
          if (newTagHash !== undefined) {
            addTags([newTagHash]);
          }
        }
      } catch (e) {
        notification.error({
          message: "Error",
          description: "Error adding items to collection",
          placement: "bottomRight",
          duration: 5,
        });
      }
    },
    [activeTab, checkedList, newCollectionForm, addTags, createTag]
  );

  return (
    <div>
      <Modal
        open={open}
        title="Add items to Collection"
        okText="Submit"
        onCancel={closeModal}
        okButtonProps={{
          loading: isMutating,
          style: { backgroundColor: "#5555ff" },
        }}
        cancelButtonProps={{ disabled: isMutating }}
        onOk={() => {
          handleAdd();
        }}
      >
        <div className="flex flex-col gap-4">
          <Segmented
            block
            options={[
              { label: "Existing Collection", value: "existing" },
              { label: "+ New Collection", value: "new" },
            ]}
            value={activeTab}
            onChange={(val) => setExistingOrNew(val as ExistingOrNew)}
          />

          {activeTab === "existing" && (
            <CheckboxGroup
              onChange={onChange}
              className="flex flex-col gap-2"
              value={checkedList.map((it) => it.hash)}
              options={collectionOptions}
            />
          )}
          {activeTab === "new" && (
            <div>
              <Form
                form={newCollectionForm}
                layout="vertical"
                name="create_project_subset_form"
                initialValues={{ modifier: "public" }}
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
          )}
        </div>
      </Modal>
    </div>
  );
}
