import { Checkbox, Form, Input, Modal, Segmented } from "antd";
import { useProjectItemsListTags } from "../../../hooks/queries/useProjectItemsListTags";
import { useCallback, useMemo, useState } from "react";
import { useProjectMutationItemsRemoveTag } from "../../../hooks/mutation/useProjectMutationItemsRemoveTag";
import { toDataItemID } from "../../util/ItemIdUtil";
import { AnalysisDomain, ProjectTagEntry } from "../../../openapi/api";
import { CheckboxValueType } from "antd/es/checkbox/Group";
import { useProjectListTags } from "../../../hooks/queries/useProjectListTags";
import { useProjectMutationItemsAddTag } from "../../../hooks/mutation/useProjectMutationItemsAddTag";
import TextArea from "antd/es/input/TextArea";
import { useProjectMutationCreateTag } from "../../../hooks/mutation/useProjectMutationCreateTag";

export function AddToCollectionModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  selectedItems: ReadonlySet<string> | "ALL";
  analysisDomain: AnalysisDomain;
}) {
  const { open, close, projectHash, selectedItems, analysisDomain } = props;
  const { data: projectTags = [] } = useProjectListTags(projectHash);
  const selectedItemsList: string[] = useMemo(
    () => (selectedItems === "ALL" ? [] : [...selectedItems]),
    [selectedItems]
  );
  const CheckboxGroup = Checkbox.Group;

  const [checkedList, setCheckedList] = useState<ProjectTagEntry[]>([]);
  const onChange = (list: CheckboxValueType[]) => {
    console.log(list);

    const newList = projectTags.filter((item) => {
      return list.includes(item.hash);
    });
    setCheckedList(newList);
  };

  const { mutateAsync: itemsAddTags, isLoading: isMutatingItemsAdd } =
    useProjectMutationItemsAddTag(projectHash);

  const getDomainItems = useCallback(
    (items: ReadonlySet<string>): string[] => {
      if (analysisDomain === "data") {
        const dataItems = new Set([...items].map(toDataItemID));
        return [...dataItems];
      } else {
        return [...items];
      }
    },
    [analysisDomain]
  );
  const addTags = async (tagList: string[]) => {
    if (selectedItems === "ALL") {
      //   filtersAddTags({ domain, filters, tags: [tagHash] });
    } else {
      await itemsAddTags({
        items: getDomainItems(selectedItems),
        tags: tagList,
      });
    }
  };

  const [existingOrNew, setExistingOrNew] = useState<"existing" | "new">(
    "existing"
  );

  const collectionOptions = useMemo(
    () => projectTags.map(({ hash, name }) => ({ label: name, value: hash })),
    [projectTags]
  );

  const [newCollectionForm] = Form.useForm<{
    collection_title: string;
    collection_description?: string | undefined;
  }>();

  const { mutateAsync: createTag, isLoading: isMutatingCreateTag } =
    useProjectMutationCreateTag(projectHash);

  // Reset logic
  const closeModal = () => {
    newCollectionForm.resetFields();
    checkedList.length = 0;
    setExistingOrNew("existing");
    close();
  };

  let isMutating = isMutatingItemsAdd || isMutatingCreateTag;
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
          if (existingOrNew == "existing") {
            addTags(checkedList.map((tag) => tag.hash))
              .then(() => closeModal())
              .catch(() =>
                console.log("Something went wrong, please try again later.")
              );
          } else {
            // write logic for adding new collection and then adding items to it.
            newCollectionForm
              .validateFields()
              .then((fields) =>
                createTag([
                  {
                    name: fields.collection_title,
                    description: fields.collection_description ?? "Hello",
                  },
                ])
              )
              .then((newTagsDict) => {
                const newTagHash =
                  newTagsDict[
                    newCollectionForm.getFieldValue("collection_title")
                  ];
                if (newTagHash !== undefined)
                  addTags([newTagHash])
                    .then(() => closeModal())
                    .catch(() =>
                      console.log(
                        "Something went wrong, please try again later."
                      )
                    );
              })
              .catch(() =>
                console.log("Something went wrong, please try again later.")
              );
          }
        }}
      >
        <div className="flex flex-col gap-4">
          <Segmented
            block
            options={[
              { label: "Existing Collection", value: "existing" },
              { label: "+ New Collection", value: "new" },
            ]}
            value={existingOrNew}
            onChange={(val) => setExistingOrNew(val as "existing" | "new")}
          />

          {existingOrNew == "existing" && (
            <CheckboxGroup
              onChange={onChange}
              className="flex flex-col gap-2"
              value={checkedList.map((it) => it.hash)}
              options={collectionOptions}
            />
          )}
          {existingOrNew == "new" && (
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
