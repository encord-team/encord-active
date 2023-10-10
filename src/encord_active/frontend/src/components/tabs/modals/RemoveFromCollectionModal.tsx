import {
  Checkbox,
  CheckboxOptionType,
  Divider,
  Form,
  Input,
  Modal,
} from "antd";
import { useProjectMutationUploadToEncord } from "../../../hooks/mutation/useProjectMutationUploadToEncord";
import { useProjectItemsListTags } from "../../../hooks/queries/useProjectItemsListTags";
import { MouseEvent, useCallback, useMemo, useState } from "react";
import { useProjectMutationItemsRemoveTag } from "../../../hooks/mutation/useProjectMutationItemsRemoveTag";
import { toDataItemID } from "../../util/ItemIdUtil";
import { AnalysisDomain, ProjectTagEntry } from "../../../openapi/api";
import { CheckboxValueType } from "antd/es/checkbox/Group";

export function RemoveFromCollectionModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  selectedItems: ReadonlySet<string> | "ALL";
  analysisDomain: AnalysisDomain;
}) {
  const { open, close, projectHash, selectedItems, analysisDomain } = props;

  const selectedItemsList: string[] = useMemo(
    () => (selectedItems === "ALL" ? [] : [...selectedItems]),
    [selectedItems]
  );
  const { data: itemsTags } = useProjectItemsListTags(
    projectHash,
    selectedItemsList,
    { enabled: selectedItemsList.length > 0 }
  );
  const CheckboxGroup = Checkbox.Group;
  const plainOptions = itemsTags?.data.map((item) => item) ?? [];

  const [checkedList, setCheckedList] = useState<ProjectTagEntry[]>([]);
  const checkAll = plainOptions.length === checkedList.length;
  const indeterminate =
    checkedList.length > 0 && checkedList.length < plainOptions.length;
  const onChange = (list: CheckboxValueType[]) => {
    console.log(list);

    const newList = plainOptions.filter((item) => {
      return list.includes(item.hash);
    });
    setCheckedList(newList);
  };
  const onCheckAllChange = (e: any) => {
    setCheckedList(e.target.checked ? plainOptions : []);
  };

  const { mutateAsync: itemsRemoveTags, isLoading: isMutatingItemsRemove } =
    useProjectMutationItemsRemoveTag(projectHash);
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
  const removeTags = async () => {
    if (selectedItems === "ALL") {
      // filtersRemoveTags({ domain, filters, tags: [tagHash] });
    } else {
      await itemsRemoveTags({
        items: getDomainItems(selectedItems),
        tags: checkedList.map((tag) => tag.hash),
      });
    }
  };

  // Reset logic
  const closeModal = () => {
    checkedList.length = 0;
    close();
  };
  return (
    <Modal
      open={open}
      title="Remove N items from"
      okText="Submit"
      onCancel={closeModal}
      okButtonProps={{
        loading: isMutatingItemsRemove,
        style: { backgroundColor: "#5555ff" },
      }}
      cancelButtonProps={{ disabled: isMutatingItemsRemove }}
      onOk={() => {
        removeTags()
          .then(() => closeModal())
          .catch(() =>
            console.log("Something went wrong, please try again later.")
          );
      }}
    >
      <>
        <Checkbox
          indeterminate={indeterminate}
          onChange={onCheckAllChange}
          checked={checkAll}
          className="my-2 font-medium text-primary"
          disabled={plainOptions.length === 0}
        >
          Select All
        </Checkbox>

        <CheckboxGroup
          onChange={onChange}
          className="flex flex-col gap-2"
          value={checkedList.map((it) => it.hash)}
          options={plainOptions.map((item) => {
            return {
              label: item.name,
              value: item.hash,
            };
          })}
        />
      </>
    </Modal>
  );
}
