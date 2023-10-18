import { Checkbox, Modal, notification } from "antd";
import { useCallback, useMemo, useState } from "react";
import { CheckboxValueType } from "antd/es/checkbox/Group";
import { useProjectItemsListTags } from "../../../hooks/queries/useProjectItemsListTags";
import { useProjectMutationItemsRemoveTag } from "../../../hooks/mutation/useProjectMutationItemsRemoveTag";
import { toDataItemID } from "../../util/ItemIdUtil";
import {
  AllTagsResult,
  AnalysisDomain,
  ProjectTagEntry,
  SearchFilters,
} from "../../../openapi/api";
import { useProjectFilterListTags } from "../../../hooks/queries/useProjectFilterListTags";
import { useProjectMutationFiltersRemoveTag } from "../../../hooks/mutation/useProjectMutationFiltersRemoveTag";

export function RemoveFromCollectionModal(props: {
  open: boolean;
  close: () => void;
  projectHash: string;
  selectedItems: ReadonlySet<string> | "ALL";
  filtersDomain: AnalysisDomain;
  filters: SearchFilters;
}) {
  const { open, close, projectHash, selectedItems, filtersDomain, filters } =
    props;

  const selectedItemsList: string[] = useMemo(
    () => (selectedItems === "ALL" ? [] : [...selectedItems]),
    [selectedItems]
  );
  const { data: itemsTags } = useProjectItemsListTags(
    projectHash,
    selectedItemsList,
    { enabled: selectedItemsList.length > 0 }
  );
  const { data: filterTags } = useProjectFilterListTags(
    projectHash,
    filtersDomain,
    filters,
    {
      enabled: selectedItems === "ALL",
    }
  );

  const CheckboxGroup = Checkbox.Group;
  const selectedTagsGroup: AllTagsResult = useMemo(() => {
    if (selectedItems === "ALL") {
      return filterTags ?? { data: [], annotation: [] };
    } else {
      return itemsTags ?? { data: [], annotation: [] };
    }
  }, [selectedItems, itemsTags, filterTags]);

  const plainOptions = useMemo(
    () => selectedTagsGroup?.data.map((item) => item) ?? [],
    [selectedTagsGroup]
  );

  const [checkedList, setCheckedList] = useState<ProjectTagEntry[]>([]);
  const checkAll = plainOptions.length === checkedList.length;
  const indeterminate =
    checkedList.length > 0 && checkedList.length < plainOptions.length;
  const onChange = (list: CheckboxValueType[]) => {
    const newList = plainOptions.filter((item) => list.includes(item.hash));
    setCheckedList(newList);
  };
  const onCheckAllChange = (e: any) => {
    setCheckedList(e.target.checked ? plainOptions : []);
  };

  const { mutateAsync: itemsRemoveTags, isLoading: isMutatingItemsRemove } =
    useProjectMutationItemsRemoveTag(projectHash);
  const { mutateAsync: filtersRemoveTags, isLoading: isMutatingFiltersRemove } =
    useProjectMutationFiltersRemoveTag(projectHash);

  const isMutating = isMutatingItemsRemove || isMutatingFiltersRemove;

  const getDomainItems = useCallback((items: ReadonlySet<string>): string[] => {
    const dataItems = new Set([...items].map(toDataItemID));

    return [...dataItems];
  }, []);

  const removeTags = useCallback(async () => {
    if (selectedItems === "ALL") {
      filtersRemoveTags({
        domain: filtersDomain,
        filters,
        tags: checkedList.map((tag) => tag.hash),
      });
    } else {
      await itemsRemoveTags({
        items: getDomainItems(selectedItems),
        tags: checkedList.map((tag) => tag.hash),
      });
    }
  }, [
    selectedItems,
    getDomainItems,
    checkedList,
    filtersRemoveTags,
    itemsRemoveTags,
    filtersDomain,
    filters,
  ]);

  // Reset logic
  const closeModal = useCallback(() => {
    checkedList.length = 0;
    close();
  }, [checkedList, close]);

  const handleRemove = useCallback(async () => {
    try {
      await removeTags();
      closeModal();
      notification.success({
        message: "Items Removed",
        description: "Items removed from collection successfully.",
        placement: "bottomRight",
        duration: 5,
      });
    } catch (e) {
      notification.error({
        message: "Error",
        description: "Items could not be removed from collection.",
        placement: "bottomRight",
        duration: 5,
      });
    }
  }, [removeTags, closeModal]);

  return (
    <Modal
      open={open}
      title="Remove N items from"
      okText="Submit"
      onCancel={closeModal}
      okButtonProps={{
        loading: isMutating,
        style: { backgroundColor: "#5555ff" },
      }}
      cancelButtonProps={{ disabled: isMutating }}
      onOk={() => {
        handleRemove();
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
          options={plainOptions.map((item) => ({
            label: item.name,
            value: item.hash,
          }))}
        />
      </>
    </Modal>
  );
}
