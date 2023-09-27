import { Select, Tabs, Tag } from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";
import { MdOutlineImage } from "react-icons/md";
import { TbPolygon } from "react-icons/tb";

import Icon from "@ant-design/icons";
import { useProjectMutationItemsAddTag } from "../../hooks/mutation/useProjectMutationItemsAddTag";
import {
  AllTagsResult,
  AnalysisDomain,
  ProjectItemTags,
  ProjectTag,
  SearchFilters,
} from "../../openapi/api";
import { useProjectMutationItemsRemoveTag } from "../../hooks/mutation/useProjectMutationItemsRemoveTag";
import { useProjectMutationFiltersAddTag } from "../../hooks/mutation/useProjectMutationFiltersAddTag";
import { useProjectMutationFiltersRemoveTag } from "../../hooks/mutation/useProjectMutationFiltersRemoveTag";
import { useProjectListTags } from "../../hooks/queries/useProjectListTags";
import { useProjectItemsListTags } from "../../hooks/queries/useProjectItemsListTags";
import { useProjectMutationCreateTag } from "../../hooks/mutation/useProjectMutationCreateTag";
import { toDataItemID } from "../util/ItemIdUtil";
import { useProjectFilterListTags } from "../../hooks/queries/useProjectFilterListTags";

export function BulkTaggingForm(props: {
  projectHash: string;
  selectedItems: ReadonlySet<string> | "ALL";
  filtersDomain: AnalysisDomain;
  filters: SearchFilters;
  allowTaggingAnnotations: boolean;
}) {
  const {
    projectHash,
    selectedItems,
    filtersDomain,
    filters,
    allowTaggingAnnotations,
  } = props;
  const [domain, setDomain] = useState<"data" | "annotation">("data");
  useEffect(() => {
    if (domain === "annotation" && !allowTaggingAnnotations) {
      setDomain("data");
    }
  }, [allowTaggingAnnotations, domain]);

  // Lookup state
  const { data: projectTags = [] } = useProjectListTags(projectHash);
  const selectOptions = useMemo(
    () => projectTags.map(({ hash, name }) => ({ label: name, value: hash })),
    [projectTags]
  );
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

  const selectedTagsGroup: AllTagsResult = useMemo(() => {
    if (selectedItems === "ALL") {
      return filterTags ?? { data: [], annotation: [] };
    } else {
      return itemsTags ?? { data: [], annotation: [] };
    }
  }, [selectedItems, itemsTags, filterTags]);
  const selectedTags = selectedTagsGroup[domain];
  const selectValue = useMemo(
    () => selectedTags.map(({ hash }) => hash),
    [selectedTags]
  );

  // All mutations
  const { mutateAsync: createTag, isLoading: isMutatingCreateTag } =
    useProjectMutationCreateTag(projectHash);
  const { mutate: itemsAddTags, isLoading: isMutatingItemsAdd } =
    useProjectMutationItemsAddTag(projectHash);
  const { mutate: itemsRemoveTags, isLoading: isMutatingItemsRemove } =
    useProjectMutationItemsRemoveTag(projectHash);
  const { mutate: filtersAddTags, isLoading: isMutatingFiltersAdd } =
    useProjectMutationFiltersAddTag(projectHash);
  const { mutate: filtersRemoveTags, isLoading: isMutatingFiltersRemove } =
    useProjectMutationFiltersRemoveTag(projectHash);
  const isMutating =
    isMutatingCreateTag ||
    isMutatingItemsAdd ||
    isMutatingItemsRemove ||
    isMutatingFiltersAdd ||
    isMutatingFiltersRemove;

  const getDomainItems = useCallback(
    (items: ReadonlySet<string>): string[] => {
      if (domain === "data") {
        const dataItems = new Set([...items].map(toDataItemID));

        return [...dataItems];
      } else {
        return [...items];
      }
    },
    [domain]
  );

  const onSelect = async (
    tagName: string,
    option: { label?: string; value?: string }
  ) => {
    let tagHash = option.value;
    if (tagHash === undefined) {
      const tagDict = await createTag([tagName]);
      const newTagHash = tagDict[tagName];
      if (newTagHash === undefined) {
        return;
      }
      tagHash = newTagHash;
    }
    if (selectedItems === "ALL") {
      filtersAddTags({ domain, filters, tags: [tagHash] });
    } else {
      itemsAddTags({ items: getDomainItems(selectedItems), tags: [tagHash] });
    }
  };
  const onDeselect = (tagHash: string) => {
    if (selectedItems === "ALL") {
      filtersRemoveTags({ domain, filters, tags: [tagHash] });
    } else {
      itemsRemoveTags({
        items: getDomainItems(selectedItems),
        tags: [tagHash],
      });
    }
  };

  return (
    <>
      <Tabs
        className="w-72"
        centered
        items={[
          {
            label: (
              <span>
                <Icon component={MdOutlineImage} />
                Data
              </span>
            ),
            key: "data",
          },
          {
            label: (
              <span>
                <Icon component={TbPolygon} />
                Annotation
              </span>
            ),
            key: "annotation",
            disabled: !allowTaggingAnnotations,
          },
        ]}
        onChange={setDomain as (key: string) => void}
        activeKey={domain}
      />
      <Select
        className="w-full"
        mode="tags"
        placeholder="Tags"
        allowClear
        disabled={domain === "annotation" && !allowTaggingAnnotations}
        value={selectValue}
        options={selectOptions}
        loading={isMutating}
        onSelect={isMutating ? undefined : onSelect}
        onDeselect={isMutating ? undefined : onDeselect}
      />
    </>
  );
}

export function ItemTags({
  tags: { data, label },
  annotationHash,
  limit,
  className,
}: {
  tags: ProjectItemTags;
  annotationHash?: string;
  limit?: number;
  className?: string;
}) {
  const selectAnnotationTags = annotationHash && label[annotationHash];
  const allAnnotationTags =
    selectAnnotationTags ||
    (Object.values(label).filter(Boolean).flat() as ProjectTag[]);

  const dataTags = data.map((d) => d.name).sort();
  const annotationTags = [...new Set(allAnnotationTags.map((t) => t.name))];

  return (
    <div className={`flex flex-col gap-1 ${className ?? ""}`}>
      {!!data.length && (
        <div className="flex items-center">
          <MdOutlineImage className="text-base" />
          <TagList tags={dataTags} limit={limit} />
        </div>
      )}
      {!!annotationTags.length && (
        <div className="flex items-center">
          <TbPolygon className="text-base" />
          <TagList tags={annotationTags} limit={limit} />
        </div>
      )}
    </div>
  );
}

export function TagList(props: { tags: string[]; limit?: number }) {
  const { tags, limit } = props;
  const firstTags = tags.slice(0, limit);
  const remainder = tags.length - firstTags.length;

  return (
    <div className="flex-wrap">
      {firstTags.map((tag) => (
        <Tag key={tag} bordered={false} className="rounded-xl">
          {tag}
        </Tag>
      ))}
      {remainder > 0 && (
        <Tag bordered={false} color="#434343" className="rounded-xl">
          + {remainder} more tags
        </Tag>
      )}
    </div>
  );
}
