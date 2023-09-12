import * as React from "react";
import { Select, SelectProps, Spin, Tag } from "antd";
import { useMemo, useRef, useState } from "react";
import { HiOutlineTag } from "react-icons/hi";
import { MdOutlineImage } from "react-icons/md";
import { TbPolygon } from "react-icons/tb";

import { classy } from "../../helpers/classy";
import { takeDataId } from "./id";
import { loadingIndicator } from "../Spin";
import { useProjectTaggedItems } from "../../hooks/queries/useProjectTaggedItems";
import { useProjectHash } from "../../hooks/useProjectHash";
import { useProjectMutationTagItems } from "../../hooks/mutation/useProjectMutationTagItems";
import { ProjectItemTags, ProjectTag } from "../../openapi/api";

const TAG_GROUPS = [
  { value: "data", label: "Data", Icon: MdOutlineImage },
  { value: "label", label: "Label", Icon: TbPolygon },
] as const;

const taggingDisabledReasons = {
  prediction: "Tagging is not available for predictions",
  "missing-target": "Select items to tag first",
} as const;

const defaultTags = { data: [], label: [] };

type GroupedTags = {
  readonly data: readonly string[];
  readonly label: readonly string[];
};

export const useAllTags = (projectHash: string, itemSet?: Set<string>) => {
  const { isLoading, data: taggedItems } = useProjectTaggedItems(projectHash);
  const defaultAllTags = {
    allDataTags: new Set<string>(),
    allLabelTags: new Set<string>(),
    selectedTags: { data: new Set<string>(), label: new Set<string>() },
    isLoading,
    taggedItems,
  };

  const dataItemSet = useMemo(
    () => new Set([...(itemSet || [])].map((id) => takeDataId(id))),
    [itemSet]
  );

  if (isLoading || !taggedItems) {
    return {
      ...defaultAllTags,
      selectedTags: {
        data: [...defaultAllTags.selectedTags.data],
        label: [...defaultAllTags.selectedTags.label],
      },
    };
  }

  const allTags = [...taggedItems].reduce((result, [id, { data, label }]) => {
    data.forEach(result.allDataTags.add, result.allDataTags);
    label.forEach(result.allLabelTags.add, result.allLabelTags);

    if (itemSet?.has(id)) {
      label.forEach(result.selectedTags.label.add, result.selectedTags.label);
    }
    if (dataItemSet.has(id)) {
      data.forEach(result.selectedTags.data.add, result.selectedTags.data);
    }

    return result;
  }, defaultAllTags);

  const selectedTags = {
    data: [...allTags.selectedTags.data],
    label: [...allTags.selectedTags.label],
  };

  return { ...allTags, selectedTags };
};

export function TaggingDropdown({
  disabledReason,
  children,
  className,
  ...rest
}: {
  disabledReason?: keyof typeof taggingDisabledReasons;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      {...rest}
      className={classy(
        "dropdown dropdown-bottom tooltip tooltip-right min-w-fit",
        className
      )}
      data-tip={disabledReason && taggingDisabledReasons[disabledReason]}
    >
      <label
        tabIndex={0}
        className={classy("btn btn-ghost gap-2", {
          "btn-disabled": disabledReason,
        })}
      >
        <HiOutlineTag />
        Tag
      </label>
      {children}
    </div>
  );
}

export function BulkTaggingForm({
  items,
  allowTaggingAnnotations,
}: {
  items: string[];
  allowTaggingAnnotations: boolean;
}) {
  const projectHash = useProjectHash();
  const { selectedTags, isLoading, taggedItems } = useAllTags(
    projectHash,
    new Set(items)
  );
  const { mutate, isLoading: isMutating } =
    useProjectMutationTagItems(projectHash);

  return (
    <TaggingForm
      loading={isLoading || isMutating}
      controlled
      allowClear={false}
      allowTaggingAnnotations={allowTaggingAnnotations}
      onSelect={(scope, selected) =>
        mutate(
          items.map((id) => {
            const { label } = taggedItems?.get(id) || defaultTags;
            const { data } = taggedItems?.get(takeDataId(id)) || defaultTags;

            const groupedTags = { data, label };

            return {
              id,
              grouped_tags: {
                ...groupedTags,
                [scope]: [...groupedTags[scope], selected],
              },
            };
          })
        )
      }
      onDeselect={(scope, deselected) =>
        mutate(
          items.reduce((payload, id) => {
            const itemPreviousTags = taggedItems?.get(id);
            if (
              itemPreviousTags &&
              itemPreviousTags[scope].includes(deselected)
            ) {
              payload.push({
                id,
                grouped_tags: {
                  ...itemPreviousTags,
                  [scope]: itemPreviousTags[scope].filter(
                    (tag) => tag !== deselected
                  ),
                },
              });
            }
            return payload;
          }, [] as Parameters<typeof mutate>[0])
        )
      }
      selectedTags={selectedTags}
    />
  );
}

export function TaggingForm({
  selectedTags,
  className,
  controlled = false,
  loading = false,
  onChange,
  onSelect,
  onDeselect,
  allowClear = true,
  allowTaggingAnnotations: allowTaggingAnnotatoins = false,
}: {
  selectedTags: GroupedTags;
  controlled?: boolean;
  loading?: boolean;
  onChange?: (tags: GroupedTags) => void;
  onDeselect?: (scope: keyof GroupedTags, tag: string) => void;
  onSelect?: (scope: keyof GroupedTags, tag: string) => void;
  allowClear?: SelectProps["allowClear"];
  allowTaggingAnnotations?: boolean;
  className: string;
}) {
  const projectHash = useProjectHash();
  const { allDataTags, allLabelTags } = useAllTags(projectHash);
  const allTags = { data: [...allDataTags], label: [...allLabelTags] };

  const [selectedTab, setTab] = useState<(typeof TAG_GROUPS)[number]>(
    TAG_GROUPS[0]
  );

  // NOTE: hack to prevent loosing focus when loading
  const ref = useRef<HTMLDivElement>(null);
  if (loading) {
    ref.current && ref.current.focus();
  }

  return (
    <div
      tabIndex={0}
      className={classy(
        "card dropdown-content card-compact w-64 bg-base-100 p-2 text-primary-content shadow",
        className
      )}
    >
      <div className="tabs flex justify-center bg-base-100">
        {TAG_GROUPS.map((group) => (
          <a
            key={group.value}
            className={classy("tab tab-bordered gap-2", {
              "tab-active": selectedTab.label === group.label,
            })}
            onClick={() => setTab(group)}
          >
            <group.Icon className="text-base" />
            {group.label}
          </a>
        ))}
      </div>
      <div ref={ref} tabIndex={-1} className="card-body">
        {loading && <Spin indicator={loadingIndicator} />}
        {TAG_GROUPS.map(({ value }) => (
          <Select
            key={value}
            className={classy({
              "!hidden": value !== selectedTab.value,
            })}
            disabled={value === "label" && !allowTaggingAnnotatoins}
            mode="tags"
            placeholder="Tags"
            allowClear={allowClear}
            onChange={(tags) => onChange?.({ ...selectedTags, [value]: tags })}
            onDeselect={(tag: string) => onDeselect?.(selectedTab.value, tag)}
            onSelect={(tag: string) => onSelect?.(selectedTab.value, tag)}
            options={allTags[value].map((tag) => ({ value: tag }))}
            {...(controlled
              ? { value: selectedTags[value] }
              : { defaultValue: selectedTags[value] })}
          />
        ))}
      </div>
    </div>
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
  const annotationTags = annotationHash && label[annotationHash];
  const labelTags =
    annotationTags ||
    (Object.values(label).filter(Boolean).flat() as ProjectTag[]);

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {!!data.length && (
        <div className="flex items-center">
          <MdOutlineImage className="text-base" />
          <TagList tags={data} limit={limit} />
        </div>
      )}
      {!!labelTags.length && (
        <div className="flex items-center">
          <TbPolygon className="text-base" />
          <TagList tags={labelTags} limit={limit} />
        </div>
      )}
    </div>
  );
}

export function TagList({
  tags,
  limit,
}: {
  tags: ProjectTag[];
  limit?: number;
}) {
  const firstTags = tags.slice(0, limit);
  const remainder = tags.length - firstTags.length;

  return (
    <div className="flex-wrap">
      {firstTags.map((tag) => (
        <Tag key={tag.tag_hash} bordered={false} className="rounded-xl">
          {tag.name}
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
