import * as React from "react";
import { Select, SelectProps, Spin } from "antd";
import { useMemo, useRef, useState } from "react";
import { HiOutlineTag } from "react-icons/hi";
import { MdOutlineImage } from "react-icons/md";
import { TbPolygon } from "react-icons/tb";

import { classy } from "../../helpers/classy";
import { defaultTags, GroupedTags, useApi } from "./api";
import { takeDataId } from "./id";
import { loadingIndicator } from "../Spin";

const TAG_GROUPS = [
  { value: "data", label: "Data", Icon: MdOutlineImage },
  { value: "label", label: "Label", Icon: TbPolygon },
] as const;

const taggingDisabledReasons = {
  prediction: "Tagging is not available for predictions",
  "missing-target": "Select items to tag first",
} as const;

export const useAllTags = (itemSet?: Set<string>) => {
  const { isLoading, data: taggedItems } = useApi().fetchTaggedItems();
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

  if (isLoading) {
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
      data.forEach(result.selectedTags.label.add, result.selectedTags.label);
    }
    if (dataItemSet.has(id)) {
      data.forEach(result.selectedTags.data.add, result.selectedTags.data);
    }

    return result;
  }, defaultAllTags);

  return {
    ...allTags,
    selectedTags: {
      data: [...allTags.selectedTags.data],
      label: [...allTags.selectedTags.label],
    },
  };
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
  const { selectedTags, isLoading, taggedItems } = useAllTags(new Set(items));
  const { mutate, isLoading: isMutating } = useApi().itemTagsMutation;

  return (
    <TaggingForm
      loading={isLoading || isMutating}
      controlled
      allowClear={false}
      allowTaggingAnnotations={allowTaggingAnnotations}
      onSelect={(scope, selected) =>
        mutate(
          items.map((id) => {
            const { label } = taggedItems.get(id) || defaultTags;
            const { data } = taggedItems.get(takeDataId(id)) || defaultTags;

            const groupedTags = { data, label };

            return {
              id,
              groupedTags: {
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
            const itemPreviousTags = taggedItems.get(id);
            if (
              itemPreviousTags &&
              itemPreviousTags[scope].includes(deselected)
            ) {
              payload.push({
                id,
                groupedTags: {
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
  ...rest
}: {
  selectedTags: GroupedTags;
  controlled?: boolean;
  loading?: boolean;
  onChange?: (tags: GroupedTags) => void;
  onDeselect?: (scope: keyof GroupedTags, tag: string) => void;
  onSelect?: (scope: keyof GroupedTags, tag: string) => void;
  allowClear?: SelectProps["allowClear"];
  allowTaggingAnnotations?: boolean;
} & Omit<JSX.IntrinsicElements["div"], "onChange" | "onSelect">) {
  const { allDataTags, allLabelTags } = useAllTags();
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
      {...rest}
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
            disabled={value == "label" && !allowTaggingAnnotatoins}
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

export function TagList({
  tags,
  className,
  ...rest
}: { tags: GroupedTags } & JSX.IntrinsicElements["div"]) {
  return (
    <div {...rest} className={`flex flex-col gap-1 ${className}`}>
      {TAG_GROUPS.map((group) => (
        <div key={group.value}>
          <div className="inline-flex items-center gap-1">
            <group.Icon className="text-base" />
            <span>{group.label} tags:</span>
          </div>
          <div className="flex-wrap">
            {tags[group.value].length ? (
              tags[group.value].map((tag, index) => (
                <span key={index} className="badge">
                  {tag}
                </span>
              ))
            ) : (
              <span>None</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
