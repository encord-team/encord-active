import { Select, SelectProps } from "antd";
import { useRef, useState } from "react";
import { HiOutlineTag } from "react-icons/hi";
import { MdOutlineImage } from "react-icons/md";
import { TbPolygon } from "react-icons/tb";

import { classy } from "../../helpers/classy";
import { defaultTags, GroupedTags, useApi } from "./api";
import { Spinner } from "./Spinner";

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
    isLoading,
    taggedItems,
  };

  if (isLoading) return defaultAllTags;

  return [...taggedItems]
    .filter(([id, _]) => (itemSet ? itemSet.has(id) : true))
    .map(([_, tags]) => tags)
    .reduce(
      (allTags, { data, label }) => (
        data.forEach(allTags.allDataTags.add, allTags.allDataTags),
        label.forEach(allTags.allLabelTags.add, allTags.allLabelTags),
        allTags
      ),
      defaultAllTags,
    );
};

export const TaggingDropdown = ({
  disabledReason,
  children,
  className,
  ...rest
}: {
  disabledReason?: keyof typeof taggingDisabledReasons;
} & JSX.IntrinsicElements["div"]) => {
  return (
    <div
      {...rest}
      className={classy(
        "dropdown dropdown-bottom min-w-fit tooltip tooltip-right",
        className,
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
};

export const BulkTaggingForm = ({
  items,
  allowTaggingAnnotations,
}: {
  items: string[];
  allowTaggingAnnotations: boolean;
}) => {
  const { allDataTags, allLabelTags, isLoading, taggedItems } = useAllTags(
    new Set(items),
  );
  const { mutate, isLoading: isMutating } = useApi().itemTagsMutation;

  return (
    <TaggingForm
      loading={isLoading || isMutating}
      controlled={true}
      allowClear={false}
      allowTaggingAnnotations={allowTaggingAnnotations}
      onSelect={(scope, selected) =>
        mutate(
          items.map((id) => {
            const itemTags = taggedItems.get(id);
            return {
              id,
              groupedTags: itemTags
                ? { ...itemTags, [scope]: [...itemTags[scope], selected] }
                : { ...defaultTags, [scope]: [selected] },
            };
          }),
        )
      }
      onDeselect={(scope, deselected) =>
        mutate(
          items.reduce(
            (payload, id) => {
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
                      (tag) => tag !== deselected,
                    ),
                  },
                });
              }
              return payload;
            },
            [] as Parameters<typeof mutate>[0],
          ),
        )
      }
      seletedTags={{ data: [...allDataTags], label: [...allLabelTags] }}
    />
  );
};

export const TaggingForm = ({
  seletedTags: selectedTags,
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
  seletedTags: GroupedTags;
  controlled?: boolean;
  loading?: boolean;
  onChange?: (tags: GroupedTags) => void;
  onDeselect?: (scope: keyof GroupedTags, tag: string) => void;
  onSelect?: (scope: keyof GroupedTags, tag: string) => void;
  allowClear?: SelectProps["allowClear"];
  allowTaggingAnnotations?: boolean;
} & Omit<JSX.IntrinsicElements["div"], "onChange" | "onSelect">) => {
  const { allDataTags, allLabelTags } = useAllTags();
  const allTags = { data: [...allDataTags], label: [...allLabelTags] };

  const [selectedTab, setTab] = useState<(typeof TAG_GROUPS)[number]>(
    TAG_GROUPS[0],
  );

  // NOTE: hack to prevent loosing focus when loading
  const ref = useRef<HTMLDivElement>(null);
  if (loading) ref.current && ref.current.focus();

  return (
    <div
      {...rest}
      tabIndex={0}
      className={classy(
        "dropdown-content card card-compact w-64 p-2 shadow bg-base-100 text-primary-content",
        className,
      )}
    >
      <div className="tabs flex justify-center bg-base-100">
        {TAG_GROUPS.map((group) => (
          <a
            key={group.value}
            className={classy("tab tab-bordered gap-2", {
              "tab-active": selectedTab.label == group.label,
            })}
            onClick={() => setTab(group)}
          >
            <group.Icon className="text-base" />
            {group.label}
          </a>
        ))}
      </div>
      <div ref={ref} tabIndex={-1} className="card-body">
        {loading && <Spinner />}
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
};

export const TagList = ({
  tags,
  className,
  ...rest
}: { tags: GroupedTags } & JSX.IntrinsicElements["div"]) => (
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
