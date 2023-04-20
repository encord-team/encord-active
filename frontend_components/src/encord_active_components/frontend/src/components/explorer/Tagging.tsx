import { Select, Spin, SelectProps } from "antd";
import { useRef, useState } from "react";
import { HiOutlineTag } from "react-icons/hi";
import { MdOutlineImage } from "react-icons/md";
import { TbPolygon } from "react-icons/tb";

import { classy } from "../../helpers/classy";
import { defaultTags, GroupedTags, useProjectQueries } from "./api";

const TAG_GROUPS = [
  { value: "data", label: "Data", Icon: MdOutlineImage },
  { value: "label", label: "Label", Icon: TbPolygon },
] as const;

export const TaggingDropdown = ({
  disabled = false,
  children,
  className,
  ...rest
}: { disabled?: boolean } & JSX.IntrinsicElements["div"]) => (
  <div
    {...rest}
    className={classy("dropdown dropdown-bottom  min-w-fit", className)}
  >
    <label
      tabIndex={0}
      className={classy("btn btn-ghost gap-2", {
        "btn-disabled": disabled,
      })}
    >
      <HiOutlineTag />
      Tag
    </label>
    {children}
  </div>
);

export const BulkTaggingForm = ({ items }: { items: string[] }) => {
  const { isLoading, data: taggedItems } =
    useProjectQueries().fetchTaggedItems();
  const { mutate, isLoading: isMutating } =
    useProjectQueries().itemTagsMutation;

  const itemSet = new Set(items);
  const { data: allDataTags, label: allLabelTags } = [...taggedItems]
    .filter(([id, _]) => itemSet.has(id))
    .map(([_, tags]) => tags)
    .reduce(
      (allTags, { data, label }) => (
        data.forEach(allTags.data.add, allTags.data),
        label.forEach(allTags.label.add, allTags.label),
        allTags
      ),
      { data: new Set<string>(), label: new Set<string>() }
    );

  return (
    <TaggingForm
      loading={isLoading || isMutating}
      controlled={true}
      allowClear={false}
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
  ...rest
}: {
  seletedTags: GroupedTags;
  controlled?: boolean;
  loading?: boolean;
  onChange?: (tags: GroupedTags) => void;
  onDeselect?: (scope: keyof GroupedTags, tag: string) => void;
  onSelect?: (scope: keyof GroupedTags, tag: string) => void;
  allowClear?: SelectProps["allowClear"];
} & Omit<JSX.IntrinsicElements["div"], "onChange" | "onSelect">) => {
  const { data: allTags } = useProjectQueries().fetchProjectTags();

  const [selectedTab, setTab] = useState<typeof TAG_GROUPS[number]>(
    TAG_GROUPS[0]
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
        className
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
        {loading && <Spin />}
        {TAG_GROUPS.map(({ value }) => (
          <Select
            key={value}
            className={classy({
              hidden: value !== selectedTab.value,
            })}
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

export const TagList = ({ tags }: { tags: GroupedTags }) => (
  <div className="flex flex-col gap-3">
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
