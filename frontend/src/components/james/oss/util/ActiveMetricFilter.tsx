import * as React from "react";
import { useEffect, useMemo } from "react";
import { Button, Row, Select, Slider, Typography } from "antd";
import { MinusOutlined, PlusOutlined } from "@ant-design/icons";
import { ActiveProjectMetricSummary } from "../ActiveTypes";
import { useAllTags } from "../../../explorer/Tagging";
import { GroupedTags } from "../../../explorer/api";
import { isEmpty, omit } from "radash";

export const defaultFilters: ActiveFilterOrderState = {
  metricFilters: {},
  enumFilters: {},
  tagFilters: { data: [], label: [] },
  ordering: [],
};

export type Bounds = {
  min: number;
  max: number;
};

export type ActiveFilterOrderState = {
  readonly ordering: ReadonlyArray<string>;
} & ActiveFilterState;

export type ActiveFilterState = {
  readonly metricFilters: Readonly<Record<string, Bounds>>;
  readonly enumFilters: Readonly<Record<string, ReadonlyArray<string>>>;
  readonly tagFilters: GroupedTags;
};

export type ActiveFilterModes = keyof ActiveFilterState;

function getMetricBounds<R extends Bounds>(
  bounds: R,
  metric: ActiveProjectMetricSummary["metrics"][string]
): {
  min: number;
  max: number;
  step: number;
} {
  if (metric.type === "normal") {
    return {
      min: 0.0,
      max: 1.0,
      step: 0.01,
    };
  } else if (metric.type === "ufloat") {
    return {
      min: 0.0,
      max: bounds.max,
      step: bounds.max / 100,
    };
  } else if (metric.type === "uint") {
    return {
      min: 0,
      max: bounds.max,
      step: 1,
    };
  } else if (metric.type === "rank") {
    return {
      min: 0,
      max: bounds.max,
      step: 1,
    };
  } else {
    throw Error(`Unknown metric type: ${metric.type}`);
  }
}

function getEnumList(
  enumSummary: ActiveProjectMetricSummary["enums"][string],
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >
): ReadonlyArray<string> {
  if (enumSummary.type === "ontology") {
    return Object.keys(featureHashMap);
  } else if (enumSummary.type === "enum") {
    return Object.keys(enumSummary.values);
  } else {
    throw Error("Unknown enum state");
  }
}

function getEnumOptions(
  enumSummary: ActiveProjectMetricSummary["enums"][string],
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >
): Array<{ label: string; value: string }> {
  if (enumSummary.type === "ontology") {
    return Object.entries(featureHashMap).map(([featureHash, feature]) => ({
      value: featureHash,
      label: feature.name,
    }));
  } else if (enumSummary.type === "enum") {
    return Object.entries(enumSummary.values).map(([value, label]) => ({
      value,
      label,
    }));
  }
  throw Error("Unknown enum state");
}

function getEnumName(
  enumKey: string,
  enumSummary: ActiveProjectMetricSummary["enums"][string]
): string {
  if (enumKey === "feature_hash") {
    return "Label Class";
  } else if (enumSummary.type === "enum") {
    return enumSummary.title;
  }
  throw Error("Unknown enum state");
}

type NoTagFilters = Omit<ActiveFilterState, "tagFilters">;

function deleteKey(
  deleteKey: string,
  filterType: ActiveFilterModes
): (old: ActiveFilterOrderState) => ActiveFilterOrderState {
  return (old) => {
    const newOrdering = old.ordering.filter((elem) => elem !== deleteKey);

    if (filterType === "tagFilters")
      return {
        ...old,
        ordering: newOrdering,
        [filterType]: {
          ...old[filterType],
          [deleteKey as keyof ActiveFilterState["tagFilters"]]: [],
        },
      };

    const { [deleteKey]: deleted, ...newFilters } = old[
      filterType
    ] as NoTagFilters[keyof NoTagFilters];

    return {
      ...old,
      ordering: newOrdering,
      [filterType]: newFilters,
    };
  };
}

function updateValue<K extends ActiveFilterModes>(
  updateKey: string,
  updateValue: ActiveFilterState[K],
  filterType: K
): (old: ActiveFilterOrderState) => ActiveFilterOrderState {
  return (old) => {
    const newFilters = {
      ...old[filterType],
      [updateKey]: updateValue,
    };
    return {
      ...old,
      [filterType]: newFilters,
    };
  };
}

function updateKey<
  R extends {
    min: number;
    max: number;
  }
>(
  oldKey: string,
  oldFilterMode: ActiveFilterModes,
  newKey: string,
  /* metricsSummary: ActiveProjectMetricSummary, */
  metricRanges: Record<string, R>,
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >
): (old: ActiveFilterOrderState) => ActiveFilterOrderState {
  return (old) => {
    const oldOrder = old.ordering.indexOf(oldKey);
    if (oldOrder < 0 || !(oldKey in old[oldFilterMode])) {
      // Cannot be deleted.
      return old;
    }

    const existingNewOrder = old.ordering.indexOf(newKey);
    if (existingNewOrder !== -1) {
      // Already exists.
      return old;
    }

    // Generate intermediate state without new filter state inserted.
    const renamedOrdering = [...old.ordering];
    renamedOrdering[oldOrder] = newKey;
    const renamed = {
      ...deleteKey(oldKey, oldFilterMode)(old),
      ordering: renamedOrdering,
    };

    // Check if a metric
    const newMetricRanges = metricRanges[newKey];

    if (newMetricRanges != null) {
      return {
        ...renamed,
        metricFilters: {
          ...renamed.metricFilters,
          [newKey]: newMetricRanges,
        },
      };
    }

    if (newKey in old.tagFilters) {
      return {
        ...renamed,
        tagFilters: {
          ...renamed.tagFilters,
          [newKey]: [],
        },
      };
    }

    return old;
  };
}

function addNewEntry<
  R extends {
    min: number;
    max: number;
  }
>(
  metricRanges: Record<string, R>,
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >
): (old: ActiveFilterOrderState) => ActiveFilterOrderState {
  return (old) => {
    // Try insert new 'metric' key.
    const newMetricEntry = Object.entries(metricRanges).find(
      ([candidate]) => !(candidate in old.metricFilters)
    );
    if (newMetricEntry != null) {
      const [newMetricKey, newMetricRanges] = newMetricEntry;
      if (newMetricRanges != null) {
        return {
          ...old,
          ordering: [...old.ordering, newMetricKey],
          metricFilters: {
            ...old.metricFilters,
            [newMetricKey]: newMetricRanges,
          },
        };
      }
    }

    // Try insert new 'tag' key.
    const newTagKey = Object.keys(defaultFilters.tagFilters).find((candidate) =>
      isEmpty(
        old.tagFilters[candidate as keyof typeof defaultFilters.tagFilters]
      )
    );
    if (newTagKey != null) {
      return {
        ...old,
        ordering: [...old.ordering, newTagKey],
        tagFilters: { ...old.tagFilters, [newTagKey]: [] },
      };
    }

    // Try insert new 'enum' key.
    /* const newEnumEntry = Object.entries(metricsSummary.enums).find( */
    /*   ([candidate]) => !(candidate in old.enumFilters) */
    /* ); */
    /* if (newEnumEntry != null) { */
    /*   const [newEnumKey, newEnumSummary] = newEnumEntry; */
    /*   const enumValues = getEnumList(newEnumSummary, featureHashMap); */
    /*   return { */
    /*     ...old, */
    /*     ordering: [...old.ordering, newEnumKey], */
    /*     enumFilters: { */
    /*       ...old.enumFilters, */
    /*       [newEnumKey]: enumValues, */
    /*     }, */
    /*   }; */
    /* } */

    // Failed to insert correctly.
    return old;
  };
}

type TagKey = keyof GroupedTags;

function ActiveMetricFilter<
  R extends {
    min: number;
    max: number;
  }
>(props: {
  filters: ActiveFilterOrderState;
  setFilters: (
    arg:
      | ActiveFilterOrderState
      | ((old: ActiveFilterOrderState) => ActiveFilterOrderState)
  ) => void;
  metricsSummary: ActiveProjectMetricSummary;
  metricRanges: Record<string, R> | undefined;
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { filters, setFilters, metricsSummary, metricRanges, featureHashMap } =
    props;
  const { allDataTags, allLabelTags } = useAllTags();
  const allTags = { data: [...allDataTags], label: [...allLabelTags] };

  const metricKeys = new Set(Object.keys(metricRanges ?? {}));

  /* const metrics = */
  /*   // Remove any invalid filters. */
  /*   useEffect(() => { */
  /*     const entries = Object.entries(filters.metricFilters); */
  /*     const filteredEntries = entries.filter( */
  /*       ([key]) => metricRanges?.[key] !== undefined */
  /*     ); */
  /*     if (filteredEntries.length !== entries.length) { */
  /*       setFilters({ */
  /*         ...filters, */
  /*         metricFilters: Object.fromEntries(filteredEntries), */
  /*       }); */
  /*     } */
  /*   }, [filters, metricRanges, setFilters]); */

  // Render all active filters, skip metrics that cannot be selected.
  const filterOptions = useMemo(() => {
    if (metricRanges == null) {
      return [];
    }
    // Set of filters that 'exist'
    const exists = new Set<string>(filters.ordering);
    const metricOptions = Object.keys(metricRanges)
      /* .filter((metricKey) => !exists.has(metricKey)) */
      .map((metricKey) => ({ value: metricKey, label: metricKey }));

    const tagOptions = [
      { value: "data", label: "Data Tags" },
      { value: "label", label: "Label Tags" },
    ];
    /* .filter(({ value }) => !exists.has(value)); */
    /* const enumOptions = Object.entries(metricsSummary.enums) */
    /*   .filter(([enumKey]) => !exists.has(enumKey)) */
    /*   .map(([enumKey, enumState]) => ({ */
    /*     value: enumKey, */
    /*     label: getEnumName(enumKey, enumState), */
    /*   })); */
    return [...metricOptions, ...tagOptions];
  }, [filters.ordering, metricRanges, allTags]);

  // We need range information.
  if (metricRanges === undefined) {
    return (
      <Button
        icon={<PlusOutlined />}
        shape="circle"
        size="small"
        type="primary"
        disabled
      />
    );
  }

  return (
    <>
      <Row>
        <Typography.Text strong>Filters:</Typography.Text>
      </Row>
      {filters.ordering.map((filterKey) => {
        const metricRange = metricRanges[filterKey];
        const metricBounds =
          metricRange !== undefined
            ? {
                ...metricRange,
                step: metricRange.max / 100,
              }
            : undefined;
        const metricFilters = filters.metricFilters[filterKey];
        const enumValues = filters.enumFilters[filterKey];
        const tagValues = filters.tagFilters[filterKey as TagKey] || [];

        const filterType: ActiveFilterModes =
          metricRange != null
            ? "metricFilters"
            : tagValues
            ? "tagFilters"
            : "enumFilters";
        const filterLabel =
          filterOptions.find(({ value }) => value === filterKey)?.label ??
          filterKey;

        return (
          <Row align="middle" key={`row_filter_${filterKey}`}>
            <Button
              icon={<MinusOutlined />}
              shape="circle"
              size="small"
              type="ghost"
              onClick={() => setFilters(deleteKey(filterKey, filterType))}
            />
            <Select
              showSearch
              value={filterLabel}
              bordered={false}
              style={{
                width: 200,
                marginLeft: 10,
                marginRight: 10,
                marginTop: 5,
                marginBottom: 5,
              }}
              onChange={(selectedOption) =>
                setFilters(
                  updateKey(
                    filterKey,
                    filterType,
                    selectedOption,
                    /* metricsSummary, */
                    metricRanges,
                    featureHashMap
                  )
                )
              }
              options={filterOptions}
            />
            {metricBounds != null ? (
              <Slider
                range={{ draggableTrack: true }}
                min={metricBounds.min}
                max={metricBounds.max}
                style={{ width: 500 }}
                step={metricBounds.step}
                defaultValue={
                  metricFilters != null
                    ? [metricFilters.min, metricFilters.max]
                    : [metricBounds.min, metricBounds.max]
                }
                onAfterChange={([min, max]: [number, number]) =>
                  setFilters(updateValue(filterKey, { min, max }, filterType))
                }
              />
            ) : (
              <Select
                allowClear
                mode="multiple"
                defaultValue={[...(enumValues ?? tagValues ?? [])]}
                onChange={(newSelection: string[]) =>
                  setFilters(updateValue(filterKey, newSelection, filterType))
                }
                options={allTags[filterKey as TagKey].map((tag) => ({
                  value: tag,
                  label: tag,
                }))}
                style={{ width: 500 }}
              />
            )}
          </Row>
        );
      })}
      <Button
        icon={<PlusOutlined />}
        shape="circle"
        size="small"
        type="primary"
        disabled={metricKeys.size === 0}
        onClick={() => setFilters(addNewEntry(metricRanges, featureHashMap))}
      />
    </>
  );
}

export default ActiveMetricFilter;
