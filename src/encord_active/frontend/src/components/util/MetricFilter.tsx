import { useEffect, useMemo } from "react";
import { Button, List, Row, Select, Slider } from "antd";
import { MinusOutlined, PlusOutlined } from "@ant-design/icons";
import {
  EnumSummary,
  MetricSummary,
  ProjectCollaboratorEntry,
  ProjectDomainSummary,
  ProjectTagEntry,
  QuerySummary,
} from "../../openapi/api";
import { FeatureHashMap } from "../Types";
import { EachMetricChartDistributionBar } from "../charts/EachMetricChartDistributionBar";
import { useProjectAnalysisSummary } from "../../hooks/queries/useProjectAnalysisSummary";

export type FilterState = {
  readonly metricFilters: Readonly<Record<string, readonly number[]>>;
  readonly enumFilters: Readonly<Record<string, ReadonlyArray<string>>>;
  readonly ordering: ReadonlyArray<string>;
};

export type FilterModes = "metricFilters" | "enumFilters";

export const DefaultFilters: FilterState = {
  metricFilters: {},
  enumFilters: {},
  ordering: [],
};

function getMetricBounds<
  R extends {
    min: number;
    max: number;
  }
>(
  bounds: R,
  metric: MetricSummary
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
  enumSummary: EnumSummary,
  featureHashMap: FeatureHashMap,
  collaborators: ReadonlyArray<ProjectCollaboratorEntry>,
  tags: ReadonlyArray<ProjectTagEntry>
): ReadonlyArray<string> {
  if (enumSummary.type === "ontology") {
    return Object.keys(featureHashMap);
  } else if (enumSummary.type === "enum") {
    return Object.keys(enumSummary.values ?? {});
  } else if (enumSummary.type === "user_email") {
    return collaborators.map(({ id }) => `${id}`);
  } else if (enumSummary.type === "tags") {
    return tags.map(({ hash }) => hash);
  } else {
    throw Error("Unknown enum state");
  }
}

function getEnumOptions(
  enumSummary: EnumSummary,
  featureHashMap: FeatureHashMap,
  collaborators: ReadonlyArray<ProjectCollaboratorEntry>,
  tags: ReadonlyArray<ProjectTagEntry>
): Array<{ label: string; value: string }> {
  if (enumSummary.type === "ontology") {
    return Object.entries(featureHashMap).map(([featureHash, feature]) => ({
      value: featureHash,
      label: feature.name,
    }));
  } else if (enumSummary.type === "enum") {
    return Object.entries(enumSummary.values ?? {}).map(([value, label]) => ({
      value,
      label: label ?? value,
    }));
  } else if (enumSummary.type === "user_email") {
    return collaborators.map(({ id, email }) => ({
      value: `${id}`,
      label: email,
    }));
  } else if (enumSummary.type === "tags") {
    return tags.map(({ hash, name }) => ({
      value: hash,
      label: name,
    }));
  }
  throw Error("Unknown enum state");
}

function deleteKey(
  deleteKey: string,
  filterType: FilterModes
): (old: FilterState) => FilterState {
  return (old) => {
    const { [deleteKey]: deleted, ...newFilters } = old[filterType];
    const newOrdering = old.ordering.filter((elem) => elem !== deleteKey);

    return {
      ...old,
      ordering: newOrdering,
      [filterType]: newFilters,
    };
  };
}

export function updateValue<K extends FilterModes>(
  updateKey: string,
  updateValue: FilterState[K][string],
  filterType: K
): (old: FilterState) => FilterState {
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

function updateKey(
  oldKey: string,
  newKey: string,
  metricsSummary: ProjectDomainSummary,
  metricRanges: QuerySummary["metrics"] | undefined,
  featureHashMap: FeatureHashMap,
  collaborators: ReadonlyArray<ProjectCollaboratorEntry>,
  tags: ReadonlyArray<ProjectTagEntry>
): (old: FilterState) => FilterState {
  return (old) => {
    const oldMetricSummary = metricsSummary.metrics[oldKey];
    const oldFilterKey =
      oldMetricSummary != null ? "metricFilters" : "enumFilters";
    const oldOrder = old.ordering.indexOf(oldKey);
    if (oldOrder < 0 || !(oldKey in old[oldFilterKey])) {
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
    const { [oldKey]: deleted, ...renamedFilter } = old[oldFilterKey];
    const renamed = {
      ...old,
      ordering: renamedOrdering,
      [oldFilterKey]: renamedFilter,
    };

    // Check if a metric
    const newMetricSummary = metricsSummary.metrics[newKey];
    if (newMetricSummary != null) {
      const newMetricRanges =
        metricRanges === undefined ? undefined : metricRanges[newKey];
      if (newMetricRanges != null) {
        const newBounds = getMetricBounds(newMetricRanges, newMetricSummary);
        const newRange: readonly [number, number] = [
          newBounds.min,
          newBounds.max,
        ];

        return {
          ...renamed,
          metricFilters: {
            ...renamed.metricFilters,
            [newKey]: newRange,
          },
        };
      }
      return old;
    }

    // Check if an enum
    const newEnumSummary = metricsSummary.enums[newKey];
    if (newEnumSummary != null && !(newKey in old.enumFilters)) {
      const newValues = getEnumList(
        newEnumSummary,
        featureHashMap,
        collaborators,
        tags
      );

      return {
        ...renamed,
        enumFilters: {
          ...renamed.enumFilters,
          [newKey]: newValues,
        },
      };
    }

    // Failed to rename key.
    return old;
  };
}

export function addNewEntry(
  metricsSummary: ProjectDomainSummary,
  metricRanges: QuerySummary["metrics"] | undefined,
  featureHashMap: FeatureHashMap,
  collaborators: ReadonlyArray<ProjectCollaboratorEntry>,
  tags: ReadonlyArray<ProjectTagEntry>,
  newMetricName?: string
): (old: FilterState) => FilterState {
  return (old) => {
    // Try insert new 'metric' key.
    let newMetricEntry = Object.entries(metricsSummary.metrics).find(
      ([candidate]) => !(candidate in old.metricFilters)
    );

    if (newMetricName !== undefined) {
      newMetricEntry = Object.entries(metricsSummary.metrics).find(
        ([candidate]) => candidate === newMetricName
      );
    }

    if (newMetricEntry != null) {
      const [newMetricKey, newMetricSummary] = newMetricEntry;
      const newMetricRanges =
        metricRanges === undefined ? undefined : metricRanges[newMetricKey];
      if (newMetricRanges != null && newMetricSummary != null) {
        const newBounds = getMetricBounds(newMetricRanges, newMetricSummary);
        const newRange: readonly [number, number] = [
          newBounds.min,
          newBounds.max,
        ];

        return {
          ...old,
          ordering: [...old.ordering, newMetricKey],
          metricFilters: {
            ...old.metricFilters,
            [newMetricKey]: newRange,
          },
        };
      }
    }

    // Try insert new 'enum' key.
    const newEnumEntry = Object.entries(metricsSummary.enums).find(
      ([candidate]) => !(candidate in old.enumFilters)
    );
    if (newEnumEntry != null) {
      const [newEnumKey, newEnumSummary] = newEnumEntry;
      if (newEnumSummary !== undefined) {
        const enumValues = getEnumList(
          newEnumSummary,
          featureHashMap,
          collaborators,
          tags
        );

        return {
          ...old,
          ordering: [...old.ordering, newEnumKey],
          enumFilters: {
            ...old.enumFilters,
            [newEnumKey]: enumValues,
          },
        };
      }
    }

    // Failed to insert correctly.
    return old;
  };
}

const toFixedNumber = (number: number, precision: number) =>
  parseFloat(number.toFixed(precision));

export function MetricFilter(props: {
  filters: FilterState;
  setFilters: (arg: FilterState | ((old: FilterState) => FilterState)) => void;
  metricsSummary: ProjectDomainSummary;
  metricRanges: QuerySummary["metrics"] | undefined;
  featureHashMap: FeatureHashMap;
  collaborators: ReadonlyArray<ProjectCollaboratorEntry>;
  tags: ReadonlyArray<ProjectTagEntry>;
  projectHash: string;
}) {
  const {
    filters,
    setFilters,
    metricsSummary: rawMetricsSummary,
    metricRanges,
    featureHashMap,
    collaborators,
    tags,
    projectHash,
  } = props;

  // Remove any invalid filters.
  useEffect(() => {
    const entries = Object.entries(filters.metricFilters);
    const filteredEntries = entries.filter(
      ([key]) => rawMetricsSummary.metrics[key] !== undefined
    );
    if (filteredEntries.length !== entries.length) {
      setFilters({
        ...filters,
        metricFilters: Object.fromEntries(filteredEntries),
      });
    }
  }, [filters, rawMetricsSummary, setFilters]);

  const metricsSummary = useMemo(() => {
    if (metricRanges == null) {
      return undefined;
    }
    const metrics = Object.entries(rawMetricsSummary.metrics).filter(
      ([k]) => k in metricRanges
    );

    return { ...rawMetricsSummary, metrics: Object.fromEntries(metrics) };
  }, [metricRanges, rawMetricsSummary]);

  // Render all active filters, skip metrics that cannot be selected.
  const filterOptions = useMemo(() => {
    if (metricsSummary == null) {
      return [];
    }
    // Set of filters that 'exist'
    const exists = new Set<string>(filters.ordering);
    const metricOptions = Object.entries(metricsSummary.metrics)
      .filter(([metricKey]) => !exists.has(metricKey))
      .map(([metricKey, metricState]) => ({
        value: metricKey,
        label: metricState?.title ?? metricKey,
      }));
    const enumOptions = Object.entries(metricsSummary.enums)
      .filter(([enumKey]) => !exists.has(enumKey))
      .map(([enumKey, enumState]) => ({
        value: enumKey,
        label: enumState !== undefined ? enumState.title : enumKey,
      }));

    return [...metricOptions, ...enumOptions];
  }, [filters.ordering, metricsSummary]);

  // all the data we need
  const summary = useProjectAnalysisSummary(projectHash, "data");
  const { data } = summary;
  // We need range information.
  if (metricRanges === undefined || metricsSummary === undefined) {
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
    <List className="gap-2">
      {filters.ordering.map((filterKey) => {
        const metric = metricsSummary.metrics[filterKey];
        const metricRange = metricRanges[filterKey];
        const metricBounds =
          metric !== undefined && metricRange !== undefined
            ? getMetricBounds(metricRange, metric)
            : undefined;
        const metricFilters = filters.metricFilters[filterKey];

        const enumValues = filters.enumFilters[filterKey];
        const enumFilter = metricsSummary.enums[filterKey];

        const filterType = metric != null ? "metricFilters" : "enumFilters";
        const filterLabel =
          metric != null ? metric.title : enumFilter?.title ?? "Error";

        return (
          <Row
            align="middle"
            key={`row_filter_${filterKey}`}
            className="border-y p-4"
          >
            <div className="flex w-full justify-between">
              {filterLabel}
              <Button
                icon={<MinusOutlined />}
                shape="default"
                size="small"
                onClick={() => setFilters(deleteKey(filterKey, filterType))}
              />
            </div>

            <EachMetricChartDistributionBar
              metricsSummary={metricsSummary}
              analysisSummary={data}
              analysisDomain="data"
              projectHash={projectHash}
              featureHashMap={featureHashMap}
              property={filterKey}
            />

            {metricBounds != null ? (
              <Slider
                range={{ draggableTrack: true }}
                min={metricBounds.min}
                max={metricBounds.max}
                className="w-full"
                step={toFixedNumber(metricBounds.step, 2)}
                value={
                  metricFilters != null
                    ? [metricFilters[0], metricFilters[1]]
                    : [
                        toFixedNumber(metricBounds.min, 2),
                        toFixedNumber(metricBounds.max, 2),
                      ]
                }
                // Question: Should it be changed to onAfterChange?
                onChange={(newRange: [number, number]) =>
                  setFilters(updateValue(filterKey, newRange, "metricFilters"))
                }
              />
            ) : (
              <Select
                allowClear
                mode="multiple"
                value={[...(enumValues ?? [])]}
                onChange={(newSelection: string[]) =>
                  setFilters(
                    updateValue(filterKey, newSelection, "enumFilters")
                  )
                }
                options={
                  enumFilter == null
                    ? []
                    : getEnumOptions(
                        enumFilter,
                        featureHashMap,
                        collaborators,
                        tags
                      )
                }
                style={{ width: 500 }}
              />
            )}
          </Row>
        );
      })}
    </List>
  );
}
