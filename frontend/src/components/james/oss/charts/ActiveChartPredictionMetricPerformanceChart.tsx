import {
  Bar,
  Line,
  ComposedChart,
  Legend,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";
import * as React from "react";
import { useMemo, useState } from "react";
import { formatTooltip, formatTooltipLabel } from "../util/ActiveFormatter";

/**
 * Returns data for the average over the input data.
 * @param data
 */
function getDataAverage(
  data: Readonly<
    Record<
      string,
      ReadonlyArray<{
        readonly m: number;
        readonly n: number;
        readonly a: number;
      }>
    >
  >
): ReadonlyArray<{
  readonly m: number;
  readonly n: number;
  readonly a: number;
}> {
  const lookup: Record<string, { m: number; n: number; a: number }> = {};
  Object.values(data).forEach((array) => {
    array.forEach((entry) => {
      const value = lookup[entry.m];
      if (value !== undefined) {
        value.n += entry.n;
        value.a = (value.a * value.n + entry.a * entry.n) / (entry.n + value.n);
      } else {
        lookup[entry.m] = { ...entry };
      }
    });
  });

  return Object.values(lookup);
}

/**
 * Post-process the data to scale it correctly (n is not scaled by SUM{n})
 * @param selectedData
 */
function tidyData(
  selectedData: ReadonlyArray<{
    readonly m: number;
    readonly n: number;
    readonly a: number;
  }>
): [
  Array<{ readonly m: number; readonly n: number; readonly a: number }>,
  number | null
] {
  const maxN = selectedData.map((v) => v.n).reduce((a, b) => Math.max(a, b), 0);
  const scaled = selectedData.map((entry) => ({ ...entry, n: entry.n / maxN }));
  scaled.sort((a, b) => a.m - b.m);
  const referenceY =
    selectedData.length === 0
      ? null
      : selectedData.map((v) => v.a).reduce((a, b) => a + b) /
        selectedData.length;
  return [scaled, referenceY];
}

function ActiveChartPredictionMetricPerformanceChart(props: {
  data: Readonly<
    Record<
      string,
      ReadonlyArray<{
        readonly m: number;
        readonly n: number;
        readonly a: number;
      }>
    >
  >;
  selectedClass: ReadonlyArray<string> | undefined;
  classDecomposition: boolean | "auto";
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
  scoreLabel: string;
}) {
  const {
    data,
    scoreLabel,
    selectedClass,
    classDecomposition,
    featureHashMap,
  } = props;
  const [barCharts, barRefs, barGroups]: [
    ({ m: number } & Record<string, number>)[],
    Record<string, null | number>,
    string[]
  ] = useMemo(() => {
    const selectedClassSet = new Set(selectedClass);
    const filteredData = Object.fromEntries(
      Object.entries(data).filter(
        ([key]) => selectedClassSet.size === 0 || selectedClassSet.has(key)
      )
    );

    // True or "auto" & selected at least 1 class.
    if (
      classDecomposition &&
      (classDecomposition === true || selectedClassSet.size > 0)
    ) {
      const lookup: Record<string, { m: number } & Record<string, number>> = {};
      const references: Record<string, null | number> = {};
      const filteredEntries = Object.entries(filteredData);

      filteredEntries.forEach(([featureHash, featureBarData]) => {
        const [tidyBarData, tidyRef] = tidyData(featureBarData);
        references[featureHash] = tidyRef;
        tidyBarData.forEach((entry) => {
          let state = lookup[entry.m];
          if (state === undefined) {
            state = { m: entry.m };
            lookup[entry.m] = state;
          }
          state[`${featureHash}n`] = entry.n;
          state[`${featureHash}a`] = entry.a;
        });
      });
      filteredEntries.forEach(([featureHash]) =>
        Object.values(lookup).forEach((entry) => {
          if (!(`${featureHash}n` in entry)) {
            entry[`${featureHash}n`] = 0;
            entry[`${featureHash}a`] = 0;
          }
        })
      );

      if (filteredEntries.length > 1) {
        // Add average column if at least 2 columns have been requested
        const averageData = getDataAverage(filteredData);
        const [tidyAverage, tidyAverageRef] = tidyData(averageData);
        references[""] = tidyAverageRef;
        tidyAverage.forEach((entry) => {
          const state = lookup[entry.m];
          if (state === undefined) {
            throw Error("Average metric performance has bad m group");
          }
          state["n"] = entry.n;
          state["a"] = entry.a;
        });
      }

      const lookupValues: ({ m: number } & Record<string, number>)[] =
        Object.values(lookup);
      lookupValues.sort((a, b) => a.m - b.m);
      return [lookupValues, references, Object.keys(references)];
    } else {
      // Special case 1 value selected, otherwise average
      const filteredEntries = Object.entries(filteredData);
      if (Object.keys(filteredData).length === 1) {
        const [featureHash, data] = filteredEntries[0] ?? ["", []];
        const [featureBar, featureRef] = tidyData(data);
        const formattedBar: ({ m: number } & Record<string, number>)[] =
          featureBar.map((entry) => ({
            m: entry.m,
            [`${featureHash}n`]: entry.n,
            [`${featureHash}a`]: entry.a,
          }));
        const featureRefs: Record<string, number | null> = { "": featureRef };
        return [formattedBar, featureRefs, [featureHash]];
      } else {
        // No decomposition, average selected classes and return the result.
        const selectedData = getDataAverage(filteredData);
        const [avgBar, avgRef]: [
          ({ m: number } & Record<string, number>)[],
          number | null
        ] = tidyData(selectedData);
        const refs: Record<string, number | null> = { "": avgRef };
        return [avgBar, refs, [""]];
      }
    }
  }, [data, selectedClass, classDecomposition]);

  const [hoverKeyword, setHoverKeyword] = useState<undefined | string>();

  return (
    <ResponsiveContainer width="100%" height={500}>
      <ComposedChart data={barCharts}>
        <XAxis
          dataKey="m"
          type="category"
          domain={[0.0, 1.0]}
          label="Bucket"
          tickFormatter={(value: number) => value.toFixed(2)}
        />
        <YAxis
          type="number"
          tickFormatter={(value: number) => value.toFixed(3)}
        />
        {barGroups.map((feature) => {
          const referenceY = barRefs[feature];
          const groupName =
            feature === ""
              ? "Average"
              : featureHashMap[feature]?.name ?? feature;
          const color =
            feature === ""
              ? "#9090ff"
              : featureHashMap[feature]?.color ?? feature;
          const opacityBar =
            hoverKeyword === undefined || hoverKeyword === `${feature}n`
              ? 1.0
              : 0.2;
          const opacityLine =
            hoverKeyword === undefined || hoverKeyword === `${feature}a`
              ? 1.0
              : 0.2;
          return (
            <>
              <Bar
                name={`${groupName} Samples`}
                key={`${feature}n`}
                dataKey={`${feature}n`}
                fill={color}
                fillOpacity={opacityBar}
              />
              <Line
                name={`${groupName} Performance`}
                key={`${feature}a`}
                dataKey={`${feature}a`}
                fill={color}
                fillOpacity={opacityLine}
                stroke={color}
                strokeOpacity={opacityLine}
              />
              {referenceY == null ? null : (
                <ReferenceLine
                  name={`${groupName} Average Performance`}
                  key={`${feature}p`}
                  y={referenceY}
                  fill={color}
                  stroke={color}
                />
              )}
            </>
          );
        })}
        <Legend
          onMouseEnter={(e) => setHoverKeyword(e.dataKey)}
          onMouseLeave={() => setHoverKeyword(undefined)}
        />
        <Tooltip
          formatter={formatTooltip}
          labelFormatter={formatTooltipLabel(`${scoreLabel}: `)}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export default ActiveChartPredictionMetricPerformanceChart;
