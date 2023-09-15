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
import { useMemo, useState } from "react";
import {
  featureHashToColor,
  formatTooltip,
  formatTooltipLabel,
} from "../util/Formatter";
import {
  QueryMetricPerformance,
  QueryMetricPerformanceEntry,
} from "../../openapi/api";

/**
 * Returns data for the average over the input data.
 * @param data
 */
function getDataAverage(
  data: Readonly<
    Record<string, readonly Readonly<QueryMetricPerformanceEntry>[] | undefined>
  >
): readonly Readonly<QueryMetricPerformanceEntry>[] {
  const lookup: Record<string, { m: number; n: number; as: number }> = {};
  Object.values(data).forEach((array) => {
    (array ?? []).forEach((entry) => {
      const value = lookup[entry.m];
      if (value !== undefined) {
        value.n += entry.n;
        value.as += entry.a * entry.n;
      } else {
        lookup[entry.m] = { m: entry.m, n: entry.n, as: entry.a * entry.n };
      }
    });
  });

  return Object.values(lookup).map(({ as, n, m }) => ({ m, n, a: as / n }));
}

/**
 * Post-process the data to sort order issues.
 * @param selectedData
 */
function tidyData(
  selectedData: readonly Readonly<QueryMetricPerformanceEntry>[]
): [readonly Readonly<QueryMetricPerformanceEntry>[], number | null] {
  const sorted = [...selectedData];
  sorted.sort((a, b) => a.m - b.m);
  const referenceY =
    selectedData.length === 0
      ? null
      : selectedData.map((v) => v.a).reduce((a, b) => a + b) /
        selectedData.length;

  return [sorted, referenceY];
}

export function ChartPredictionMetricPerformanceChart(props: {
  data: QueryMetricPerformance["precision"] | QueryMetricPerformance["fns"];
  selectedClass: ReadonlyArray<string> | undefined;
  classDecomposition: boolean | "auto";
  featureHashMap: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
  scoreLabel: string;
  showDistributionBar: boolean;
}) {
  const {
    data,
    scoreLabel,
    selectedClass,
    classDecomposition,
    featureHashMap,
    showDistributionBar,
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
        const [tidyBarData, tidyRef] = tidyData(featureBarData ?? []);
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
          state.n = entry.n;
          state.a = entry.a;
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
        const [featureBar, featureRef] = tidyData(data ?? []);
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
        const [avgBar, avgRef] = tidyData(selectedData);
        const refs: Record<string, number | null> = { "": avgRef };

        return [[...avgBar], refs, [""]];
      }
    }
  }, [data, selectedClass, classDecomposition]);

  const [hoverKeyword, setHoverKeyword] = useState<undefined | string>();

  return (
    <ResponsiveContainer width="100%" height={500}>
      <ComposedChart data={barCharts} className="active-chart">
        <XAxis
          dataKey="m"
          type={barCharts.length > 1 ? "number" : "category"}
          domain={
            barCharts.length > 1
              ? [barCharts[0].m, barCharts[barCharts.length - 1].m]
              : undefined
          }
          padding="no-gap"
          label={{
            value: scoreLabel,
            angle: 0,
            position: "insideBottom",
            offset: -3,
          }}
          tickFormatter={(value: number) => value.toFixed(2)}
        />
        <YAxis
          yAxisId="performance"
          type="number"
          tickFormatter={(value: number) => value.toFixed(3)}
        />
        <YAxis yAxisId="samples" type="number" orientation="right" />
        {barGroups.map((feature) => {
          const referenceY = barRefs[feature];
          const groupName =
            feature === ""
              ? "Average"
              : featureHashMap[feature]?.name ?? feature;
          const color =
            feature === ""
              ? "#9090ff"
              : featureHashMap[feature]?.color ?? featureHashToColor(feature);
          const opacityBar =
            hoverKeyword === undefined || hoverKeyword === `${feature}n`
              ? 0.75
              : 0.1;
          const opacityLine =
            hoverKeyword === undefined || hoverKeyword === `${feature}a`
              ? 1.0
              : 0.1;

          return (
            <>
              {showDistributionBar ? (
                <Bar
                  name={`${groupName} Samples`}
                  key={`${feature}n`}
                  dataKey={`${feature}n`}
                  yAxisId="samples"
                  fill={color}
                  fillOpacity={opacityBar}
                />
              ) : null}
              <Line
                name={`${groupName} Performance`}
                key={`${feature}a`}
                dataKey={`${feature}a`}
                yAxisId="performance"
                fill={color}
                fillOpacity={opacityLine}
                stroke={color}
                strokeOpacity={opacityLine}
                connectNulls
              />
              {referenceY == null ? null : (
                <ReferenceLine
                  name={`${groupName} Average Performance`}
                  key={`${feature}p`}
                  yAxisId="performance"
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
