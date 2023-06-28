import {
  Column,
  ColumnConfig,
  Scatter,
  ScatterConfig,
  Tooltip,
} from "@ant-design/plots";
import { useCallback, useEffect, useMemo, useState } from "react";
import { IdValue, Item2DEmbedding, PredictionType } from "./api";
import { bin, group } from "d3-array";
import { hexbin } from "d3-hexbin";
import { scaleLinear } from "d3-scale";
import { counting, fork, max } from "radash";

const BINS = 20;

export const MetricDistributionTiny = ({
  values,
  setSeletedIds,
}: {
  values: IdValue[];
  setSeletedIds: (ids: string[]) => void;
}) => {
  const { bins, columns } = useMemo(() => {
    const bins = bin<IdValue, number>()
      .thresholds(BINS)
      .value((v) => v.value as number)(values);
    const columns = bins.map((items, bin) => ({ value: items.length, bin }));
    return { bins, columns };
  }, [JSON.stringify(values)]);

  const [selectedBins, setSelectedBins] = useState<number[]>([]);

  useEffect(() => {
    if (!values.length) return;
    setSeletedIds(
      selectedBins.length
        ? selectedBins.flatMap((bin) => bins[bin].map(({ id }) => id))
        : values.map(({ id }) => id)
    );
  }, [selectedBins]);

  const onEvent = useCallback<NonNullable<ColumnConfig["onEvent"]>>(
    (_, { type, view }) => {
      if ("mouseup" === type)
        setSelectedBins(
          // @ts-ignore
          (view.filteredData as typeof columns).map(({ bin }) => bin)
        );
    },
    []
  );

  const customContent = useCallback<NonNullable<Tooltip["customContent"]>>(
    (_, hoveredElements) => {
      if (!hoveredElements.length || !hoveredElements[0].data) return null;
      const { x0, x1 } = bins[hoveredElements[0].data.bin];

      return (
        <div className="flex flex-col items-center gap-1 py-2">
          <div className="inline-flex justify-between gap-1 w-full">
            <span>{x0}</span>
            <span>,</span>
            <span>{x1}</span>
          </div>
          <span>Count: {hoveredElements[0].data.value}</span>
        </div>
      );
    },
    [bins]
  );

  return (
    <Column
      className="w-full max-h-12"
      autoFit={true}
      data={columns}
      columnWidthRatio={1}
      yField={"value"}
      xField={"bin"}
      xAxis={false}
      yAxis={false}
      brush={
        bins.length > 1
          ? {
              enabled: true,
              type: "x-rect",
              action: "filter",
              mask: {
                style: { fill: "rgba(255,0,0,0.15)" },
              },
            }
          : {}
      }
      onEvent={onEvent}
      tooltip={{
        customContent,
      }}
    />
  );
};

const fixedFormatter = (value: string | number | null) =>
  value != null ? parseFloat(value.toString()).toFixed(2) : "Missing";

const HEX_BINS = 1000;
const getColor = scaleLinear([0, 1], ["#ef4444", "#22c55e"]);

export const ScatteredEmbeddings = ({
  embeddings,
  onSelectionChange,
  onReset,
  predictionType,
}: {
  embeddings: Item2DEmbedding[];
  onSelectionChange: (ids: Item2DEmbedding[]) => void;
  onReset: () => void;
  predictionType?: PredictionType;
}) => {
  const binnedItems = useMemo(() => {
    if (embeddings.length < HEX_BINS)
      return embeddings.map((emb) => ({ ...emb, size: 1, items: [emb] }));

    const { maxX, minX, maxY, minY } = embeddings.reduce(
      (maxPoints, { x, y }) => {
        maxPoints.maxX = Math.max(x, maxPoints.maxX);
        maxPoints.minX = Math.min(x, maxPoints.minX);
        maxPoints.maxY = Math.max(y, maxPoints.maxY);
        maxPoints.minY = Math.min(y, maxPoints.minY);
        return maxPoints;
      },
      { maxX: 0, maxY: 0, minX: 0, minY: 0 }
    );

    const bins = hexbin<Item2DEmbedding>()
      .x((d) => d.x)
      .y((d) => d.y)
      .radius(((maxX - minX) * (maxY - minY)) / HEX_BINS)(embeddings);

    return bins.map((items) => {
      const counters = counting(items, ({ label }) => label);
      const [label] = max(Object.entries(counters), ([_, value]) => value) || [
        items[0].label,
        1,
      ];

      const bin = { ...items, size: items.length, label, items };

      if (!predictionType) return bin;

      const [correct, inccorect] = fork(items, ({ score }) => !!score);
      const score =
        inccorect.length > 0
          ? correct.length / (correct.length + inccorect.length)
          : 1;

      return { ...bin, score };
    });
  }, [JSON.stringify(embeddings)]);

  const onEvent = useCallback<NonNullable<ScatterConfig["onEvent"]>>(
    (_, { type, view }) => {
      if (["mouseup", "legend-item:click"].includes(type))
        onSelectionChange(
          // @ts-ignore
          view.filteredData.flatMap((bin) => bin.items) as Item2DEmbedding[]
        );
      else if (type === "brush-reset-button:click") onReset();
    },
    [JSON.stringify(binnedItems)]
  );

  const colorConfig = useMemo<{
    colorField: string;
    color?: Parameters<typeof Scatter>[0]["color"];
  }>(() => {
    if (predictionType)
      return {
        colorField: "score",
        color: (datum) => getColor(datum.score ?? 0),
      };

    return { colorField: "label" };
  }, [predictionType]);

  return (
    <Scatter
      {...colorConfig}
      data={binnedItems}
      xField="x"
      yField="y"
      sizeField="size"
      size={[5, 30]}
      shape="circle"
      legend={{
        layout: "vertical",
        position: "right",
        rail: { size: 20, defaultLength: 200 },
        label: {
          formatter: fixedFormatter,
        },
      }}
      pointStyle={{ fillOpacity: 1 }}
      interactions={[{ type: "reset-button", enable: false }]}
      brush={{
        enabled: true,
        mask: {
          style: { fill: "rgba(255,0,0,0.15)" },
        },
      }}
      meta={{
        x: { formatter: fixedFormatter },
        y: { formatter: fixedFormatter },
      }}
      onEvent={onEvent}
    />
  );
};
