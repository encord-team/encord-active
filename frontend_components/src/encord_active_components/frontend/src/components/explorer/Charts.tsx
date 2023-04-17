import { Histogram, Scatter, ScatterConfig } from "@ant-design/plots";
import { useCallback } from "react";
import { Item2DEmbedding } from "./api";

const BINS = 20;

export const MetricDistribution = ({ values }: { values: number[] }) => {
  const max = Math.max(...values);
  const min = Math.min(...values);
  const binSize = (max - min) / BINS || 1;

  return (
    <Histogram
      data={values.map((value) => ({ value }))}
      binField="value"
      binWidth={binSize}
      meta={{ range: { min, tickInterval: ~~binSize } }}
    />
  );
};

export const ScatteredEmbeddings = ({
  embeddings,
  onSelectionChange,
}: {
  embeddings: Item2DEmbedding[];
  onSelectionChange: (ids: Item2DEmbedding[]) => void;
}) => {
  const onEvent = useCallback<NonNullable<ScatterConfig["onEvent"]>>(
    (_, { type, view }) => {
      if (type === "mouseup" || type === "brush-reset-button:click")
        // @ts-ignore
        onSelectionChange(view.filteredData as Item2DEmbedding[]);
    },
    []
  );

  return (
    <Scatter
      data={embeddings}
      xField="x"
      yField="y"
      colorField="label"
      size={3}
      shape="circle"
      pointStyle={{ fillOpacity: 1 }}
      brush={{
        enabled: true,
        mask: {
          style: { fill: "rgba(255,0,0,0.15)" },
        },
      }}
      onEvent={onEvent}
    />
  );
};
