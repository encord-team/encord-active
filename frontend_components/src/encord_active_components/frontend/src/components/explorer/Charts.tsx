import { Histogram } from "@ant-design/plots";
import { IdValue } from "./api";

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
