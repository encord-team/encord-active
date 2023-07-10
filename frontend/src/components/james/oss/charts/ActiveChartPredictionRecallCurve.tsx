import * as React from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ScatterChart,
  Scatter,
} from "recharts";
import { useMemo, useState } from "react";
import { formatTooltip } from "../util/ActiveFormatter";

function ActiveChartPredictionRecallCurve(props: {
  data:
    | undefined
    | Record<
        string,
        Array<{
          readonly p: number;
          readonly r: number;
        }>
      >;
  featureHashMap?: Record<
    string,
    { readonly color: string; readonly name: string }
  >;
}) {
  const { data, featureHashMap } = props;

  const processedData = useMemo(() => {
    if (data === undefined) {
      return undefined;
    }
    return Object.fromEntries(
      Object.entries(data).map(([key, values]) => {
        let prefixMaxP = 0.0;
        let maxR = 0.0;
        let newValues = values.map(({ p, r }) => {
          prefixMaxP = Math.max(p, prefixMaxP);
          maxR = Math.max(r, maxR);
          return {
            p: prefixMaxP,
            r,
          };
        });
        if (maxR !== 1.0 && prefixMaxP !== 0.0) {
          if (maxR !== 9.99) {
            newValues = [
              {
                r: 1,
                p: 0,
              },
              {
                r: maxR + 0.01,
                p: 0,
              },
              ...newValues,
            ];
          } else {
            newValues = [
              {
                r: maxR + 0.01,
                p: 0,
              },
              ...newValues,
            ];
          }
        }
        return [key, newValues];
      })
    );
  }, [data]);

  const [hoverKeyword, setHoverKeyword] = useState<string | undefined>();

  return (
    <ResponsiveContainer width="100%" height={500}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="r" name="Recall" type="number" domain={[0.0, 1.0]} />
        <YAxis dataKey="p" name="Precision" type="number" domain={[0.0, 1.0]} />
        <Tooltip formatter={formatTooltip} labelFormatter={() => null} />
        <Legend
          onMouseEnter={(e: { value: string }) => setHoverKeyword(e.value)}
          onMouseLeave={() => setHoverKeyword(undefined)}
        />
        {processedData == null
          ? null
          : Object.entries(processedData).map(([featureHash, prCurve]) => {
              const name =
                featureHashMap != null
                  ? featureHashMap[featureHash]?.name ?? featureHash
                  : featureHash;
              return (
                <Scatter
                  name={name}
                  id={name}
                  key={featureHash}
                  data={prCurve}
                  fill={
                    (featureHashMap != null
                      ? featureHashMap[featureHash]?.color
                      : null) ?? "#8884d8"
                  }
                  fillOpacity={
                    hoverKeyword === undefined || hoverKeyword === name
                      ? 1.0
                      : 0.3
                  }
                  stroke={
                    (featureHashMap != null
                      ? featureHashMap[featureHash]?.color
                      : null) ?? "#8884d8"
                  }
                  strokeOpacity={
                    hoverKeyword === undefined || hoverKeyword === name
                      ? 1.0
                      : 0.3
                  }
                  line={{
                    stroke:
                      (featureHashMap != null
                        ? featureHashMap[featureHash]?.color
                        : null) ?? "#8884d8",
                    strokeWidth: 2,
                  }}
                  lineType="joint"
                />
              );
            })}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export default ActiveChartPredictionRecallCurve;
