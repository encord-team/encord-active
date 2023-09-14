import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useProjectAnalysisDistribution } from "../../hooks/queries/useProjectAnalysisDistribution";
import { ExplorerFilterState } from "./ExplorerTypes";
import { formatTooltip } from "../util/Formatter";
import { usePredictionAnalysisDistribution } from "../../hooks/queries/usePredictionAnalysisDistribution";

export const ExplorerDistribution = ExplorerDistributionRaw; // FIXME: React.memo

function ExplorerDistributionRaw(props: {
  projectHash: string;
  predictionHash: string | undefined;
  filters: ExplorerFilterState;
  addFilter: (
    domain: "data" | "annotation",
    metric: string,
    min: number,
    max: number
  ) => void;
}) {
  const { projectHash, predictionHash, filters, addFilter } = props;
  const { data: distributionProject } = useProjectAnalysisDistribution(
    projectHash,
    filters.analysisDomain,
    filters.orderBy,
    undefined,
    filters.filters,
    { enabled: predictionHash === undefined }
  );
  const { data: distributionPrediction } = usePredictionAnalysisDistribution(
    projectHash,
    predictionHash ?? "",
    filters.predictionOutcome,
    filters.iou,
    filters.orderBy,
    undefined,
    filters.filters,
    { enabled: predictionHash !== undefined }
  );
  const distribution =
    predictionHash === undefined ? distributionProject : distributionPrediction;

  const barData = useMemo(() => {
    if (distribution == null) {
      return [];
    }
    return distribution.results.map(({ group, count }) => ({
      group: Number(group),
      count,
    }));
  }, [distribution]);

  const [selection, setSelection] = useState<
    | {
        min: number;
        max: number;
      }
    | undefined
  >();

  return (
    <ResponsiveContainer width="100%" height={100}>
      <BarChart
        data={barData}
        className="active-chart select-none"
        onMouseDown={(chart) => {
          const { activeLabel } = chart;
          return activeLabel !== undefined
            ? setSelection({
                min: Number(activeLabel),
                max: Number(activeLabel),
              })
            : undefined;
        }}
        onMouseMove={({ activeLabel }) =>
          activeLabel !== undefined && selection !== undefined
            ? setSelection((val) =>
                val === undefined
                  ? undefined
                  : { ...val, max: Number(activeLabel) }
              )
            : undefined
        }
        onMouseUp={() => {
          setSelection(undefined);
          if (selection !== undefined) {
            addFilter(
              filters.analysisDomain,
              filters.orderBy,
              Math.min(selection.min, selection.max),
              Math.max(selection.min, selection.max)
            );
          }
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="group"
          type="number"
          padding="no-gap"
          domain={["dataMin", "dataMax"]}
        />
        <YAxis tick={false} />
        <Tooltip formatter={formatTooltip} />
        <Bar dataKey="count" isAnimationActive={false} fill="#0000bf" />
        {selection !== undefined ? (
          <ReferenceArea x1={selection.min} x2={selection.max} />
        ) : null}
      </BarChart>
    </ResponsiveContainer>
  );
}
