import * as React from "react";
import { Alert, Spin } from "antd";
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { scaleLinear } from "d3-scale";
import { loadingIndicator } from "../Spin";
import { useProjectAnalysisReducedEmbeddings } from "../../hooks/queries/useProjectAnalysisReducedEmbeddings";
import { ExplorerFilterState } from "./ExplorerTypes";
import { usePredictionAnalysisReducedEmbeddings } from "../../hooks/queries/usePredictionAnalysisReducedEmbeddings";
import {
  Embedding2DFilter,
  PredictionQuery2DEmbedding,
  PredictionQueryScatterPoint,
  Query2DEmbedding,
  QueryScatterPoint,
} from "../../openapi/api";
import { formatTooltip } from "../util/Formatter";

const getColorPrediction = scaleLinear([0, 1], ["#ef4444", "#22c55e"]);

export function ExplorerEmbeddings(props: {
  projectHash: string;
  predictionHash: string | undefined;
  reductionHash: string | undefined;
  filters: ExplorerFilterState;
  setEmbeddingSelection: (bounds: Embedding2DFilter | undefined) => void;
}) {
  const {
    projectHash,
    predictionHash,
    reductionHash,
    filters,
    setEmbeddingSelection,
  } = props;
  const { isLoading: isLoadingProject, data: scatteredEmbeddingsProject } =
    useProjectAnalysisReducedEmbeddings(
      projectHash,
      filters.analysisDomain,
      reductionHash ?? "",
      undefined,
      filters.filters,
      { enabled: reductionHash != null && predictionHash === undefined }
    );
  const {
    isLoading: isLoadingPrediction,
    data: scatteredEmbeddingsPrediction,
  } = usePredictionAnalysisReducedEmbeddings(
    projectHash,
    predictionHash ?? "",
    filters.predictionOutcome,
    filters.iou,
    reductionHash ?? "",
    undefined,
    filters.filters,
    { enabled: reductionHash != null && predictionHash !== undefined }
  );
  const isLoading =
    predictionHash === undefined ? isLoadingProject : isLoadingPrediction;
  const scatteredEmbeddings:
    | PredictionQuery2DEmbedding
    | Query2DEmbedding
    | undefined =
    predictionHash === undefined
      ? scatteredEmbeddingsProject
      : scatteredEmbeddingsPrediction;

  const reductionWithColor = useMemo(() => {
    if (scatteredEmbeddings == null) {
      return [];
    }
    return scatteredEmbeddings.reductions.map(
      (entry: QueryScatterPoint | PredictionQueryScatterPoint) => {
        const fill =
          "tp" in entry
            ? getColorPrediction(entry.tp / (entry.fp + entry.fn))
            : "#4a4aee";

        return {
          ...entry,
          fill,
          value: entry.n,
        };
      }
    );
  }, [scatteredEmbeddings]);

  const [selection, setSelection] = useState<
    | {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
      }
    | undefined
  >();

  if (reductionHash === undefined) {
    return (
      <Alert
        message="2D embedding are not available for this project."
        type="warning"
        showIcon
        className="mb-4"
      />
    );
  }

  if (isLoading) {
    return (
      <Spin
        indicator={loadingIndicator}
        tip="Loading Embedding Plot"
        className="h-400 w-full"
      />
    );
  }

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart
        className="active-chart"
        onMouseDown={({ xValue, yValue }) =>
          xValue !== undefined && yValue !== undefined
            ? setSelection({
                x1: xValue,
                y1: yValue,
                x2: xValue,
                y2: yValue,
              })
            : undefined
        }
        onMouseMove={(state) => {
          if (
            state != null &&
            state.xValue != null &&
            state.yValue != null &&
            selection != null
          ) {
            const { xValue, yValue } = state;
            setSelection((val) => val && { ...val, x2: xValue, y2: yValue });
          }
        }}
        onMouseUp={() => {
          setSelection(undefined);
          if (selection !== undefined && reductionHash !== undefined) {
            setEmbeddingSelection({
              reduction_hash: reductionHash,
              x1: Math.min(selection.x1, selection.x2),
              x2: Math.max(selection.x1, selection.x2),
              y1: Math.min(selection.y1, selection.y2),
              y2: Math.max(selection.y1, selection.y2),
            });
          }
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="x"
          type="number"
          name="x"
          domain={["dataMin - 10", "dataMax + 10"]}
        />
        <YAxis
          dataKey="y"
          type="number"
          name="y"
          domain={["dataMin - 10", "dataMax + 10"]}
        />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          formatter={formatTooltip}
        />
        {selection !== undefined ? (
          <ReferenceArea
            x1={selection.x1}
            x2={selection.x2}
            y1={selection.y1}
            y2={selection.y2}
          />
        ) : undefined}
        {reductionWithColor != null ? (
          <Scatter data={reductionWithColor} isAnimationActive={false} />
        ) : null}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
