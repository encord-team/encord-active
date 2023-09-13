import * as React from "react";
import { Alert, Button, Row, Spin } from "antd";
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
  ZAxis,
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
  reductionHashLoading: boolean;
  filters: ExplorerFilterState;
  setEmbeddingSelection: (bounds: Embedding2DFilter | undefined) => void;
}) {
  const {
    projectHash,
    predictionHash,
    reductionHash,
    reductionHashLoading,
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
    reductionHashLoading ||
    (predictionHash === undefined ? isLoadingProject : isLoadingPrediction);
  const scatteredEmbeddings:
    | PredictionQuery2DEmbedding
    | Query2DEmbedding
    | undefined =
    predictionHash === undefined
      ? scatteredEmbeddingsProject
      : scatteredEmbeddingsPrediction;

  const [hoveredIndex, setHoveredIndex] = useState<number | undefined>();

  const reductionWithColor = useMemo(() => {
    if (scatteredEmbeddings == null) {
      return [];
    }
    return scatteredEmbeddings.reductions.map(
      (entry: QueryScatterPoint | PredictionQueryScatterPoint, index) => {
        const fill =
          "tp" in entry
            ? getColorPrediction(entry.tp / (entry.fp + entry.fn))
            : "#4a4aee";

        return {
          ...entry,
          fill,
          stroke: "#0f172a",
          value: entry.n,
          index,
        };
      }
    );
  }, [scatteredEmbeddings]);

  const hoveredReduction = useMemo(
    () =>
      reductionWithColor.length === null || hoveredIndex === undefined
        ? null
        : { ...reductionWithColor[hoveredIndex], fill: "#e2e8f0" },
    [reductionWithColor, hoveredIndex]
  );

  const [selection, setSelection] = useState<
    | {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
      }
    | undefined
  >();

  if (reductionHash === undefined && !reductionHashLoading) {
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
      <Row className="h-96 w-full" justify="center" align="middle">
        <Spin indicator={loadingIndicator} tip="Loading Embedding Plot" />
      </Row>
    );
  }

  return (
    <>
      {filters.filters.data?.reduction !== undefined ||
      filters.filters.annotation?.reduction !== undefined ? (
        <Button
          className="absolute top-3 right-3 z-40"
          onClick={(e) => {
            setEmbeddingSelection(undefined);
            e.preventDefault();
          }}
        >
          Reset Filter
        </Button>
      ) : null}
      <ResponsiveContainer width="100%" height={384}>
        <ScatterChart
          className="active-chart select-none"
          onMouseDown={(elem) => {
            if (elem == null) {
              return;
            }
            const { xValue, yValue } = elem;
            if (xValue !== undefined && yValue !== undefined) {
              setSelection({
                x1: xValue,
                y1: yValue,
                x2: xValue,
                y2: yValue,
              });
            }
          }}
          onMouseMove={(elem) => {
            if (elem == null) {
              return null;
            }
            const { xValue, yValue } = elem;

            return xValue !== undefined &&
              yValue !== undefined &&
              selection !== undefined
              ? setSelection((val) =>
                  val === undefined
                    ? undefined
                    : { ...val, x2: xValue, y2: yValue }
                )
              : undefined;
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
            domain={["dataMin - 1", "dataMax + 1"]}
            padding={{ left: 10, right: 10 }}
          />
          <YAxis
            dataKey="y"
            type="number"
            name="y"
            domain={["dataMin - 1", "dataMax + 1"]}
          />
          <ZAxis type="number" dataKey="n" range={[5, 500]} />

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
            <Scatter
              onMouseEnter={({ index }: (typeof reductionWithColor)[number]) =>
                setHoveredIndex(index)
              }
              onMouseLeave={() => setHoveredIndex(undefined)}
              data={[...reductionWithColor, hoveredReduction]}
              isAnimationActive={false}
            />
          ) : null}
        </ScatterChart>
      </ResponsiveContainer>
    </>
  );
}
