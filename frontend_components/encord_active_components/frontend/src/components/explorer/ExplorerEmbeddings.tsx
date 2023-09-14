import { Alert, Button, Row, Spin } from "antd";
import { useCallback, useMemo, useState } from "react";
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
  Query2DEmbeddingScatterPoint,
} from "../../openapi/api";
import {
  featureHashToColor,
  formatTick,
  formatTooltip,
} from "../util/Formatter";
import useRateLimit from "../../hooks/useRateLimit";
import { MetricFilter } from "../util/MetricFilter";

const EmbeddingScatterAxisDomain = ["dataMin - 1", "dataMax + 1"];
const EmbeddingScatterXAxisPadding = { left: 10, right: 10 };
const EmbeddingScatterZAxisRange = [5, 500];

const getColorPrediction = scaleLinear([0, 1], ["#ef4444", "#22c55e"]);

type SelectionType =
  | {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
    }
  | undefined;

export function ExplorerEmbeddings(props: {
  projectHash: string;
  predictionHash: string | undefined;
  reductionHash: string | undefined;
  reductionHashLoading: boolean;
  featureHashMap: Parameters<typeof MetricFilter>[0]["featureHashMap"];
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
    featureHashMap,
  } = props;

  const [selectionRaw, setSelection] = useState<SelectionType>();
  const selection = useRateLimit(selectionRaw, 16);

  return (
    <ExplorerEmbeddingsMemo
      projectHash={projectHash}
      predictionHash={predictionHash}
      reductionHash={reductionHash}
      reductionHashLoading={reductionHashLoading}
      filters={filters}
      setEmbeddingSelection={setEmbeddingSelection}
      selection={selection}
      setSelection={setSelection}
      featureHashMap={featureHashMap}
    />
  );
}

export const ExplorerEmbeddingsMemo = ExplorerEmbeddingsRaw; // FIXME: re-enable React.memo

function ExplorerEmbeddingsRaw(props: {
  projectHash: string;
  predictionHash: string | undefined;
  reductionHash: string | undefined;
  reductionHashLoading: boolean;
  featureHashMap: Parameters<typeof MetricFilter>[0]["featureHashMap"];
  filters: ExplorerFilterState;
  setEmbeddingSelection: (bounds: Embedding2DFilter | undefined) => void;
  selection: SelectionType;
  setSelection: (
    selection: SelectionType | ((old: SelectionType) => SelectionType)
  ) => void;
}) {
  const {
    projectHash,
    predictionHash,
    reductionHash,
    reductionHashLoading,
    filters,
    setEmbeddingSelection,
    selection,
    setSelection,
    featureHashMap,
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
      (
        entry: Query2DEmbeddingScatterPoint | PredictionQueryScatterPoint,
        index
      ) => {
        let fill = "#4a4aee";
        if ("tp" in entry) {
          fill = getColorPrediction(entry.tp / (entry.fp + entry.fn));
        } else if (entry.fh !== "") {
          fill =
            featureHashMap[entry.fh]?.color ?? featureHashToColor(entry.fh);
        }

        return {
          ...entry,
          fill,
          stroke: "#0f172a",
          value: entry.n,
          index,
        };
      }
    );
  }, [scatteredEmbeddings, featureHashMap]);

  const hoveredReduction = useMemo(
    () =>
      reductionWithColor.length === null || hoveredIndex === undefined
        ? null
        : { ...reductionWithColor[hoveredIndex], fill: "#e2e8f0" },
    [reductionWithColor, hoveredIndex]
  );

  const onMouseDown = useCallback(
    (
      elem: null | { xValue?: number | undefined; yValue?: number | undefined }
    ) => {
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
    },
    [setSelection]
  );

  const noActiveSelection = selection === undefined;
  const onMouseMove = useCallback(
    (
      elem: null | { xValue?: number | undefined; yValue?: number | undefined }
    ) => {
      if (elem == null || noActiveSelection) {
        return undefined;
      }
      const { xValue, yValue } = elem;

      return xValue !== undefined && yValue !== undefined
        ? setSelection((val) =>
            val === undefined ? undefined : { ...val, x2: xValue, y2: yValue }
          )
        : undefined;
    },
    [noActiveSelection, setSelection]
  );

  const onMouseUp = useCallback(() => {
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
  }, [reductionHash, selection, setSelection, setEmbeddingSelection]);

  const onScatterHoverMouseEnter = useCallback(
    ({ index }: (typeof reductionWithColor)[number]) => {
      if (noActiveSelection) {
        setHoveredIndex(index);
      }
    },
    [noActiveSelection]
  );

  const onScatterHoverMouseLeave = useCallback(() => {
    setHoveredIndex(undefined);
  }, []);

  const scatterData = useMemo(
    () => [...reductionWithColor, hoveredReduction],
    [reductionWithColor, hoveredReduction]
  );

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
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="x"
            type="number"
            name="x"
            domain={EmbeddingScatterAxisDomain}
            padding={EmbeddingScatterXAxisPadding}
            tickFormatter={formatTick}
          />
          <YAxis
            dataKey="y"
            type="number"
            name="y"
            domain={EmbeddingScatterAxisDomain}
            tickFormatter={formatTick}
          />
          <ZAxis type="number" dataKey="n" range={EmbeddingScatterZAxisRange} />
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
          <Scatter
            onMouseEnter={onScatterHoverMouseEnter}
            onMouseLeave={onScatterHoverMouseLeave}
            data={scatterData}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </>
  );
}
