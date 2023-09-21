import { Alert, Button, Row, Spin } from "antd";
import { useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { Scatter } from "react-chartjs-2";
import { ChartData, ChartOptions } from "chart.js";
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
import { featureHashToColor } from "../util/Formatter";
import { FeatureHashMap } from "../Types";
import { SelectionAreaPlugin } from "../charts/plugins/SelectionAreaPlugin";

const getColorPrediction = scaleLinear([0, 1], ["#ef4444", "#22c55e"]);

export const ExplorerEmbeddings = ExplorerEmbeddingsRaw; // FIXME: re-enable React.memo

function ExplorerEmbeddingsRaw(props: {
  projectHash: string;
  predictionHash: string | undefined;
  reductionHash: string | undefined;
  reductionHashLoading: boolean;
  featureHashMap: FeatureHashMap;
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

  const chartData = useMemo((): ChartData<
    "scatter",
    { x: number; y: number }[]
  > => {
    if (scatteredEmbeddings === undefined) {
      return {
        datasets: [],
      };
    }
    const datasetMap = new Map<
      string,
      {
        data: { x: number; y: number }[];
        pointBackgroundColor: string[];
        pointRadius: number[];
      }
    >();
    let minRadius = 100000;
    let maxRadius = 1;
    scatteredEmbeddings.reductions.forEach(
      (point: Query2DEmbeddingScatterPoint | PredictionQueryScatterPoint) => {
        let key = "";
        let fill = "#4a4aee";
        if ("tp" in point) {
          fill = getColorPrediction(point.tp / (point.fp + point.fn));
          key = "Prediction";
        } else if (point.fh !== "") {
          fill =
            featureHashMap[point.fh]?.color ?? featureHashToColor(point.fh);
          key = point.fh;
        } else {
          key = "Data";
        }
        const value = datasetMap.get(key) ?? {
          data: [],
          pointBackgroundColor: [],
          pointRadius: [],
        };
        if (!datasetMap.has(key)) {
          datasetMap.set(key, value);
        }
        value.data.push({ x: point.x, y: point.y });
        value.pointBackgroundColor.push(fill);
        value.pointRadius.push(point.n);
        minRadius = Math.min(minRadius, point.n);
        maxRadius = Math.max(maxRadius, point.n);
      }
    );
    const datasetMapKeys = [...datasetMap.keys()];
    datasetMapKeys.sort();
    const scaleRadius = (value: number): number => {
      if (value <= 3) {
        return 2;
      }
      const d = (value - minRadius) / (maxRadius - minRadius);

      return 3 + d * 10;
    };

    return {
      datasets: datasetMapKeys.map((key) => ({
        label: featureHashMap[key]?.name ?? key,
        data: datasetMap.get(key)?.data ?? [],
        pointBackgroundColor: datasetMap.get(key)?.pointBackgroundColor ?? [],
        pointRadius: (datasetMap.get(key)?.pointRadius ?? []).map(scaleRadius),
        backgroundColor: featureHashMap[key]?.color ?? featureHashToColor(key),
        xAxisID: "x",
        yAxisID: "y",
      })),
    };
  }, [featureHashMap, scatteredEmbeddings]);

  const chartOptions = useMemo(
    (): ChartOptions<"scatter"> => ({
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: {
            boxWidth: 10,
          },
        },
        tooltip: {
          mode: "point",
        },
      },
      maintainAspectRatio: false,
      events: ["mousedown", "mousemove", "mouseup", "click"],
    }),
    []
  );

  const chartPlugins = useMemo(
    () => [new SelectionAreaPlugin(setEmbeddingSelection, reductionHash)],
    [setEmbeddingSelection, reductionHash]
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
        <Spin indicator={loadingIndicator} />
      </Row>
    );
  }
  return (
    <div className="h-96 w-full">
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
      <div className="relative h-96 w-full">
        <Scatter
          options={chartOptions}
          plugins={chartPlugins}
          data={chartData}
        />
      </div>
    </div>
  );
}
