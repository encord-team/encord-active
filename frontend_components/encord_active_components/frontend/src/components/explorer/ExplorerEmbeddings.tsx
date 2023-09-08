import * as React from "react";
import { BiInfoCircle } from "react-icons/bi";
import { Spin } from "antd";
import { ScatteredEmbeddings } from "./ExplorerCharts";
import { loadingIndicator } from "../Spin";
import { useProjectAnalysisReducedEmbeddings } from "../../hooks/queries/useProjectAnalysisReducedEmbeddings";
import { ExplorerFilterState } from "./ExplorerTypes";
import { usePredictionAnalysisReducedEmbeddings } from "../../hooks/queries/usePredictionAnalysisReducedEmbeddings";
import { Embedding2DFilter } from "../../openapi/api";

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
  const scatteredEmbeddings =
    predictionHash === undefined
      ? scatteredEmbeddingsProject
      : scatteredEmbeddingsPrediction;

  return !isLoading && !scatteredEmbeddings?.reductions?.length ? (
    <div className="alert h-fit shadow-lg">
      <div>
        <BiInfoCircle className="text-base" />
        <span>2D embedding are not available for this project. </span>
      </div>
    </div>
  ) : (
    <div className="flex h-96  w-full items-center [&>*]:flex-1">
      {isLoading ? (
        <div className="absolute" style={{ left: "50%" }}>
          <Spin indicator={loadingIndicator} tip="Loading Embedding Plot" />
        </div>
      ) : (
        <ScatteredEmbeddings
          reductionScatter={scatteredEmbeddings}
          reductionHash={reductionHash}
          predictionHash={predictionHash}
          setEmbeddingSelection={setEmbeddingSelection}
          onReset={() => setEmbeddingSelection(undefined)}
        />
      )}
    </div>
  );
}
