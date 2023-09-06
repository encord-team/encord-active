import * as React from "react";
import { BiInfoCircle } from "react-icons/bi";
import { Spin } from "antd";
import { ScatteredEmbeddings } from "./ExplorerCharts";
import { QueryAPI } from "../Types";
import { InternalFilters } from "./Explorer";
import { loadingIndicator } from "../Spin";

export function ExplorerEmbeddings(props: {
  queryApi: QueryAPI;
  projectHash: string;
  reductionHash: string | undefined;
  filters: InternalFilters;
  setEmbeddingSelection: (
    bounds:
      | {
          x1: number;
          x2: number;
          y1: number;
          y2: number;
        }
      | undefined
  ) => void;
}) {
  const {
    queryApi,
    projectHash,
    reductionHash,
    filters,
    setEmbeddingSelection,
  } = props;
  const { isLoading, data: scatteredEmbeddings } =
    queryApi.useProjectAnalysisReducedEmbeddings(
      projectHash,
      filters.analysisDomain,
      reductionHash ?? "",
      filters.filters,
      { enabled: reductionHash != null }
    );

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
          setEmbeddingSelection={setEmbeddingSelection}
          onReset={() => setEmbeddingSelection(undefined)}
        />
      )}
    </div>
  );
}
