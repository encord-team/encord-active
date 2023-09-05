import { useMemo } from "react";
import {
  BiInfoCircle,
} from "react-icons/bi";
import { Spinner } from "./Spinner";
import {ScatteredEmbeddings } from "./ExplorerCharts";
import {QueryAPI} from "../Types";
import {InternalFilters} from "./Explorer";
import {Spin} from "antd";
import {loadingIndicator} from "../Spin";

export function ExplorerEmbeddings(props: {
  queryApi: QueryAPI;
  projectHash: string;
  filters: InternalFilters;
  setEmbeddingSelection: (bounds: {
    x1: number;
    x2: number;
    y1: number;
    y2: number;
  } | undefined) => void;
}) {
  const {
    queryApi,
    projectHash,
    filters,
    setEmbeddingSelection,
  } = props;
  const {
    data: reductionHashes,
  } = queryApi.useProjectListEmbeddingReductions(projectHash);
  const reductionHash: string | undefined = useMemo(
    () => reductionHashes === undefined
        || reductionHashes.results.length === 0
        ? undefined : reductionHashes.results[0].hash,
    [reductionHashes]
  );
  const {
    isLoading,
    data: scatteredEmbeddings
  } = queryApi.useProjectAnalysisReducedEmbeddings(
    projectHash,
    filters.analysisDomain,
    reductionHash ?? "",
    filters.filters,
    { enabled: reductionHash != null}
  );

  return !isLoading && !scatteredEmbeddings?.reductions?.length ? (
    <div className="alert shadow-lg h-fit">
      <div>
        <BiInfoCircle className="text-base"/>
        <span>2D embedding are not available for this project. </span>
      </div>
    </div>
  ) : (
    <div className="w-full flex  h-96 [&>*]:flex-1 items-center">
      {isLoading ? (
        <div className="absolute" style={{left: "50%"}}>
          <Spin indicator={loadingIndicator}/>
        </div>
      ) : (
        <ScatteredEmbeddings
          reductionScatter={scatteredEmbeddings}
          predictionType={filters.predictionType}
          setEmbeddingSelection={setEmbeddingSelection}
          onReset={() => setEmbeddingSelection()}
        />
      )}
    </div>
  );
}
