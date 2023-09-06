import {
  Column,
  ColumnConfig,
  Scatter,
  ScatterConfig,
} from "@ant-design/plots";
import { useCallback, useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { QueryContext } from "../../hooks/Context";
import { useProjectAnalysisDistribution } from "../../hooks/queries/useProjectAnalysisDistribution";
import { ExplorerFilterState } from "./ExplorerTypes";

export function MetricDistributionTiny(props: {
  projectHash: string;
  queryContext: QueryContext;
  filters: ExplorerFilterState;
}) {
  const { projectHash, filters, queryContext } = props;
  const { data: distribution } = useProjectAnalysisDistribution(
    queryContext,
    projectHash,
    filters.analysisDomain,
    filters.orderBy
  );

  const onEvent = useCallback<NonNullable<ColumnConfig["onEvent"]>>(
    (_, { type, view }) => {
      /*
      if ("mouseup" === type)
        setSelectedBins(
          // @ts-ignore
          (view.filteredData as typeof columns).map(({ bin }) => bin)
        );
       */
    },
    []
  );

  /* const customContent = useCallback<NonNullable<Tooltip["customContent"]>>(
    (_, hoveredElements) => {
      if (!hoveredElements.length || !hoveredElements[0].data) return null;
      const { x0, x1 } = bins[hoveredElements[0].data.bin];

      return (
        <div className="flex flex-col items-center gap-1 py-2">
          <div className="inline-flex justify-between gap-1 w-full">
            <span>{x0}</span>
            <span>,</span>
            <span>{x1}</span>
          </div>
          <span>Count: {hoveredElements[0].data.value}</span>
        </div>
      );
    },
    [bins]
  ); */

  const data = useMemo(() => {
    const res = [...(distribution?.results ?? [])];
    res.sort((a, b) => (a.group as number) - (b.group as number));
    return res as unknown as Record<string, any>[];
  }, [distribution]);

  return (
    <Column
      className="max-h-12 w-64"
      autoFit
      data={data}
      columnWidthRatio={1}
      yField="count"
      xField="group"
      xAxis={false}
      yAxis={false}
      brush={
        (distribution?.results?.length ?? 0) > 1
          ? {
              enabled: true,
              type: "x-rect",
              action: "filter",
              mask: {
                style: { fill: "rgba(255,0,0,0.15)" },
              },
            }
          : {}
      }
      onEvent={onEvent}
      /* tooltip={{
        customContent,
      }} */
    />
  );
}

const fixedFormatter = (value: string | number | null) =>
  value != null ? parseFloat(value.toString()).toFixed(2) : "Missing";

const HEX_BINS = 1000;
const getColor = scaleLinear([0, 1], ["#ef4444", "#22c55e"]);

export function ScatteredEmbeddings(props: {
  reductionScatter: ProjectAnalysisReductionResult | undefined;
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
  onReset: () => void;
}) {
  const { reductionScatter, setEmbeddingSelection, onReset } = props;

  const onEvent = useCallback<NonNullable<ScatterConfig["onEvent"]>>(
    (_, { type, view }) => {
      if (["mouseup", "legend-item:click"].includes(type)) {
        const bbox = view.coordinateBBox;
        setEmbeddingSelection({
          x1: bbox.x,
          x2: bbox.x + bbox.width,
          y1: bbox.y,
          y2: bbox.y + bbox.height,
        });
      } else if (type === "brush-reset-button:click") {
        onReset();
      }
    },
    [setEmbeddingSelection, onReset]
  );

  const colorConfig = useMemo<{
    colorField: string;
    color?: Parameters<typeof Scatter>[0]["color"];
  }>(() => {
    if (false) {
      // prediction
      return {
        colorField: "score",
        color: (datum) => getColor(datum.score ?? 0),
      };
    }

    return { colorField: "label" };
  }, []);

  return (
    <Scatter
      {...colorConfig}
      autoFit
      data={reductionScatter?.reductions as unknown as Record<string, any>[]}
      xField="x"
      yField="y"
      sizeField="n"
      size={[5, 30]}
      shape="circle"
      legend={{
        layout: "vertical",
        position: "right",
        rail: { size: 20, defaultLength: 200 },
        label: {
          formatter: fixedFormatter,
        },
      }}
      pointStyle={{ fillOpacity: 1 }}
      interactions={[{ type: "reset-button", enable: false }]}
      brush={{
        enabled: true,
        mask: {
          style: { fill: "rgba(255,0,0,0.15)" },
        },
      }}
      meta={{
        x: { formatter: fixedFormatter },
        y: { formatter: fixedFormatter },
      }}
      onEvent={onEvent}
    />
  );
}
