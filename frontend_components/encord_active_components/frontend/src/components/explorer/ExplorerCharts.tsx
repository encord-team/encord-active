import * as React from "react";
import { Column, ColumnConfig } from "@ant-design/plots";
import { useCallback, useMemo } from "react";
import { useProjectAnalysisDistribution } from "../../hooks/queries/useProjectAnalysisDistribution";
import { ExplorerFilterState } from "./ExplorerTypes";

export function MetricDistributionTiny(props: {
  projectHash: string;
  filters: ExplorerFilterState;
}) {
  const { projectHash, filters } = props;
  const { data: distribution } = useProjectAnalysisDistribution(
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
    res.sort((a, b) => Number(a.group) - Number(b.group));
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
