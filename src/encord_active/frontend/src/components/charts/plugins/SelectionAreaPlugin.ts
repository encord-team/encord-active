import { Chart, ChartEvent, Plugin } from "chart.js";
import { EmptyObject } from "chart.js/dist/types/basic";
import { Embedding2DFilter } from "../../../openapi/api";

type SelectionEntry = {
  readonly x: number;
  readonly y: number;
};

export class SelectionAreaPlugin implements Plugin<"scatter", SelectionEntry> {
  id: "selection";

  pStart: [number, number] | null;

  pEnd: [number, number] | null;

  setEmbeddingSelection: (bounds: Embedding2DFilter | undefined) => void;

  reductionHash: string | undefined;

  constructor(
    setEmbeddingSelection: (bounds: Embedding2DFilter | undefined) => void,
    reductionHash: string | undefined
  ) {
    this.id = "selection";
    this.pStart = null;
    this.pEnd = null;
    this.setEmbeddingSelection = setEmbeddingSelection;
    this.reductionHash = reductionHash;
  }

  afterDraw(chart: Chart, args: EmptyObject, options: unknown): void {
    if (this.pStart !== null && this.pEnd !== null) {
      const { ctx } = chart;
      const x1 = chart.scales.x.getPixelForValue(this.pStart[0]);
      const y1 = chart.scales.y.getPixelForValue(this.pStart[1]);
      const x2 = chart.scales.x.getPixelForValue(this.pEnd[0]);
      const y2 = chart.scales.y.getPixelForValue(this.pEnd[1]);
      if (y1 === undefined || y2 === undefined) {
        return;
      }
      const x = Math.min(x1, x2);
      const y = Math.min(y1, y2);
      const w = Math.max(x1, x2) - x;
      const h = Math.max(y1, y2) - y;
      ctx.fillStyle = "#aaaaaaaa";
      ctx.fillRect(x, y, w, h);
    }
  }

  beforeEvent(
    chart: Chart,
    args: {
      event: ChartEvent;
      replay: boolean;
      cancelable: true;
      inChartArea: boolean;
    },
    options: unknown
  ): boolean {
    if (!args.inChartArea) {
      return true;
    }
    if (
      args.event.type === "mousedown" ||
      args.event.type === "mouseup" ||
      args.event.type === "mousemove"
    ) {
      const { x, y } = args.event;
      if (x === null || y === null) {
        return true;
      }
      if (args.event.type === "mousemove" && this.pStart === null) {
        return true;
      }
      // Interaction
      const scaleX = chart.scales.x.getValueForPixel(x);
      const scaleY = chart.scales.y.getValueForPixel(y);
      if (scaleX === undefined || scaleY === undefined) {
        return true;
      }
      if (args.event.type === "mousedown") {
        this.pStart = [scaleX, scaleY];
      } else if (args.event.type === "mousemove") {
        this.pEnd = [scaleX, scaleY];
        chart.render();
      } else if (args.event.type === "mouseup") {
        if (
          this.pStart != null &&
          this.pEnd != null &&
          this.reductionHash != null
        ) {
          const x1 = Math.min(this.pStart[0], this.pEnd[0]);
          const y1 = Math.min(this.pStart[1], this.pEnd[1]);
          const x2 = Math.max(this.pStart[0], this.pEnd[0]);
          const y2 = Math.max(this.pStart[1], this.pEnd[1]);
          this.setEmbeddingSelection({
            x1,
            y1,
            x2,
            y2,
            reduction_hash: this.reductionHash,
          });
        }
        this.pStart = null;
        this.pEnd = null;
        chart.render();
      }
      return false;
    }
    return true;
  }
}
