import { PredictionItem } from "../../openapi/api";

export function calculateTruePositiveSet(
  preview: Pick<PredictionItem, "annotation_iou_bounds">,
  iou: number
): ReadonlySet<string> {
  const tpSet = new Set();
  Object.entries(preview.annotation_iou_bounds ?? {}).forEach(([key, v]) => {
    if (v != null) {
      const [keyIOU, keyIOUBound] = v;
      if (keyIOU >= iou && keyIOUBound < iou) {
        tpSet.add(key);
      }
    }
  });
  return new Set(tpSet) as ReadonlySet<string>;
}
