export function toDataItemID(annotation_or_data_id: string): string {
  return annotation_or_data_id.split("_").slice(0, 2).join("_");
}

export function toAnnotationHash(
  annotation_id: string,
  isPrediction: boolean
): string {
  return annotation_id
    .split("_")
    .slice(2, isPrediction ? -1 : undefined)
    .join("_");
}

export function toPredictionTy(
  prediction_id: string
): "TP" | "FP" | "FN" | undefined {
  const components = prediction_id.split("_");
  if (components.length > 3) {
    const predictionTy = components[components.length - 1];
    if (
      predictionTy === "TP" ||
      predictionTy === "FP" ||
      predictionTy === "FN"
    ) {
      return predictionTy;
    } else {
      throw Error(`Unknown prediction type: ${predictionTy}`);
    }
  }
  return undefined;
}
