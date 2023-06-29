const COLORS: ReadonlyArray<string> = [
  "#21e4fc",
  "#00ff00",
  "#ff0000",
  "#ffc322",
  "#e77728",
  "#a436e8",
  "#ed6ef3",
  "#4c6735",
  "#0ac785",
  "#5200bb",
  "#9bce15",
  "#fff6f6",
];

export function getStableColor(key: string): string {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    const chr = key.charCodeAt(i);
    hash = (hash << 5) - hash + chr;
    hash |= 0; // Convert to 32bit integer
  }
  const colorIdx = (hash | 0) % COLORS.length;
  return COLORS[colorIdx] ?? "#000000";
}
