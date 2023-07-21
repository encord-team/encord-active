export function formatTooltip<T, N>(value: T, name: N): [T | string, N] {
  if (typeof value === "number") {
    const valueStr = value.toFixed(4);
    const indexDP = valueStr.indexOf(".");
    if (indexDP > 5) {
      return [valueStr.substring(0, indexDP), name];
    } else {
      let i = valueStr.length - 1;
      while (i > 0 && valueStr[i] === "0") {
        i -= 1;
      }
      if (valueStr[i] === ".") {
        i -= 1;
      }
      return [valueStr.substring(0, i + 1), name];
    }
  }
  return [value, name];
}

export function formatTooltipLabel<T>(prefix?: string): (value: T) => string {
  return (value) => {
    const [t] = formatTooltip(value, null);
    return (prefix ?? "") + t;
  };
}
