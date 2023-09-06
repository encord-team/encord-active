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

/*
    cyrb53a (c) 2023 bryc (github.com/bryc)
    License: Public domain. Attribution appreciated.
*/
const cyrb53a = (str: string, seed: number = 0) => {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for(let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 0x85ebca77);
    h2 = Math.imul(h2 ^ ch, 0xc2b2ae3d);
  }
  h1 ^= Math.imul(h1 ^ (h2 >>> 15), 0x735a2d97);
  h2 ^= Math.imul(h2 ^ (h1 >>> 15), 0xcaf649a9);
  h1 ^= h2 >>> 16; h2 ^= h1 >>> 16;
  return 2097152 * (h2 >>> 0) + (h1 >>> 11);
};

export function featureHashToColor(featureHash: string)  {
  // Run a hash function to mix the bits as we only use the low bits.
  const hash = cyrb53a(featureHash) | 0;

  // Convert to random color using hsl format.
  // h was most bits, s & l add minor variation between a small set of values to ensure colors remain in 'good' range.
  const h = ((hash % 90) * 4);
  const s = ((((hash / 90) | 0) % 3) + 70) * 10;
  const l = ((((hash / 45) | 0) % 2) + 2) * 25;
  return `hsl(${h}, ${s}%, ${l}%)`
}