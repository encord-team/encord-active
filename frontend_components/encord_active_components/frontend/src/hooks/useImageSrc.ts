import { useQuery } from "@tanstack/react-query";
import axios from "axios";

const NO_AUTH_PATTERNS = [
  "ea-sandbox-static",
  "storage.googleapis.com",
  "s3.amazon.com",
];

export const useImageSrc = (url: string) =>
  useQuery([url], async () => {
    if (NO_AUTH_PATTERNS.some((pattern) => url.includes(pattern))) return url;

    const res = await axios.get(url, { responseType: "blob" });
    return URL.createObjectURL(res.data);
  });
