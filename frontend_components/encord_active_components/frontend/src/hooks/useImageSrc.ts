import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { useEffect, useMemo } from "react";
import { apiUrl } from "../constants";
import { useQuerier } from "./Context";

const NO_AUTH_PATTERNS = ["ea-sandbox-static", "storage.googleapis.com"];

export function useImageSrc(
  url: string | undefined
): string | undefined {
  const querier = useQuerier()
  const itemUrl =
    url === undefined || url.startsWith("http") ? url : `${apiUrl}${url}`;

  const requiresAuth: boolean = useMemo(
    () =>
      querier.usesAuth &&
      !NO_AUTH_PATTERNS.some(
        (pattern) => itemUrl === undefined || itemUrl.includes(pattern)
      ),
    [querier, itemUrl]
  );
  const { data: queryBlob } = useQuery(
    ["ImageSrcAuth", itemUrl],
    async () => {
      if (itemUrl === undefined) {
        return undefined;
      }
      const res = await axios.get(itemUrl, { responseType: "blob" });

      return res.data as Blob | MediaSource;
    },
    {
      enabled: requiresAuth,
    }
  );
  const objectUrl = useMemo(() => {
    if (requiresAuth && queryBlob !== undefined) {
      return URL.createObjectURL(queryBlob);
    }
    return undefined;
  }, [requiresAuth, queryBlob]);
  useEffect(() => {
    if (objectUrl !== undefined) {
      return () => URL.revokeObjectURL(objectUrl);
    }
    return undefined;
  }, [objectUrl]);
  return requiresAuth ? objectUrl : itemUrl;
}
