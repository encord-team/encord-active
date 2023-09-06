import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { useEffect, useMemo } from "react";
import { QueryContext } from "./Context";
import { apiUrl } from "../constants";

const NO_AUTH_PATTERNS = ["ea-sandbox-static", "storage.googleapis.com"];

export function useImageSrc(
  queryContext: QueryContext,
  url: string | undefined
): string | undefined {
  const itemUrl =
    url === undefined || url.startsWith("http") ? url : `${apiUrl}${url}`;

  const requiresAuth: boolean = useMemo(
    () =>
      queryContext.usesAuth &&
      !NO_AUTH_PATTERNS.some(
        (pattern) => itemUrl === undefined || itemUrl.includes(pattern)
      ),
    [queryContext, itemUrl]
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
