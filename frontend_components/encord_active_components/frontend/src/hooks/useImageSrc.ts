import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { useEffect, useMemo } from "react";

const NO_AUTH_PATTERNS = ["ea-sandbox-static", "storage.googleapis.com"];

export function useImageSrc(url: string | undefined): string | undefined {
  const requiresAuth: boolean = useMemo(() => !NO_AUTH_PATTERNS.some(
      (pattern) => url === undefined || url.includes(pattern)
    ), [url]);
  const { data: queryBlob } = useQuery(
    ["ImageSrcAuth", url],
    async () => {
      if (url === undefined) {return undefined;}
      const res = await axios.get(url, { responseType: "blob" });

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
  return requiresAuth ? objectUrl : url;
}
