import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import {useEffect, useMemo, useState} from "react";
import * as url from "url";

const NO_AUTH_PATTERNS = ["ea-sandbox-static", "storage.googleapis.com"];

export function useImageSrc(url: string): string | undefined {
  const requiresAuth: boolean = useMemo(() => {
    return !(NO_AUTH_PATTERNS.some((pattern) => url.includes(pattern)));
  }, [])
  const { data: queryBlob} = useQuery(["ImageSrcAuth", url], async () => {
    const res = await axios.get(url, { responseType: "blob" });
    return res.data as Blob | MediaSource;
  }, {
    enabled: requiresAuth
  });
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
  }, [objectUrl]);
  return requiresAuth ? objectUrl : url;
}


