import { Configuration } from "rollbar";

export const env = import.meta.env.VITE_ENV;
export const local = env === "packaged" || env === "development";
export const apiUrl = env !== "development" ? "" : "http://localhost:8000";

export const rollbarConfig = {
  enabled: env !== "development",
  accessToken: import.meta.env.VITE_ROLLBAR_ACCESS_TOKEN ?? "",
  captureUncaught: true,
  captureUnhandledRejections: true,
  reportLevel: "info",
  payload: {
    environment: env,
    client: {
      javascript: {
        code_version: import.meta.env.VITE_CODE_VERSION ?? "unknown",
        source_map_enabled: true,
      },
    },
  },
  ignoredMessages: [],
  // GDPR compliance
  scrubTelemetryInputs: true,
  captureIp: "anonymize",
} satisfies Configuration;

export const Colors = {
  purple: "#5658dd",
  darkGray: "#434343",
};
export const GRID_MIN_COUNT = 1;
export const GRID_MAX_COUNT = 6;
