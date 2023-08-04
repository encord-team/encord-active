export const env = import.meta.env.VITE_ENV;
export const local = env === "packaged" || env === "development";
export const apiUrl = env !== "development" ? "" : "http://localhost:8000";
