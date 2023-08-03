export const env = import.meta.env.VITE_ENV;
export const apiUrl = env !== "development" ? "" : "http://localhost:8000";
