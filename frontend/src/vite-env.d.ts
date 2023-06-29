/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_ENV: "prod" | "sandbox" | "local";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
