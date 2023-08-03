/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ENV: "production" | "packaged" | "sandbox" | "development";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
