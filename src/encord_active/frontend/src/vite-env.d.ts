/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ENV: "production" | "packaged" | "sandbox" | "development";
  readonly VITE_ROLLBAR_ACCESS_TOKEN?: string;
  readonly VITE_CODE_VERSION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
