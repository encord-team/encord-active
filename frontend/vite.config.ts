import {defineConfig, loadEnv, UserConfig} from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }): UserConfig => {
  process.env = {...process.env, ...loadEnv(mode, process.cwd(), "")};
  return {
    build: {
      outDir: "build",
    },
    optimizeDeps: {
      esbuildOptions: {
        // Node.js global to browser globalThis
        define: {
          global: "globalThis",
        },
      },
      needsInterop: ["react-csv"],
    },
    css: {
      preprocessorOptions: {
        less: {
          javascriptEnabled: true, // needed for ant design less files
        },
      },
    },
    define: {
      "process.env": process.env,
    },
    plugins: [
      react(),
    ],
  };
});
