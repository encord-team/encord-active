import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { Provider as RollbarProvider, ErrorBoundary } from "@rollbar/react"; // Provider imports 'rollbar'
import { App } from "./app";
import "./index.css";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import {
  createBrowserRouter,
  Route,
  RouterProvider,
  Routes,
} from "react-router-dom";
import { env, rollbarConfig } from "./constants";
import { AuthContext, createAuthContext } from "./authContext";
import "antd/dist/antd.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

const Root = () => (
  <RollbarProvider config={rollbarConfig}>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        {env === "development" && (
          <ReactQueryDevtools position="bottom-right" />
        )}
        <AuthContext.Provider value={createAuthContext()}>
          <Routes>
            <Route path="/" element={<App />} />
          </Routes>
        </AuthContext.Provider>
      </QueryClientProvider>
    </ErrorBoundary>
  </RollbarProvider>
);

const router = createBrowserRouter([
  {
    path: "*",
    element: <Root />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
