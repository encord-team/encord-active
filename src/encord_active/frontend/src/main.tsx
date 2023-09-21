import { ErrorBoundary, Provider as RollbarProvider } from "@rollbar/react"; // Provider imports 'rollbar'
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import "antd/dist/reset.css";
import React, { useMemo } from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import {
  Chart,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  SubTitle,
} from "chart.js";
import { App } from "./app";
import { AuthContext, useCreateAuthContext } from "./authContext";
import { apiUrl, env, rollbarConfig } from "./constants";
import { Querier, QuerierContext } from "./hooks/Context";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

function Root() {
  const authContext = useCreateAuthContext();
  const querier = useMemo(
    () => new Querier(apiUrl, authContext.token),
    [authContext.token]
  );

  return (
    <RollbarProvider config={rollbarConfig}>
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          {env === "development" && (
            <ReactQueryDevtools position="bottom-right" />
          )}
          <AuthContext.Provider value={authContext}>
            <QuerierContext.Provider value={querier}>
              <App />
            </QuerierContext.Provider>
          </AuthContext.Provider>
        </QueryClientProvider>
      </ErrorBoundary>
    </RollbarProvider>
  );
}

const router = createBrowserRouter([
  {
    path: "*",
    element: <Root />,
  },
]);

// charts.js register
Chart.register(
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Title,
  SubTitle
);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
