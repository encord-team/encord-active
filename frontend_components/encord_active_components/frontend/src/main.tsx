import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { useMemo } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import {
  createBrowserRouter,
  Route,
  RouterProvider,
  Routes,
} from "react-router-dom";
import { App } from "./app";
import { apiUrl, env } from "./constants";
import { AuthContext, useCreateAuthContext } from "./authContext";
import "antd/dist/reset.css";
import { Querier, QuerierContext } from "./hooks/Context";

const queryClient = new QueryClient();

function Root() {
  const authContext = useCreateAuthContext();
  const querier = useMemo(() => new Querier(apiUrl, authContext.token), [authContext.token]);

  return (
    <QueryClientProvider client={queryClient}>
      {env === "development" && <ReactQueryDevtools position="bottom-right" />}
      <AuthContext.Provider value={authContext}>
        <QuerierContext.Provider value={querier}>
          <Routes>
            <Route path="/" element={<App />} />
          </Routes>
        </QuerierContext.Provider>
      </AuthContext.Provider>
    </QueryClientProvider >
  );
}

const router = createBrowserRouter([
  {
    path: "*",
    element: <Root />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
