import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./app";
import "./index.css";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import {
  createBrowserRouter,
  Route,
  RouterProvider,
  Routes,
} from "react-router-dom";
import { env } from "./constants";
import { AuthContext, createAuthContext } from "./authContext";
import "antd/dist/reset.css";

const queryClient = new QueryClient();

const Root = () => (
  <QueryClientProvider client={queryClient}>
    {env === "development" && <ReactQueryDevtools position="bottom-right" />}
    <AuthContext.Provider value={createAuthContext()}>
      <Routes>
        <Route path="/" element={<App />} />
      </Routes>
    </AuthContext.Provider>
  </QueryClientProvider>
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
