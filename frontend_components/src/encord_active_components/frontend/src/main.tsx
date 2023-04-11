import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { EncordActiveComponents } from "./components";
import "./index.css";

import { StreamlitProvider } from "./streamlit/StreamlitProvider";

const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <StreamlitProvider>
      <QueryClientProvider client={queryClient}>
        <EncordActiveComponents />
      </QueryClientProvider>
    </StreamlitProvider>
  </React.StrictMode>
);
