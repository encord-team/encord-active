import React from "react";
import ReactDOM from "react-dom/client";
import { PagesMenu } from "./components";
import "./index.css";

import { StreamlitProvider } from "./streamlit/StreamlitProvider";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <StreamlitProvider>
      <PagesMenu />
    </StreamlitProvider>
  </React.StrictMode>
);
