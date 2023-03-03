import React from "react";
import ReactDOM from "react-dom/client";
import { EncordActiveComponents } from "./components";
import "./index.css";

import { StreamlitProvider } from "./streamlit/StreamlitProvider";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <StreamlitProvider>
      <EncordActiveComponents />
    </StreamlitProvider>
  </React.StrictMode>
);
