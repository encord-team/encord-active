import React from "react";
import ReactDOM from "react-dom/client";
import { PagesMenu } from "./components";
import "./index.css";

import { BrowserRouter, Route, Routes } from "react-router-dom";
import { StreamlitProvider } from "./streamlit/StreamlitProvider";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <StreamlitProvider>
      {/* <BrowserRouter> */}
      {/*   <Routes> */}
      {/* <Route path="pages-menu" element={<PagesMenu />} /> */}
      <PagesMenu />
      {/*   </Routes> */}
      {/* </BrowserRouter> */}
    </StreamlitProvider>
  </React.StrictMode>
);
