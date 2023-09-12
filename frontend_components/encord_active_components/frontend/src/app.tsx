import * as React from "react";
import ErrorBoundary from "antd/lib/alert/ErrorBoundary";
import { Navigate, Route, Routes, redirect, useNavigate } from "react-router";
import { ProjectPage } from "./components/ProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";

export function App() {
  // FIXME: make variable conditionally loaded from parent
  const encordDomain = "https://app.encord.com";

  const navigate = useNavigate();
  const selectProject = (projectHash?: string) =>
    navigate(projectHash ? `/projects/${projectHash}` : "/");

  return (
    <div className="bg-white p-12">
      <Routes>
        <Route
          path="/"
          element={<ProjectsPage onSelectLocalProject={selectProject} />}
        />
        <Route
          path="/projects/:projectHash" element={<Navigate to="./summary" replace={true} />} />
        <Route
          path="/projects/:projectHash/:tab"
          element={
            <ErrorBoundary message="An error occurred rendering the project">
              <ProjectPage
                encordDomain={encordDomain}
                setSelectedProjectHash={selectProject}
              />
            </ErrorBoundary>
          }
        />
      </Routes>
    </div>
  );
}
