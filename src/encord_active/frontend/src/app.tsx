import ErrorBoundary from "antd/lib/alert/ErrorBoundary";
import { Navigate, Route, Routes, useNavigate } from "react-router";
import { ProjectPage } from "./components/ProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";

export function App() {
  // FIXME: make variable conditionally loaded from parent
  const encordDomain = "https://app.encord.com";

  const navigate = useNavigate();
  const selectProject = (projectHash?: string) =>
    navigate(projectHash ? `/projects/${projectHash}` : "/");

  return (
    <div className="flex h-full flex-col">
      <Routes>
        <Route
          path="/"
          element={<ProjectsPage onSelectLocalProject={selectProject} />}
        />
        <Route
          path="/projects/:projectHash"
          element={<Navigate to="./explorer" replace />}
        />
        <Route
          path="/projects/:projectHash/:tab/:previewItem?"
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
