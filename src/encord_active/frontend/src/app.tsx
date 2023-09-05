import { Alert } from "antd";
import { useState } from "react";
import { useAuth } from "./authContext";
import { useIntegratedAPI, useProjectsList } from "./components/IntegratedAPI";
import { ProjectPage } from "./components/ProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";
import ErrorBoundary from "antd/lib/alert/ErrorBoundary";
import { Spinner } from "./components/explorer/Spinner";

export const App = () => {
  const { data: projects, isLoading, error } = useProjectsList();
  const [selectedProjectHash, setSelectedProjectHash] = useState<
    string | undefined
  >();
  const { token } = useAuth();
  const queryAPI = useIntegratedAPI(token, projects ?? {});

  if (isLoading)
    return (
      <div className="w-full h-screen flex items-center justify-center">
        <Spinner />
      </div>
    );
  if (error?.response && "detail" in error.response?.data)
    return (
      <Alert
        message={`${error.response.statusText} - ${error?.response.data.detail}`}
        type="error"
      />
    );

  if (!projects) throw "something bad happened";

  const projectList = Object.values(projects);

  return (
    <div className="p-12 bg-white">
      {selectedProjectHash ? (
        <ErrorBoundary
          message={
            "An error occurred rendering the project: " + selectedProjectHash
          }
        >
          <ProjectPage
            queryAPI={queryAPI}
            projectHash={selectedProjectHash}
            projects={Object.values(projects)}
            setSelectedProjectHash={setSelectedProjectHash}
          />
        </ErrorBoundary>
      ) : (
        <ProjectsPage
          projects={projectList}
          onSelectLocalProject={setSelectedProjectHash}
        />
      )}
    </div>
  );
};
