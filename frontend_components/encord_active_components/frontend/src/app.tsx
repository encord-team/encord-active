import * as React from "react";
import { Alert, Spin } from "antd";
import { useState } from "react";
import ErrorBoundary from "antd/lib/alert/ErrorBoundary";
import { useAuth } from "./authContext";
import { useIntegratedAPI, useProjectsList } from "./components/IntegratedAPI";
import { ProjectPage } from "./components/ProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";
import { loadingIndicator } from "./components/Spin";

export function App() {
  const { data: projects, isLoading, error } = useProjectsList();
  const [selectedProjectHash, setSelectedProjectHash] = useState<
    string | undefined
  >();
  const { token } = useAuth();
  const queryAPI = useIntegratedAPI(token, projects?.projects_dict ?? {});

  if (isLoading) {return <Spin indicator={loadingIndicator} />;}
  if (error?.response && "detail" in error?.response?.data)
    {return (
      <Alert
        message={`${error.response.statusText} - ${error?.response.data.detail}`}
        type="error"
      />
    );}

  if (!projects) {throw "something bad happened";}

  const projectList = Object.values(projects);

  return (
    <div className="bg-white p-12">
      {selectedProjectHash ? (
        <ErrorBoundary
          message={
            `An error occurred rendering the project: ${  selectedProjectHash}`
          }
        >
          <ProjectPage
            queryAPI={queryAPI}
            projectHash={selectedProjectHash}
            projects={projects.projects_list}
            setSelectedProjectHash={setSelectedProjectHash}
          />
        </ErrorBoundary>
      ) : (
        <ProjectsPage
          projects={projects.projects_list}
          onSelectLocalProject={setSelectedProjectHash}
        />
      )}
    </div>
  );
}
