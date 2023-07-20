import { Alert, Spin } from "antd";
import { useState } from "react";
import { createAuthContext, useAuth } from "./authContext";
import {
  IntegratedProjectMetadata,
  useIntegratedActiveAPI,
  useProjectsList,
} from "./components/james/IntegratedActiveAPI";
import ActiveProjectPage from "./components/james/oss/ActiveProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";

export const App = () => {
  const { data: projects, isLoading, error } = useProjectsList();
  const [selectedProjectHash, setSelectedProjectHash] = useState<
    string | undefined
  >();
  const { token } = useAuth();
  const queryAPI = useIntegratedActiveAPI(token, projects ?? {});

  if (isLoading)
    return (
      <Spin className="w-full h-screen flex items-center justify-center" />
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
        <ActiveProjectPage
          queryAPI={queryAPI}
          projectHash={selectedProjectHash}
          projects={Object.values(projects)}
          setSelectedProject={setSelectedProjectHash}
        />
      ) : (
        <ProjectsPage
          projects={projectList}
          onSelectLocalProject={setSelectedProjectHash}
        />
      )}
    </div>
  );
};
