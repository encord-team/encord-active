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
  const [selectedProject, setSelectedProject] = useState<
    IntegratedProjectMetadata | undefined
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
      {selectedProject?.projectHash ? (
        <ActiveProjectPage
          queryAPI={queryAPI}
          projectHash={selectedProject?.projectHash}
          projects={Object.values(projects)}
          setSelectedProject={(projectHash) =>
            setSelectedProject(
              projectHash === undefined ? undefined : projects[projectHash]
            )
          }
        />
      ) : (
        <ProjectsPage
          projects={projectList}
          onSelectLocalProject={(projectHash) =>
            setSelectedProject(projects[projectHash])
          }
        />
      )}
    </div>
  );
};
