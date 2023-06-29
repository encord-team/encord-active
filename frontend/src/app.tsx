import { Spin } from "antd";
import { useState } from "react";
import {
  IntegratedProjectMetadata,
  useIntegratedActiveAPI,
  useLookupProjectsFromUrlList,
} from "./components/james/IntegratedActiveAPI";
import ActiveProjectPage from "./components/james/oss/ActiveProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";

export const App = () => {
  const { data: projects, isLoading } = useLookupProjectsFromUrlList([
    import.meta.env.VITE_API_URL,
  ]);
  const [selectedProject, setSelectedProject] = useState<
    IntegratedProjectMetadata | undefined
  >();
  const queryAPI = useIntegratedActiveAPI(projects ?? {});

  if (isLoading) return <Spin />;
  if (!projects) throw "something bad happened";

  const projectList = Object.values(projects);

  return (
    <div className="p-12 bg-white">
      {selectedProject?.projectHash ? (
        <ActiveProjectPage
          setSelectedProject={(projectHash) =>
            setSelectedProject(
              projectHash === undefined ? undefined : projects[projectHash]
            )
          }
          queryAPI={queryAPI}
          projectHash={selectedProject?.projectHash}
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
