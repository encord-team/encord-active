import * as React from "react";
import { useMemo, useState } from "react";
import ErrorBoundary from "antd/lib/alert/ErrorBoundary";
import { useAuth } from "./authContext";
import { ProjectPage } from "./components/ProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";
import { QueryContext } from "./hooks/Context";
import { apiUrl } from "./constants";

export function App() {
  const [selectedProjectHash, setSelectedProjectHash] = useState<
    string | undefined
  >();
  const { token } = useAuth();
  const queryContext = useMemo(() => new QueryContext(apiUrl, token), [token]);

  // FIXME: make variable conditionally loaded from parent
  const encordDomain = "https://app.encord.com";

  return (
    <div className="bg-white p-12">
      {selectedProjectHash ? (
        <ErrorBoundary
          message={`An error occurred rendering the project: ${selectedProjectHash}`}
        >
          <ProjectPage
            queryContext={queryContext}
            encordDomain={encordDomain}
            projectHash={selectedProjectHash}
            setSelectedProjectHash={setSelectedProjectHash}
          />
        </ErrorBoundary>
      ) : (
        <ProjectsPage
          queryContext={queryContext}
          onSelectLocalProject={setSelectedProjectHash}
        />
      )}
    </div>
  );
}
