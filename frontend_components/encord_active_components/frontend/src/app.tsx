import * as React from "react";
import { useState } from "react";
import ErrorBoundary from "antd/lib/alert/ErrorBoundary";
import { ProjectPage } from "./components/ProjectPage";
import { ProjectsPage } from "./components/ProjectsPage";

export function App() {
  const [selectedProjectHash, setSelectedProjectHash] = useState<
    string | undefined
  >();
  // FIXME: make variable conditionally loaded from parent
  const encordDomain = "https://app.encord.com";

  return (
    <div className="bg-white p-12">
      {selectedProjectHash ? (
        <ErrorBoundary
          message={`An error occurred rendering the project: ${selectedProjectHash}`}
        >
          <ProjectPage
            encordDomain={encordDomain}
            projectHash={selectedProjectHash}
            setSelectedProjectHash={setSelectedProjectHash}
          />
        </ErrorBoundary>
      ) : (
        <ProjectsPage
          onSelectLocalProject={setSelectedProjectHash}
        />
      )}
    </div>
  );
}
