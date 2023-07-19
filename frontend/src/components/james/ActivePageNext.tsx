import { Spin } from "antd";
import { useEffect } from "react";
import { Streamlit } from "streamlit-component-lib";
import useResizeObserver from "use-resize-observer";
import {
  IntegratedProjectMetadata,
  useIntegratedActiveAPI,
  useProjectsList,
} from "./IntegratedActiveAPI";
import ActiveProjectPage from "./oss/ActiveProjectPage";
import ActiveAnalysisDomainTab from "./oss/tabs/ActiveAnalysisDomainTab";
import ActiveSummaryTab from "./oss/tabs/ActiveSummaryTab";

export type Props = {
  project: IntegratedProjectMetadata;
  queryAPI: ActiveQueryAPI;
};

export function ActiveProject({ projectHash, baseUrl, page }: Props) {
  const { data: projects, isLoading } = useProjectsList([baseUrl]);
  const queryAPI = useIntegratedActiveAPI(projects ?? {});
  const { data: projectSummary } = queryAPI.useProjectSummary(projectHash, {
    enabled: !!projects,
  });

  if (isLoading || projectSummary == null) {
    return <Spin />;
  }

  return (
    <div ref={ref}>
      <ActiveProjectPage queryAPI={queryAPI} projectHash={projectHash} />
      {/* <ActiveSummaryTab */}
      {/*   projectHash={projectHash} */}
      {/*   queryAPI={queryAPI} */}
      {/*   metricsSummary={projectSummary["data"]} */}
      {/*   analysisDomain={"data"} */}
      {/* /> */}
    </div>
  );
}
