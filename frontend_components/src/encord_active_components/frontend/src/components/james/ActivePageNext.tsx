import { Spin } from "antd";
import { useEffect } from "react";
import { Streamlit } from "streamlit-component-lib";
import useResizeObserver from "use-resize-observer";
import {
  useIntegratedActiveAPI,
  useLookupProjectsFromUrlList,
} from "./IntegratedActiveAPI";
import ActiveProjectPage from "./oss/ActiveProjectPage";
import ActiveAnalysisDomainTab from "./oss/tabs/ActiveAnalysisDomainTab";
import ActiveSummaryTab from "./oss/tabs/ActiveSummaryTab";

export type Props = {
  projectHash: string;
  baseUrl: string;
};

export function Active({ projectHash, baseUrl, page }: Props) {
  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();
  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const { data: projects, isLoading } = useLookupProjectsFromUrlList([baseUrl]);
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
