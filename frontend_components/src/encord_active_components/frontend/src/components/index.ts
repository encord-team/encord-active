import { useRenderData } from "../streamlit/StreamlitProvider";

import { Explorer, Props as ExplorerProps } from "./explorer";
import { PagesMenu, Props as PagesMenuProps } from "./pages-menu";
import { ProjectsPage, Props as ProjectsPageProps } from "./projects-page";
import { Active, Props as ActiveProps } from "./james/ActivePageNext";

type Components = "PagesMenu" | "ProjectsPage" | "Explorer" | "Active";
type Props = PagesMenuProps | ProjectsPageProps | ExplorerProps | ActiveProps;

export const EncordActiveComponents = () => {
  const {
    args: { component, props },
  } = useRenderData<{
    component: Components;
    props: Props;
  }>();

  if (component == "PagesMenu") return PagesMenu(props as PagesMenuProps);
  if (component == "ProjectsPage")
    return ProjectsPage(props as ProjectsPageProps);
  if (component == "Explorer") return Explorer(props as ExplorerProps);
  if (component == "Active") return Active(props as ActiveProps);
  throw `Missing component '${component}'`;
};
