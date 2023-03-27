import { useRenderData } from "../streamlit/StreamlitProvider";

import { PagesMenu, Props as PagesMenuProps } from "./pages-menu";
import { ProjectsPage, Props as ProjectsPageProps } from "./projects-page";

type Components = "PagesMenu" | "ProjectsPage";
type Props = PagesMenuProps | ProjectsPageProps;

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
  throw `Missing component '${component}'`;
};
