import { useRenderData } from "../streamlit/StreamlitProvider";

import { PagesMenu, Props as PagesMenuProps } from "./pages-menu";

type Components = "PagesMenu";
type Props = PagesMenuProps;

export const EncordActiveComponents = () => {
  const {
    args: { component, props },
  } = useRenderData<{
    component: Components;
    props: Props;
  }>();

  if (component == "PagesMenu") return PagesMenu(props as PagesMenuProps);

  throw `Missing component '${component}'`;
};
