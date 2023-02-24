import { Menu } from "antd";
import { SubMenuType } from "antd/lib/menu/hooks/useItems";
import useResizeObserver from "use-resize-observer";

import { useEffect } from "react";
import { Streamlit } from "streamlit-component-lib";
import { useRenderData } from "../../streamlit/StreamlitProvider";

import classes from "./PagesMenu.module.css";

export const PagesMenu = () => {
  const {
    args: { items },
  } = useRenderData<{ items: SubMenuType[] }>();

  if (!items) throw new Error("`items` prop must be provided");

  const first = items.filter(Boolean)[0];

  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  return (
    <div ref={ref}>
      <Menu
        defaultSelectedKeys={[
          (first.children?.[0]?.key || first.key).toString(),
        ]}
        defaultOpenKeys={[first.key]}
        subMenuOpenDelay={1}
        items={items}
        mode="inline"
        className={`bg-transparent text-xl select-none ${classes["transparent-submenu"]}`}
        onSelect={({ key }) => Streamlit.setComponentValue(key)}
        forceSubMenuRender={true}
      />
    </div>
  );
};
