import { Menu } from "antd";
import { ItemType } from "antd/lib/menu/hooks/useItems";
import useResizeObserver from "use-resize-observer";

import { useEffect } from "react";
import { Streamlit } from "streamlit-component-lib";
import { useRenderData } from "../../streamlit/StreamlitProvider";

import classes from "./PagesMenu.module.css";

export const PagesMenu = () => {
  const {
    args: { items },
  } = useRenderData<{ items: ItemType[] }>();
  const [first] = items
    ?.map((item) => item?.key?.toString())
    .filter(Boolean) as string[];

  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  if (!first || !items) throw new Error("`items` prop must be provided");

  return (
    <div ref={ref}>
      <Menu
        defaultSelectedKeys={[first]}
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
