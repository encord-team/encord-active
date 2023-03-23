import { Menu } from "antd";
import { SubMenuType } from "antd/lib/menu/hooks/useItems";
import useResizeObserver from "use-resize-observer";

import { useEffect, useState } from "react";
import { Streamlit } from "streamlit-component-lib";

import classes from "./PagesMenu.module.css";
import { Project } from "../projects-page";
import { fork } from "radash";

type ViewAll = ["VIEW_ALL_PROJECTS", null];
type SelectProject = ["SELECT_PROJECT", string];
type SelectPage = ["SELECT_PAGE", string];
type Output = ViewAll | SelectProject | SelectPage;

const pushOutput = (output: Output) => Streamlit.setComponentValue(output);

export type Props = {
  items: SubMenuType[];
  projects?: Project[];
  selectedProjectHash?: string;
};
export const PagesMenu = ({
  items,
  projects = [],
  selectedProjectHash,
}: Props) => {
  if (!items) throw new Error("`items` prop must be provided");

  const first = items.filter(Boolean)[0];

  const { ref, height = 0 } = useResizeObserver<HTMLDivElement>();

  const defaultSelectedKey = (first.children?.[0]?.key || first.key).toString();
  const [selectedKey, setSelectedKeys] = useState(defaultSelectedKey);

  useEffect(() => {
    pushOutput(["SELECT_PAGE", selectedKey]);
  }, [selectedKey]);

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const [sandboxProjects, userProjects] = fork(
    projects.filter(({ path }) => !!path),
    ({ sandbox }) => !!sandbox
  );

  return (
    <div ref={ref} className="flex flex-col gap-5">
      <div className="form-control">
        <div className="input-group input-group-vertical">
          <button
            className="btn normal-case justify-start"
            onClick={() => pushOutput(["VIEW_ALL_PROJECTS", null])}
          >
            View all projects
          </button>
          <select
            className="select select-bordered focus:outline-none"
            defaultValue={selectedProjectHash || "disabled"}
            onChange={({ target: { value } }) =>
              pushOutput(["SELECT_PROJECT", value])
            }
          >
            {!selectedProjectHash && (
              <option value="disabled" disabled>
                Select a project
              </option>
            )}
            {userProjects.length ? (
              <optgroup label="User projects">
                {userProjects.map(({ hash, name }) => (
                  <option value={hash} key={hash}>
                    {name}
                  </option>
                ))}
              </optgroup>
            ) : null}
            {sandboxProjects.length ? (
              <optgroup label="Sandbox projects">
                {sandboxProjects.map(({ hash, name }) => (
                  <option value={hash} key={hash}>
                    {name}
                  </option>
                ))}
              </optgroup>
            ) : null}
          </select>
        </div>
      </div>
      <Menu
        defaultSelectedKeys={[defaultSelectedKey]}
        defaultOpenKeys={[first.key]}
        subMenuOpenDelay={1}
        items={items}
        mode="inline"
        className={`bg-transparent text-xl select-none ${classes["transparent-submenu"]}`}
        onSelect={({ key }) => setSelectedKeys(key)}
        forceSubMenuRender={true}
      />
    </div>
  );
};
