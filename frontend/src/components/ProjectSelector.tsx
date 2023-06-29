import { fork } from "radash";
import { IntegratedProjectMetadata } from "./james/IntegratedActiveAPI";

export type Props = {
  projects: readonly IntegratedProjectMetadata[];
  selectedProjectHash: string;
  onViewAllProjects: JSX.IntrinsicElements["button"]["onChange"];
  onSelectedProjectChange: (projectHash: string) => void;
};

export const ProjectSelector = ({
  projects,
  selectedProjectHash,
  onViewAllProjects,
  onSelectedProjectChange,
}: Props) => {
  const [sandboxProjects, userProjects] = fork(
    projects.filter(({ downloaded }) => downloaded),
    ({ sandbox }) => !!sandbox
  );

  return (
    <div className="flex flex-col gap-5 pb-2 min-w-fit">
      <div className="form-control">
        <div className="input-group">
          <button
            className="btn normal-case justify-start"
            onClick={onViewAllProjects}
          >
            View all projects
          </button>
          <select
            className="select select-bordered focus:outline-none"
            defaultValue={selectedProjectHash || "disabled"}
            onChange={({ target: { value } }) => onSelectedProjectChange(value)}
          >
            {!selectedProjectHash && (
              <option value="disabled" disabled>
                Select a project
              </option>
            )}
            {userProjects.length ? (
              <optgroup label="User projects">
                {userProjects.map(({ projectHash, title }) => (
                  <option value={projectHash} key={projectHash}>
                    {title}
                  </option>
                ))}
              </optgroup>
            ) : null}
            {sandboxProjects.length ? (
              <optgroup label="Sandbox projects">
                {sandboxProjects.map(({ projectHash, title }) => (
                  <option value={projectHash} key={projectHash}>
                    {title}
                  </option>
                ))}
              </optgroup>
            ) : null}
          </select>
        </div>
      </div>
    </div>
  );
};
