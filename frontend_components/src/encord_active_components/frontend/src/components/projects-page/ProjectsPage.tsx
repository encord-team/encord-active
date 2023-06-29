import emptyUrl from "../../../assets/empty.svg";
import importUrl from "../../../assets/import.svg";
import encordImportUrl from "../../../assets/encord-import.svg";
import fileImageUrl from "../../../assets/file-image.svg";
import annotationsUrl from "../../../assets/annotations.svg";
import classesUrl from "../../../assets/classes.svg";
import DEFAUL_PROJECT_IMAGE from "../../../assets/default_project_image.webp";

import { Streamlit } from "streamlit-component-lib";
import { classy } from "../../helpers/classy";
import { fork } from "radash";

type Env = "prod" | "local";

export type Project = {
  name: string;
  hash: string;
  imageUrl?: string;
  path?: string;
  sandbox?: boolean;
  stats: {
    dataUnits: number;
    labels: number;
    classes: number;
  };
};

type Output = [
  (
    | "SELECT_SANDBOX_PROJECT"
    | "SELECT_USER_PROJECT"
    | "IMPORT_ENCORD"
    | "IMPORT_COCO"
    | "INIT"
  ),
  string
];

const pushOutput = (output: Output) => Streamlit.setComponentValue(output);

export type Props = { projects: Project[]; env: Env };
export const ProjectsPage = ({ projects = [], env }: Props) => {
  const [sandboxProjects, userProjects] = fork(
    projects,
    ({ sandbox }) => !!sandbox
  );

  return (
    <div className="h-max p-5 flex flex-col gap-5">
      {env !== "prod" && (
        <>
          <h1 className="font-medium text-3xl">Let’s get started!</h1>
          <div className="flex flex-row gap-5 flex-wrap">
            <NewProjectButton
              title="Import from Encord Annotate"
              description="Bring in existing Encord project"
              iconUrl={encordImportUrl}
            />
            <NewProjectButton
              title="Import a COCO project"
              description="Bring in your COCO projects"
              iconUrl={importUrl}
            />
            <NewProjectButton
              title="Initialize from directory"
              description="Upload all images within a folder"
              iconUrl={importUrl}
            />
          </div>
        </>
      )}
      <h2 className="font-light text-xl text-neutral-700">Your projects</h2>
      <div className="flex flex-wrap gap-5">
        {userProjects.length ? (
          userProjects.map((project) => (
            <ProjectCard
              key={project.hash}
              project={project}
              onClick={() => pushOutput(["SELECT_USER_PROJECT", project.hash])}
            />
          ))
        ) : (
          <ProjectNotFoundCard />
        )}
      </div>
      {env !== "prod" && sandboxProjects.length && (
        <>
          <h2 className="font-light text-xl text-neutral-700">
            View a sandbox project
          </h2>
          <div className="flex flex-wrap gap-5">
            {sandboxProjects
              .sort((a, b) => -!!a.path - -!!b.path)
              .map((project) => (
                <ProjectCard
                  key={project.hash}
                  project={project}
                  showDownloadedBadge={true}
                  onClick={() =>
                    pushOutput(["SELECT_SANDBOX_PROJECT", project.hash])
                  }
                />
              ))}
          </div>
        </>
      )}
    </div>
  );
};

const NewProjectButton = ({
  title,
  description,
  iconUrl,
  onClick,
}: {
  title: string;
  description: string;
  iconUrl: string;
  onClick?: JSX.IntrinsicElements["button"]["onClick"];
}) => {
  const disabled = !onClick;
  const containerProps = disabled
    ? {
        className: "tooltip",
        "data-tip": "Coming soon, please use the CLI",
      }
    : {};

  return (
    <div {...containerProps}>
      <button
        className={classy(
          "btn btn-ghost normal-case flex felx-row gap-3 justify-start w-96 h-28 border-1 border-zinc-50 p-3.5",
          { "shadow-lg": !disabled }
        )}
        onClick={onClick}
        disabled={disabled}
      >
        <div className="bg-zinc-50 w-20	h-20 rounded-md flex items-center justify-center">
          <img src={iconUrl} alt="import-project" className="rounded" />
        </div>
        <div className="flex flex-col items-start gap-1">
          <span
            className={classy("font-semibol text-sm", {
              "text-gray-500": disabled,
            })}
          >
            {title}
          </span>
          {description && (
            <span className="font-normal text-xs text-gray-400">
              {description}
            </span>
          )}
        </div>
      </button>
    </div>
  );
};

const ProjectCard = ({
  project,
  showDownloadedBadge = false,
  onClick,
}: {
  project: Project;
  showDownloadedBadge?: boolean;
  onClick: ButtonCardProps["onClick"];
}) => (
  <ButtonCard onClick={onClick}>
    <figure className="max-h-36 rounded">
      <img
        src={
          project.imageUrl
            ? document.referrer + project.imageUrl
            : DEFAUL_PROJECT_IMAGE
        }
        alt={project.name}
      />
    </figure>
    <div className="card-body w-full p-0 justify-between gap-1">
      <h2 className="card-title text-sm line-clamp-2">{project.name}</h2>
      <div className="flex flex-col">
        <ProjectStat
          title={"Dataset"}
          value={project.stats.dataUnits}
          iconUrl={fileImageUrl}
        />
        <ProjectStat
          title={"Annotations"}
          value={project.stats.labels}
          iconUrl={annotationsUrl}
        />
        <ProjectStat
          title={"Classes"}
          value={project.stats.classes}
          iconUrl={classesUrl}
        />
      </div>
    </div>
    {showDownloadedBadge && project.path ? (
      <div className="badge absolute top-1">Downloaded</div>
    ) : null}
  </ButtonCard>
);
const ProjectStat = ({
  title,
  value,
  iconUrl,
}: {
  title: string;
  value: number;
  iconUrl: string;
}) => (
  <div className="flex flex-row gap-1 text-xs">
    <img src={iconUrl} />
    <span className="font-normal text-neutral-400">{title}</span>
    <span className="font-medium">{value}</span>
  </div>
);

const ProjectNotFoundCard = () => (
  <ButtonCard>
    <figure className="py-7">
      <img src={emptyUrl} alt="project-not-found" className="rounded" />
    </figure>
    <div className="card-body p-0">
      <h2 className="card-title text-sm font-semibold">No projects found</h2>
      <p className="text-xs font-normal text-neutral-400">
        Import a project or select a sandbox project
      </p>
    </div>
  </ButtonCard>
);

type ButtonCardProps = Pick<
  JSX.IntrinsicElements["button"],
  "children" | "onClick"
>;
const ButtonCard = ({ children, onClick }: ButtonCardProps) => (
  <button
    className="card card-bordered btn btn-ghost normal-case text-start w-72 h-[17rem] bg-base-100 shadow-sm border-1 border-zinc-50 rounded-xl py-3 gap-2"
    onClick={onClick}
    disabled={!onClick}
  >
    {children}
  </button>
);
