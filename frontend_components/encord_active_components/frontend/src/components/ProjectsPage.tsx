import * as React from "react";
import { fork } from "radash";
import { useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import axios from "axios";
import { Card, Space, Spin } from "antd";
import Meta from "antd/es/card/Meta";
import emptyUrl from "../../assets/empty.svg";
import importUrl from "../../assets/import.svg";
import encordImportUrl from "../../assets/encord-import.svg";
import fileImageUrl from "../../assets/file-image.svg";
import annotationsUrl from "../../assets/annotations.svg";
import classesUrl from "../../assets/classes.svg";
import DEFAULT_PROJECT_IMAGE from "../../assets/default_project_image.webp";

import { classy } from "../helpers/classy";
import { IntegratedProjectMetadata } from "./IntegratedAPI";
import { apiUrl, env } from "../constants";
import { useImageSrc } from "../hooks/useImageSrc";
import { loadingIndicator } from "./Spin";
import { QueryAPI } from "./Types";

const useDownloadProject = (options?: Parameters<typeof useMutation>[2]) =>
  useMutation(
    ["useDownloadProject"],
    async (projectHash: string) =>
      axios.get(`${apiUrl}/projects/${projectHash}/download_sandbox`),
    options
  );

export type Props = {
  readonly projects: IntegratedProjectMetadata[];
  queryAPI: QueryAPI;
  onSelectLocalProject: (projectHash: string) => void;
};
export function ProjectsPage({
  projects = [],
  queryAPI,
  onSelectLocalProject,
}: Props) {
  const [sandboxProjects, userProjects] = fork(
    projects,
    ({ sandbox }) => !!sandbox
  );
  const { mutate: download, isLoading } = useDownloadProject();

  return (
    <div className="flex h-max flex-col gap-5 p-5">
      {env !== "production" && (
        <>
          <h1 className="text-3xl font-medium">Let’s get started!</h1>
          <div className="flex flex-row flex-wrap gap-5">
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
      <h2 className="text-xl font-light text-neutral-700">Your projects</h2>
      <Space>
        {userProjects.length ? (
          userProjects.map((project) => (
            <ProjectCard
              key={project.project_hash}
              queryAPI={queryAPI}
              project={project}
              setSelectedProjectHash={onSelectLocalProject}
            />
          ))
        ) : (
          <ProjectNotFoundCard />
        )}
      </Space>
      {/* TODO: temporarily hide sandbox projects  */}
      {false && env !== "production" && sandboxProjects.length && (
        <>
          <h2 className="text-xl font-light text-neutral-700">
            View a sandbox project
          </h2>
          <div className="flex flex-wrap gap-5">
            {sandboxProjects
              .sort((a, b) => -a.downloaded - -b.downloaded)
              .map((project) => (
                <ProjectCard
                  key={project.project_hash}
                  project={project}
                  showDownloadedBadge
                  setSelectedProjectHash={onSelectLocalProject}
                  /* onClick={() =>
                    (project.downloaded ? onSelectLocalProject : download)(
                      project["projectHash"],
                    )
                  } */
                />
              ))}
          </div>
        </>
      )}
    </div>
  );
}

function NewProjectButton({
  title,
  description,
  iconUrl,
  onClick,
}: {
  title: string;
  description: string;
  iconUrl: string;
  onClick?: JSX.IntrinsicElements["button"]["onClick"];
}) {
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
          "felx-row border-1 btn btn-ghost flex h-28 w-96 justify-start gap-3 border-zinc-50 p-3.5 normal-case",
          { "shadow-lg": !disabled }
        )}
        onClick={onClick}
        disabled={disabled}
      >
        <div className="flex h-20	w-20 items-center justify-center rounded-md bg-zinc-50">
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
            <span className="text-xs font-normal text-gray-400">
              {description}
            </span>
          )}
        </div>
      </button>
    </div>
  );
}

function ProjectCard({
  project,
  queryAPI,
  showDownloadedBadge = false,
  setSelectedProjectHash,
}: {
  project: IntegratedProjectMetadata;
  queryAPI: QueryAPI;
  showDownloadedBadge?: boolean;
  setSelectedProjectHash: (projectHash: string) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const { data: projectSummary } = queryAPI.useProjectSummary(
    project.project_hash
  );
  const { data: previewItem } = queryAPI.useProjectItemPreview();
  const imgSrcUrl = useImageSrc(previewItem?.url);

  if (imgSrcUrl === undefined || previewItem === undefined) {
    return <Spin indicator={loadingIndicator} />;
  }

  return (
    <Card
      hoverable
      style={{ width: 240 }}
      onClick={() => setSelectedProjectHash(project.project_hash)}
      cover={
        <figure
          className="max-h-36 rounded"
          style={{ width: 240, height: 165, objectFit: "cover" }}
        >
          {previewItem.timestamp != null && imgSrcUrl ? (
            <video
              src={imgSrcUrl}
              muted
              controls={false}
              onLoadedMetadata={() => {
                const videoRef = video.current;
                if (videoRef != null) {
                  videoRef.currentTime = project.imageUrlTimestamp || 0;
                }
              }}
              style={{ width: 240, height: 165, objectFit: "cover" }}
            />
          ) : (
            <img
              src={imgSrcUrl ?? DEFAULT_PROJECT_IMAGE}
              alt={project.title}
              style={{ width: 240, height: 165, objectFit: "cover" }}
            />
          )}
        </figure>
      }
    >
      <Meta title={project.title} description={project.description} />
      <div className="card-body w-full justify-between gap-3 p-0">
        <div className="flex flex-col">
          <ProjectStat
            title="Dataset"
            value={project?.stats?.dataUnits}
            iconUrl={fileImageUrl}
          />
          <ProjectStat
            title="Annotations"
            value={project?.stats?.labels}
            iconUrl={annotationsUrl}
          />
          <ProjectStat
            title="Classes"
            value={project?.stats?.classes}
            iconUrl={classesUrl}
          />
        </div>
      </div>
      {showDownloadedBadge && project?.downloaded ? (
        <div className="badge absolute top-1">Downloaded</div>
      ) : null}
    </Card>
  );
}
function ProjectStat({
  title,
  value,
  iconUrl,
}: {
  title: string;
  value: number;
  iconUrl: string;
}) {
  return (
    <div className="flex flex-row gap-1 text-xs">
      <img src={iconUrl} />
      <span className="font-normal text-neutral-400">{title}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

function ProjectNotFoundCard() {
  return (
    <Card>
      <figure className="py-7">
        <img src={emptyUrl} alt="project-not-found" className="rounded" />
      </figure>
      <div className="card-body p-0">
        <h2 className="card-title text-sm font-semibold">No projects found</h2>
        <p className="text-xs font-normal text-neutral-400">
          Import a project or select a sandbox project
        </p>
      </div>
    </Card>
  );
}
