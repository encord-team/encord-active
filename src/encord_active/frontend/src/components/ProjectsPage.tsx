import { Card, Space, Spin, Tooltip } from "antd";
import Meta from "antd/es/card/Meta";
import emptyUrl from "../../assets/empty.svg";
import fileImageUrl from "../../assets/file-image.svg";
import annotationsUrl from "../../assets/annotations.svg";
import classesUrl from "../../assets/classes.svg";
import { useImageSrc } from "../hooks/useImageSrc";
import { loadingIndicator } from "./Spin";
import { useProjectSummary } from "../hooks/queries/useProjectSummary";
import { useProjectList } from "../hooks/queries/useListProjects";
import { ProjectSearchEntry } from "../openapi/api";
import { useProjectMetadata } from "../hooks/queries/useProjectMetadata";
import { useProjectItem } from "../hooks/queries/useProjectItem";
import { AnnotatedImage } from "./preview/AnnotatedImage";

export type Props = {
  onSelectLocalProject: (projectHash: string) => void;
};
export function ProjectsPage({ onSelectLocalProject }: Props) {
  const { data: projects, isLoading } = useProjectList();
  const userProjects = projects?.projects ?? [];

  if (isLoading) {
    return <Spin indicator={loadingIndicator} />;
  }

  return (
    <div className="flex h-max w-full flex-col gap-5 p-5">
      <h2 className="text-xl font-light text-neutral-700">Your projects</h2>
      <Space wrap>
        {userProjects.length ? (
          userProjects.map((project) => (
            <ProjectCard
              key={project.project_hash}
              project={project}
              setSelectedProjectHash={onSelectLocalProject}
            />
          ))
        ) : (
          <ProjectNotFoundCard />
        )}
      </Space>
    </div>
  );
}

function ProjectCard({
  project,
  showDownloadedBadge = false,
  setSelectedProjectHash,
}: {
  project: ProjectSearchEntry;
  showDownloadedBadge?: boolean;
  setSelectedProjectHash: (projectHash: string) => void;
}) {
  const { data: projectMetadata } = useProjectMetadata(project.project_hash);
  const { data: projectSummary } = useProjectSummary(project.project_hash);
  const { data: projectDataItem } = useProjectItem(
    project.project_hash,
    projectSummary?.preview ?? "",
    { enabled: projectSummary !== undefined }
  );
  const imgSrcUrl = useImageSrc(projectDataItem?.url);

  return (
    <Card
      hoverable
      className="w-60"
      loading={imgSrcUrl === undefined || projectDataItem === undefined}
      onClick={() => setSelectedProjectHash(project.project_hash)}
      cover={
        projectDataItem !== undefined ? (
          <AnnotatedImage
            item={projectDataItem}
            annotationHash={undefined}
            mode="preview"
            predictionTruePositive={undefined}
          />
        ) : null
      }
    >
      <Meta
        title={<Tooltip overlay={project.title}>{project.title}</Tooltip>}
        description={
          <Tooltip
            overlay={project.description}
            className="text-ellipsis whitespace-nowrap text-xs"
          >
            {project.description}
          </Tooltip>
        }
      />
      <div className="card-body w-full justify-between gap-3 p-0">
        <div className="flex flex-col">
          <ProjectStat
            title="Dataset"
            value={projectMetadata?.data_count ?? 0}
            iconUrl={fileImageUrl}
          />
          <ProjectStat
            title="Annotations"
            value={projectMetadata?.annotation_count ?? 0}
            iconUrl={annotationsUrl}
          />
          <ProjectStat
            title="Classes"
            value={projectMetadata?.class_count ?? 0}
            iconUrl={classesUrl}
          />
        </div>
      </div>
      {showDownloadedBadge && project?.sandbox ? (
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
      <img src={iconUrl} alt={title} />
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
