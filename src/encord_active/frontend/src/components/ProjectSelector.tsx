import { fork } from "radash";
import { Button, Select, Space } from "antd";
import { useMemo } from "react";
import { useProjectList } from "../hooks/queries/useListProjects";

export type Props = {
  selectedProjectHash: string;
  setSelectedProjectHash: (projectHash: string | undefined) => void;
};

export function ProjectSelector({
  selectedProjectHash,
  setSelectedProjectHash,
}: Props) {
  const { data: projectListData } = useProjectList();
  const projects = projectListData?.projects;
  const [sandboxProjects, userProjects] = useMemo(
    () => fork(projects ?? [], ({ sandbox }) => !!sandbox),
    [projects]
  );

  return (
    <Space.Compact block size="large">
      <Button onClick={() => setSelectedProjectHash(undefined)}>
        View all projects
      </Button>
      <Select
        value={selectedProjectHash}
        onChange={(projectHash) => setSelectedProjectHash(projectHash)}
        className="min-w-[192px] max-w-lg"
        options={[
          {
            label: "User Projects",
            options: userProjects.map((project) => ({
              label: project.title,
              value: project.project_hash,
            })),
          },
          {
            label: "Sandbox Projects",
            options: sandboxProjects.map((project) => ({
              label: project.title,
              value: project.project_hash,
            })),
          },
        ]}
      />
    </Space.Compact>
  );
}
