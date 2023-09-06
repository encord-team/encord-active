import { fork } from "radash";
import { IntegratedProjectMetadata } from "./IntegratedAPI";
import { Button, Select, Space } from "antd";

export type Props = {
  projects: readonly IntegratedProjectMetadata[];
  selectedProjectHash: string;
  setSelectedProjectHash: (projectHash: string | undefined) => void;
};

export const ProjectSelector = ({
  projects,
  selectedProjectHash,
  setSelectedProjectHash,
}: Props) => {
  const [sandboxProjects, userProjects] = fork(
    projects.filter(({ downloaded }) => downloaded),
    ({ sandbox }) => !!sandbox
  );

  return (
    <Space.Compact block size="large">
      <Button onClick={() => setSelectedProjectHash(undefined)}>
        View all projects
      </Button>
      <Select
        value={selectedProjectHash}
        onChange={(projectHash) => setSelectedProjectHash(projectHash)}
        style={{ maxWidth: 500 }}
        options={[
          {
            label: "User Projects",
            options: userProjects.map((project) => ({
              label: project.title,
              value: project.projectHash,
            })),
          },
          {
            label: "Sandbox Projects",
            options: sandboxProjects.map((project) => ({
              label: project.title,
              value: project.projectHash,
            })),
          },
        ]}
      />
    </Space.Compact>
  );
};
