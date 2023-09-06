import * as React from "react";
import { fork } from "radash";
import { Button, Select, Space } from "antd";
import { IntegratedProjectMetadata } from "./IntegratedAPI";

export type Props = {
  projects: readonly IntegratedProjectMetadata[];
  selectedProjectHash: string;
  setSelectedProjectHash: (projectHash: string | undefined) => void;
};

export function ProjectSelector({
  projects,
  selectedProjectHash,
  setSelectedProjectHash,
}: Props) {
  const [sandboxProjects, userProjects] = fork(
    projects.filter(({ sandbox }) => sandbox),
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
