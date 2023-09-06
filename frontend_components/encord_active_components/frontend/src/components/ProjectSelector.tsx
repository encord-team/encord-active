import * as React from "react";
import { fork } from "radash";
import { Button, Select, Space } from "antd";
import { QueryContext } from "../hooks/Context";
import { useProjectList } from "../hooks/queries/useListProjects";
import { useMemo } from "react";

export type Props = {
  queryContext: QueryContext;
  selectedProjectHash: string;
  setSelectedProjectHash: (projectHash: string | undefined) => void;
};

export function ProjectSelector({
  queryContext,
  selectedProjectHash,
  setSelectedProjectHash,
}: Props) {
  const { data: projectListData } = useProjectList(queryContext);
  const projects = projectListData?.projects ?? [];
  const [sandboxProjects, userProjects] = useMemo(
    () => fork(projects, ({ sandbox }) => !!sandbox),
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
