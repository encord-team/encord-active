import { useParams } from "react-router";

export function useProjectHash() {
  const { projectHash } = useParams();

  if (!projectHash) {
    throw Error("`useProjectHash` was used outside of a project route");
  }

  return projectHash;
}
