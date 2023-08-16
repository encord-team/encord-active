export type IdParts = {
  du_hash: string;
  frame: number;
  annotation_hash: string | null;
};

export const splitId = (id: string): IdParts => {
  const [du, frame, obj] = id.split("_");

  if (!du || !frame) {
    throw Error(`invalid id: ${id}`);
  }

  return {
    du_hash: du,
    frame: parseInt(frame),
    annotation_hash: obj ?? null,
  };
};

export const takeDataId = (id: string): string =>
  id.split("_").slice(0, 2).join("_");
