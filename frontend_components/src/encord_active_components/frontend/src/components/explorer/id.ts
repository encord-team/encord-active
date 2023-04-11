export type IdParts = {
  labelRow: string;
  data: string;
  frame: number;
  objectHash: string | null;
};

export const splitId = (id: string): IdParts => {
  const [lr, du, frame, obj] = id.split("_");

  if (!lr || !du || !frame) throw `invalid id: ${id}`;

  return {
    labelRow: lr,
    data: du,
    frame: parseInt(frame),
    objectHash: obj ?? null,
  };
};
