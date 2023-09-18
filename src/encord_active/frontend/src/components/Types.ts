type OntologyObjectAttributeBase = {
  readonly id: number | string;
  readonly featureNodeHash: string;
  readonly name: string;
};

export type OntologyObjectAttribute =
  | (OntologyObjectAttributeBase & {
      type: "checklist" | "radio";
      options: OntologyObjectAttributeOptions[];
    })
  | (OntologyObjectAttributeBase & {
      type: "text";
    });

export type OntologyObjectAttributeOptions = {
  readonly id: number | string;
  readonly featureNodeHash: string;
  readonly label: string;
  readonly value: string;
  readonly options?: (
    | OntologyObjectAttribute
    | OntologyObjectAttributeOptions
  )[];
};

export type ProjectOntology = {
  readonly objects: readonly {
    readonly id: string;
    readonly name: string;
    readonly color: string;
    readonly shape: string;
    readonly featureNodeHash: string;
    readonly attributes?: readonly OntologyObjectAttribute[];
  }[];
  readonly classifications: readonly {
    readonly id: string;
    readonly name: string;
    readonly color: string;
    readonly shape: string;
    readonly featureNodeHash: string;
    readonly attributes?: readonly OntologyObjectAttribute[];
  }[];
};

export type FeatureHashMap = Readonly<
  Record<string, { readonly color: string; readonly name: string }>
>;
