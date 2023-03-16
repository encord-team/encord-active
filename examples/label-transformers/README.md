# Label Transformers

This directory contains examples of how to transform custom labels into Encord Active labels.

The examples are structured such that every sub-directory contains a `data` directory with a small set of images and labels structured in various ways.
Along with a concrete implementation of a `LabelTransformer`, each directory will have a README with a description of the data and the command to run to initialize a project with the data and the labels.

### Examples:

1. [classification](./classification) shows how to infer classification labels based on the file structure.
2. [bounding-boxes](./bounding-boxes) shows how to infer bounding box labels based on json files.
3. [polygons](./polygons) shows how to infer polygon labels based on masks stored in png files.
4. [multiple-transformers](./multiple-transformers) shows how you can use multiple transformers in the same initialization.
