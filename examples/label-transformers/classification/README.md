# Classification

This dataset is structured as follows:

```
data
├── cat
│   ├── 1.jpg
│   ├── ...
│   └── 7.jpg
├── dog
│   ├── 1.jpg
│   ├── ...
│   └── 9.jpg
└── horse
    ├── 1.jpg
    ├── ...
    └── 12.jpeg
```

As such, the labels can be inferred from the file structure.

To initialize an Encord Active project from the data, run

```shell
encord-active init --transformer classification_transformer.py data
```

The [classification_transformer.py](./classification_transformer.py) file is the one that transforms the data paths to classifications.

The outcome should be a new project with 28 images of which

- 7 are labelled as cats
- 9 are labelled as dogs
- 12 are labelled as horses

Try running

```shell
encord-active visualize
```

to see the results.
