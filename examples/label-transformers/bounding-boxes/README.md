# Bounding Boxes

For this example, the data structure is as follows:

```
├── Annotations
├── JPEGImages
└── meta.json
```

#### Data

The `JPEGImages` directory contains sub-directories with images for each of the sequences.

```
JPEGImages
├── video1
│   ├── 00000.jpg
│   ├── ...
│   └── 00030.jpg
└── ...
```

#### Annotations

The `Annotations` directory contains sub-directories with annotations for individual sequences of images.

```
Annotations
├── video1
│   ├── 00000.json
│   ├── ...
│   └── 00030.json
└── ...
```

Each of the `json` files are structured with a key per instance in the frame.

```json
{
  "1": { "x": 0.4275, "y": 0.4321, "w": 0.1989, "h": 0.1593 }
}
```

#### Meta data

The `meta.json` file contains an entry for each sequence (sub-directory above).
That entry has a dictionary of objects, where keys are the object ids.
Each object has a class, which will be the class given to the imported bounding boxes.

```
{
"videos": {
    "video1": {
        "objects": {
            "1": {
            "category": "shark",
            "frames": [
                "00000",
                "00005",
                "00010",
                "00015",
                "00020",
                "00025",
                "00030"
            ]
        }
    }
},

```

## Initializing an Encord Active Project

The [bbox_transformer.py](./bbox_transformer.py) is the class responsible for transforming the labels.
It will need to know the `json` files that are supposed to be parsed.

We specify those paths by the `--label-glob` argument to the `init` command as well as the path to the `meta.json` file for class lookups:

```
encord-active init \
    --data-glob "JPEGImages/**/*.jpg" \
    --label-glob "Annotations/**/*.json" \
    --label-glob "meta.json" \
    --transformer bbox_transformer.py \
    data
```
