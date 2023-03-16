# Masks

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
│   ├── 00000.png
│   ├── ...
│   └── 00030.png
└── ...
```

Each of the `png` files contains masks were pixel values different from 0 correspond to the instance of that value.
For example, in the pixel map below, there are two instances; instance `1` in the top-left corner and instance `2` in the bottom-right.

```
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 0 0 0 0 0 0 0 0 0 2 2 0 0
0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 0
0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0
0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

Such masks are parsed into individual labels defined by polygon coordinates.

#### Meta data

The `meta.json` file contains an entry for each sequence (sub-directory above).
That entry has a dictionary of objects, where keys are the instance ids.
For the example above, there would be two instances, one with key `1` and one with key `2`.
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

The [polygon_transformer.py](./polygon_transformer.py) is the class responsible for transforming the labels.
It will need to know the `png` files that are supposed to be parsed.

We specify those paths by the `--label-glob` argument to the `init` command as well as the path to the `meta.json` file for class lookups:

```
encord-active init \
    --data-glob "JPEGImages/**/*.jpg" \
    --label-glob "Annotations/**/*.png" \
    --label-glob "meta.json" \
    --transformer polygon_transformer.py \
    data
```
