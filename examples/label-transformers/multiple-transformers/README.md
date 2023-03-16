# Multiple Transformers at Once

You can use multiple transformers for the same `init` command.

This dataset is a merge of that in [bounding-boxes](../bounding-boxes) and [polygons](../polygons).
The data structure is the same and the transformers are the same.
The only thing we did was refactor utilities into a file and adding another file [both.py](./both.py) which imports both transformers.

If you now run

```
encord-active init \
    --data-glob "JPEGImages/**/*.jpg" \
    --label-glob "Annotations/**/*.json" \
    --label-glob "Annotations/**/*.png" \
    --label-glob "meta.json" \
    --transformer both.py \
    data
```

you will be prompted with a choice of which transformer you want to use.
Use the arrows to choose the one you would like to use and hit `[ENTER]`.

This is useful if you want to keep your label transformers in a common place and reuse them across projects.
