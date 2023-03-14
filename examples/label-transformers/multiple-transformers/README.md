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

you will be prompted with a choice of which transformers you want to use.
Use `[TAB]` or `[SPACE]` to select/deselect metrics before hitting `[ENTER]` to execute the initialization.

This is useful if you want to keep your label transformers in a common place and reuse them across projects.
