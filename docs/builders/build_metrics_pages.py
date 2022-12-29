"""
Script for building markdown pages with information from the metrics.
When run this will build pages directly into the `docs/docs/metrics directory.
"""
from importlib import import_module
from pathlib import Path
from typing import Any, List, Union, cast

from tabulate import tabulate

import encord_active.lib.metrics.metric as metrics
import encord_active.lib.metrics.execute as run_all

github_url = "https://github.com/encord-team/encord-active"
descriptions = {
    "heuristic": "Work on images or individual video frames and are heuristic in the sense that they mostly depend on the image content without labels.",
    "geometric": "Operate on the geometries of objects like bounding boxes, polygons, and polylines.",
    "semantic": "Operates with the semantic information of images or individual video frames.",
}

metric_path = Path(run_all.__file__).parent
docs_root = metric_path.parents[1]

submodules = [p.name for p in metric_path.iterdir() if p.name[0] != "_" and p.is_dir()]

ms = cast(  # rename to avoid conflict with metrics module
    List[List[metrics.Metric]],
    [
        sorted(
            [import_module(m1).__getattribute__(m2)() for m1, m2 in run_all.get_metrics(submodule)],
            key=lambda m: m.TITLE,
        )
        for submodule in submodules
    ],
)


def fix_strings(row: List[Any]) -> List[str]:
    for i, val in enumerate(row):
        val = row[i]
        if not isinstance(val, list):
            val = [val]
        val2: List[Union[str, int, float]] = []
        for v in val:
            if hasattr(v, "value"):
                v = getattr(v, "value")
            elif hasattr(v, "__name__"):
                v = "Any"
            elif v is None:
                v = "Any"

            if i in {2, 3,} and isinstance(
                v, str
            ):  # image and annotation type, so replace "_" with space and let every type be a tag
                v = v.replace("_", " ")
                v = f"`{v}`"
            val2.append(v)
        try:
            row[i] = ", ".join(map(str, sorted(set(val2))))
        except TypeError:
            print(row)
            print(val2)

    row[0] = f'[{row[0]}]({row[0].lower().replace(" ", "-")}) - <small>{row[1]}</small>'
    row.pop(1)
    return row


for submodule, metrics_list in zip(submodules, ms):
    rows = []
    for m in metrics_list:
        rows.append(fix_strings([m.TITLE, m.SHORT_DESCRIPTION, m.DATA_TYPE, m.ANNOTATION_TYPE]))
    markdown_table = tabulate(
        rows,
        headers=["Title", "Metric Type", "Data Type", "Annotation Type"],
        tablefmt="github",
    )

    md = ""
    for metric in metrics_list:
        url = f"{github_url}/blob/main/{type(metric).__module__.replace('.', '/')}.py"
        md += f"\n## {metric.TITLE}  \n" f"{metric.LONG_DESCRIPTION}  \n\n" f"Implementation on [GitHub]({url})\n"

    markdown_file = docs_root / "docs" / "metrics" / f"{submodule}.md"
    with markdown_file.open("w", encoding="utf-8") as f:

        def w(line):
            f.write(f"{line}")
            f.write("\n\n")

        w(f"# {submodule.title()}")
        w(descriptions[submodule])
        w(markdown_table)
        w(md)
