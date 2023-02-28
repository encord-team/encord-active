from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

import rich
import typer

R = TypeVar("R")


def bypass_streamlit_question(fn: Callable[..., R]) -> Callable[..., R]:
    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        st_conf = Path.home() / ".streamlit" / "credentials.toml"
        if not st_conf.exists():
            st_conf.parent.mkdir(exist_ok=True)
            st_conf.write_text('[general]\nemail = ""')
        fn(*args, **kwargs)

    return inner


def try_find_parent_project(target: Path) -> Optional[Path]:
    for parent in target.parents:
        if (parent / "project_meta.yaml").is_file():
            return parent
    return None


def find_child_projects(target: Path) -> List[Path]:
    child_meta_files = target.glob("*/project_meta.yaml")
    return [cmf.parent.relative_to(target) for cmf in child_meta_files]


def is_project(target: Path):
    return (target / "project_meta.yaml").is_file()


def choose_project(child_projects):
    from InquirerPy import inquirer as i
    from InquirerPy.base.control import Choice
    from rich.panel import Panel

    rich.print(
        Panel(
            """
The specified directory is not an Encord Project.
Encord Active found the following projects to choose from.
            """,
            title="🕵️  Select a project 🕵️",
            expand=False,
            style="blue",
        )
    )
    choices = list(map(lambda p: Choice(p, name=p.name), child_projects))
    return i.fuzzy(
        message="Choose a project",
        choices=choices,
        vi_mode=True,
        instruction="Type a search string or use arrow keys and hit [enter]. Hit [ctrl-c] to abort.",
    ).execute()


def ensure_project(*, allow_multi=False):
    def decorator(fn: Callable[..., R]) -> Callable[..., R]:
        @wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            target: Optional[Path] = kwargs.get("target", None)
            if target is None:
                raise ValueError("Missing argument: `target`")

            if is_project(target):
                return fn(*args, **kwargs)

            from rich.panel import Panel

            parent_target = try_find_parent_project(target)
            if parent_target is not None:
                rich.print(
                    Panel(
                        f"""
    The directory is a subdirectory of a project.
    Executing command on parent directory
    [purple]{parent_target}[/purple]
                        """,
                        title=":file_folder: Found parent project :file_folder:",
                        style="blue",
                        expand=False,
                    )
                )
                kwargs["target"] = parent_target
                return fn(*args, **kwargs)

            child_projects = find_child_projects(target)

            if not child_projects:
                rich.print(
                    Panel(
                        f"""
        Couldn't find a project at: [purple]{kwargs['target']}[/purple].
        Either [blue]`cd`[/blue] into a directory containing a project or specify the path with the `--target` option.

        :bulb: hint: By default, projects are stored in the current working directory when you run the [blue]init[/blue], [blue]import[/blue], and [blue]download[/blue] commands.
                    """,
                        title=":exclamation: Couldn't find a project :exclamation:",
                        style="yellow",
                        expand=False,
                    )
                )

                raise typer.Exit()

            if not allow_multi:
                chosen_target = choose_project(child_projects)
                if not chosen_target:
                    rich.print("No project was selected. Aborting.")
                    raise typer.Exit()

                kwargs["target"] = chosen_target

            return fn(*args, **kwargs)

        return inner

    return decorator
