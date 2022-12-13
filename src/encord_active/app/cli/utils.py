from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

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
