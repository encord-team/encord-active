import os
import sys
from functools import wraps


def silence_stdout(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        stdout_orig = sys.stdout
        sys.stdout = open(os.devnull, "w")
        fn(*args, **kwargs)
        sys.stdout = stdout_orig

    return wrapper
