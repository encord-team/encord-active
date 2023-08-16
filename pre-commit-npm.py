import subprocess
import sys
from pathlib import Path
from typing import List


def call_npm():
    args: List[str] = sys.argv[1:]
    command: List[str] = ["npm"] + args
    self_path = Path(__file__)
    fe_dir = self_path.parent / "frontend_components" / "encord_active_components" / "frontend"
    subprocess.run(" ".join(command), shell=True, check=True, cwd=fe_dir)


if __name__ == "__main__":
    call_npm()
