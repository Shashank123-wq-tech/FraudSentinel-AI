import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP = (
    PROJECT_ROOT
    / "dashboard"
    / "app.py"
)


def main():

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.headless",
        "true",
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()