import subprocess
import sys


def main():

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "copilot/app.py",
        "--server.headless=true",
    ]

    raise SystemExit(
        subprocess.call(command)
    )


if __name__ == "__main__":
    main()