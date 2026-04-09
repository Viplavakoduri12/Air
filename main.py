import os
import runpy
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent
local_deps_dir = project_dir / ".deps"

if local_deps_dir.exists():
    sys.path.insert(0, str(local_deps_dir))

from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web import cli as stcli


def resource_path(filename: str) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_dir / filename


def is_streamlit_session() -> bool:
    return get_script_run_ctx() is not None


def main() -> int:
    app_path = resource_path("app.py")

    if not app_path.exists():
        raise FileNotFoundError(f"Could not find Streamlit app at {app_path}")

    if is_streamlit_session():
        runpy.run_path(str(app_path), run_name="__main__")
        return 0

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.fileWatcherType=none",
        *sys.argv[1:],
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
