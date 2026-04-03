#!/usr/bin/env python3
"""Profile the current git branch and inspect results in a browser.

Flame graph (pyinstrument) — shows where wall-clock time is spent:
    python benchmark/profile_branch.py                   # profile → _latest.html
    python benchmark/profile_branch.py --keep            # profile + save permanently

Call-count profile (cProfile + snakeviz) — shows hottest methods by call count:
    python benchmark/profile_branch.py --calls           # profile → open snakeviz
    python benchmark/profile_branch.py --calls --keep    # profile + save .prof file

Managing saved results:
    python benchmark/profile_branch.py --list            # list all saved results
"""

import argparse
import cProfile
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Python code to profile (executed as a string in the current process)
PYTHON_SCRIPT = (
    "import Generate; from Main import main as ERmain;"
    " erargs, seed = Generate.main(); ERmain(erargs, seed)"
)

# Arguments forwarded to the profiled script via sys.argv
SCRIPT_ARGS = ["--seed", "1", "--skip_output"]

# =============================================================================
# END OF CONFIGURATION — do not modify below unless you know what you are doing
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "benchmark" / "profiles"
TEMP_FLAME = OUTPUT_DIR / "_latest.html"
TEMP_CALLS = OUTPUT_DIR / "_latest.prof"

_FLAME_RE = re.compile(r"^profile_(.+)_([0-9a-f]{7,})_(\d{8}T\d{6})\.html$")
_CALLS_RE = re.compile(r"^calls_(.+)_([0-9a-f]{7,})_(\d{8}T\d{6})\.prof$")


def _check_dependency(module: str, install: str) -> None:
    try:
        __import__(module)
    except ImportError:
        print(f"{module} is not installed. Run: pip install {install}", file=sys.stderr)
        sys.exit(1)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def get_git_info() -> tuple[str, str]:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit = _git("rev-parse", "--short", "HEAD")
    return branch, commit


def _print_header(branch: str, commit: str) -> None:
    print(f"[profile] Branch '{branch}' @ {commit}")
    print(f"[profile] Script : {PYTHON_SCRIPT}")
    print(f"[profile] Args   : {' '.join(SCRIPT_ARGS)}")
    print()


def _exec_script() -> None:
    """Execute PYTHON_SCRIPT with SCRIPT_ARGS set in sys.argv, from REPO_ROOT."""
    exec(compile(PYTHON_SCRIPT, "<profile_script>", "exec"), {"__name__": "__main__"})


def run_flame_profile() -> tuple[str, str]:
    """Run pyinstrument and write an HTML flame graph to TEMP_FLAME."""
    import pyinstrument

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    branch, commit = get_git_info()
    _print_header(branch, commit)

    original_argv, original_dir = sys.argv, Path.cwd()
    try:
        sys.argv = [PYTHON_SCRIPT] + SCRIPT_ARGS
        os.chdir(REPO_ROOT)
        profiler = pyinstrument.Profiler()
        profiler.start()
        _exec_script()
        profiler.stop()
    finally:
        sys.argv = original_argv
        os.chdir(original_dir)

    TEMP_FLAME.write_text(profiler.output_html(), encoding="utf-8")
    print(f"\n[profile] Flame graph saved → {TEMP_FLAME}")
    return branch, commit


def run_calls_profile() -> tuple[str, str]:
    """Run cProfile and write a .prof file to TEMP_CALLS, then open snakeviz."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    branch, commit = get_git_info()
    _print_header(branch, commit)

    original_argv, original_dir = sys.argv, Path.cwd()
    try:
        sys.argv = [PYTHON_SCRIPT] + SCRIPT_ARGS
        os.chdir(REPO_ROOT)
        cProfile.run(PYTHON_SCRIPT, str(TEMP_CALLS))
    finally:
        sys.argv = original_argv
        os.chdir(original_dir)

    print(f"\n[profile] Call profile saved → {TEMP_CALLS}")
    return branch, commit


def keep_flame(branch: str, commit: str) -> Path:
    date_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = OUTPUT_DIR / f"profile_{branch.replace('/', '_')}_{commit}_{date_str}.html"
    shutil.copy2(TEMP_FLAME, dest)
    print(f"[profile] Saved → {dest.name}")
    return dest


def keep_calls(branch: str, commit: str) -> Path:
    date_str = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = OUTPUT_DIR / f"calls_{branch.replace('/', '_')}_{commit}_{date_str}.prof"
    shutil.copy2(TEMP_CALLS, dest)
    print(f"[profile] Saved → {dest.name}")
    return dest


def list_profiles() -> None:
    """Print all saved flame graphs and call profiles sorted by creation time."""
    if not OUTPUT_DIR.exists():
        print("[profile] No saved profiles found.")
        return

    entries: list[tuple[float, str, str, str, str]] = []  # (mtime, type, branch, commit, date_str)
    for p in OUTPUT_DIR.iterdir():
        m = _FLAME_RE.match(p.name) or _CALLS_RE.match(p.name)
        if m:
            kind = "flame" if p.suffix == ".html" else "calls"
            branch, commit, date_str = m.groups()
            entries.append((p.stat().st_mtime, kind, branch.replace("_", "/"), commit, date_str))

    if not entries:
        print("[profile] No saved profiles found.")
        return

    entries.sort(key=lambda e: e[0])
    print(f"{'Index':<7} {'Type':<7} {'Branch':<40} {'Commit':<10} {'Date'}")
    print("-" * 82)
    for i, (_, kind, branch, commit, date_str) in enumerate(entries):
        date = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
        print(f"  [{i}]   {kind:<7} {branch:<40} {commit:<10} {date.strftime('%Y-%m-%d %H:%M:%S')}")


def open_snakeviz(path: Path) -> None:
    """Launch snakeviz for the given .prof file (blocks until Ctrl-C)."""
    print(f"[profile] Launching snakeviz for {path.name}  (Ctrl-C to stop)")
    subprocess.run([sys.executable, "-m", "snakeviz", str(path)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the current git branch and inspect results in a browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--calls",
        action="store_true",
        help=(
            "Run cProfile instead of pyinstrument and open the result in snakeviz. "
            "Shows exact call counts and time per method."
        ),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Save the result with a permanent branch/commit/date filename.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all saved profiling results (flame graphs and call profiles).",
    )

    args = parser.parse_args()

    if args.list:
        list_profiles()
        return

    if args.calls:
        _check_dependency("snakeviz", "snakeviz")
        branch, commit = run_calls_profile()
        if args.keep:
            keep_calls(branch, commit)
        open_snakeviz(TEMP_CALLS)
    else:
        _check_dependency("pyinstrument", "pyinstrument")
        branch, commit = run_flame_profile()
        if args.keep:
            keep_flame(branch, commit)


if __name__ == "__main__":
    main()
