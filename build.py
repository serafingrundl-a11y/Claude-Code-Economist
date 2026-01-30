"""
Build script for the static DACA Results website.
Run: python build.py
This will:
  1. Copy all relevant files (PDFs, run logs, Python code) into data/
  2. Generate manifest.json
"""

import json
import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
TASKS = ["DACA Results Task 1", "DACA Results Task 2", "DACA Results Task 3"]


def build():
    # Clean previous data
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir()

    manifest = {}
    total_files = 0

    for task_idx, task_name in enumerate(TASKS, 1):
        task_dir = BASE_DIR / task_name
        if not task_dir.is_dir():
            print(f"WARNING: {task_dir} not found, skipping.")
            continue

        task_key = f"task{task_idx}"
        replications = {}
        out_task_dir = DATA_DIR / task_name
        out_task_dir.mkdir()

        rep_dirs = sorted(
            [d for d in task_dir.iterdir() if d.is_dir() and re.fullmatch(r"replication_(\d+)", d.name)],
            key=lambda d: int(re.search(r"(\d+)", d.name).group(1)),
        )

        for rep_dir in rep_dirs:
            match = re.fullmatch(r"replication_(\d+)", rep_dir.name)
            rep_num = match.group(1)
            out_rep_dir = out_task_dir / rep_dir.name
            out_rep_dir.mkdir()

            files_info = {"folder": rep_dir.name, "num": rep_num}

            # Copy PDF report
            pdfs = list(rep_dir.glob("replication_report_*.pdf"))
            if pdfs:
                shutil.copy2(pdfs[0], out_rep_dir / pdfs[0].name)
                files_info["pdf"] = pdfs[0].name
                total_files += 1
            else:
                files_info["pdf"] = None

            # Copy run log
            logs = list(rep_dir.glob("run_log_*.md"))
            if logs:
                shutil.copy2(logs[0], out_rep_dir / logs[0].name)
                files_info["log"] = logs[0].name
                total_files += 1
            else:
                files_info["log"] = None

            # Copy Python files
            py_files = sorted(rep_dir.glob("*.py"))
            code_names = []
            for py in py_files:
                shutil.copy2(py, out_rep_dir / py.name)
                code_names.append(py.name)
                total_files += 1
            files_info["code"] = code_names

            replications[rep_dir.name] = files_info

        manifest[task_key] = {
            "name": task_name,
            "replications": replications,
        }
        print(f"  {task_name}: {len(replications)} replications")

    # Write manifest
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nDone! Copied {total_files} files.")
    print(f"Manifest written to {manifest_path}")

    # Report size
    total_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    print(f"Total data size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
