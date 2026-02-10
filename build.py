"""
Build script for the static DACA Results website.
Run: python build.py
This will:
  1. Copy all relevant files (PDFs, run logs, Python code) into data/
  2. Generate manifest.json with bug/recovered metadata
  3. Copy paper PDF to data/paper.pdf
"""

import csv
import json
import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
TASKS = ["DACA Results Task 1", "DACA Results Task 2", "DACA Results Task 3"]

# Bug metadata: replications with coding errors that affect estimates
# Keys are (task_key, rep_num_str)
BUGS = {
    ("task1", "07"): {
        "type": "int8_overflow",
        "file": "analysis.py",
        "lines": [35, 188],
        "desc": "AGE stored as int8; age_sq overflows for ages >= 12",
    },
    ("task1", "10"): {
        "type": "int8_overflow",
        "file": "analysis.py",
        "lines": [56, 169],
        "desc": "AGE stored as int8; age_sq overflows for ages >= 12",
    },
    ("task1", "24"): {
        "type": "int8_overflow",
        "file": "analysis.py",
        "lines": [34, 204],
        "desc": "AGE stored as int8; age_sq overflows before float cast",
    },
    ("task1", "45"): {
        "type": "int8_overflow",
        "file": "analysis.py",
        "lines": [37, 235],
        "desc": "AGE stored as int8; age_sq overflows for ages >= 12",
    },
    ("task1", "76"): {
        "type": "int8_overflow",
        "file": "analysis_script.py",
        "lines": [43, 199],
        "desc": "AGE stored as int8; I(AGE**2) overflows in formula",
    },
    ("task1", "97"): {
        "type": "int8_overflow",
        "file": "analysis_97.py",
        "lines": [46, 182],
        "desc": "AGE stored as int8; age_sq overflows for ages >= 12",
    },
    ("task2", "40"): {
        "type": "int8_overflow",
        "file": "analysis.py",
        "lines": [31, 245, 259, 274, 288],
        "desc": "AGE stored as int8; I(age**2) overflows in formula",
    },
    ("task2", "90"): {
        "type": "int8_overflow",
        "file": "analysis.py",
        "lines": [48, 188],
        "desc": "AGE stored as int8; age_sq overflows for ages >= 12",
    },
    ("task3", "16"): {
        "type": "educ_recode",
        "file": "analysis_code.py",
        "lines": [127, 128, 129, 130],
        "desc": "EDUC_RECODE compared to int but column is string; education dummies all zero",
    },
    ("task3", "52"): {
        "type": "educ_recode",
        "file": "analysis.py",
        "lines": [214, 215, 216, 217],
        "desc": "EDUC_RECODE compared to int but column is string; education dummies all zero",
    },
}

# Replications with recovered code (originally had no saved code file)
RECOVERED = {
    ("task1", "13"), ("task1", "63"),
    ("task3", "02"), ("task3", "05"), ("task3", "08"), ("task3", "14"),
    ("task3", "20"), ("task3", "21"), ("task3", "24"), ("task3", "26"),
    ("task3", "37"), ("task3", "45"), ("task3", "46"), ("task3", "54"),
    ("task3", "58"), ("task3", "65"), ("task3", "75"), ("task3", "86"),
    ("task3", "99"),
}


def load_estimates():
    """Load point estimates from daca_extraction.csv. Returns {(task_key, rep_num_str): float}."""
    estimates = {}
    extraction_path = BASE_DIR / "daca_extraction.csv"
    if not extraction_path.exists():
        print(f"WARNING: {extraction_path} not found, estimates will be null.")
        return estimates
    with open(extraction_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["task"] == "meta":
                continue
            task_key = f"task{row['task']}"
            rep_num = row["rep"].zfill(2)
            try:
                estimates[(task_key, rep_num)] = round(float(row["est"]), 4)
            except (ValueError, KeyError):
                pass
    return estimates


def build():
    # Load point estimates
    estimates = load_estimates()

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

            # Add bug metadata if this rep has a known bug
            bug_key = (task_key, rep_num.zfill(2))
            if bug_key in BUGS:
                files_info["bug"] = BUGS[bug_key]
            else:
                files_info["bug"] = None

            # Mark as recovered if code was recovered from run logs
            files_info["recovered"] = bug_key in RECOVERED

            # Add point estimate
            files_info["est"] = estimates.get(bug_key)

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

    # Copy paper PDF
    paper_src = BASE_DIR / "claude_code_human_comparison_v2.pdf"
    if paper_src.exists():
        shutil.copy2(paper_src, DATA_DIR / "paper.pdf")
        print(f"  Paper PDF copied to {DATA_DIR / 'paper.pdf'}")
        total_files += 1
    else:
        print(f"WARNING: Paper PDF not found at {paper_src}")

    print(f"\nDone! Copied {total_files} files.")
    print(f"Manifest written to {manifest_path}")

    # Report size
    total_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())
    print(f"Total data size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
