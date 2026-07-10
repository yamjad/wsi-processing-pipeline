#!/usr/bin/env python3
"""
download_cptac_slides.py

Filters the CPTAC slide index CSV down to a specific list of sample_ids
and downloads the corresponding whole-slide images from the Imaging Data
Commons (IDC) by running each row's `idc_download_cmd`.

Requirements:
    - Python 3.7+
    - pandas          (pip install pandas)
    - idc-index-data / idc client CLI available on PATH as `idc`
      (pip install idc-index   -->  provides the `idc` command)

Usage:
    python download_cptac_slides.py

    Optional flags:
        --csv PATH        Path to the slide index CSV (default: cptac_slide_index.csv)
        --outdir PATH      Directory to download slides into (default: ./idc_downloads)
        --dry-run          Print the commands that would run, but don't execute them
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. The sample_ids you want to pull slides for.
# ---------------------------------------------------------------------------
SAMPLE_IDS = [
    # LUAD
    "C3L-02513",
    "C3L-02515",
    "C3L-02560",
    "C3L-02616",
    "C3L-03271",
    "C3L-03462",
    "C3L-03726",
    "C3L-03969",
    "C3L-03976",
    "C3L-03984",
    # Uterine (UCEC)
    "C3L-00898",
    "C3L-00943",
    "C3L-01277",
    "C3L-01631",
    "C3L-01864",
    "C3L-01967",
    "C3L-02119",
    "C3L-02121",
    "C3L-02125",
    "C3L-02347",
]


def find_matching_rows(csv_path: Path, sample_ids):
    """Return list of dict rows whose sample_id is in sample_ids."""
    wanted = set(sample_ids)
    matches = []
    seen_cmds = set()  # dedupe identical idc_download_cmd values

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "sample_id" not in reader.fieldnames or "idc_download_cmd" not in reader.fieldnames:
            sys.exit(
                "ERROR: CSV must contain 'sample_id' and 'idc_download_cmd' columns. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            sid = row["sample_id"].strip()
            if sid in wanted:
                cmd = row["idc_download_cmd"].strip()
                if cmd and cmd not in seen_cmds:
                    seen_cmds.add(cmd)
                    matches.append(row)
    return matches


def report_missing(matches, sample_ids):
    found_ids = {row["sample_id"] for row in matches}
    missing = [sid for sid in sample_ids if sid not in found_ids]
    if missing:
        print("WARNING: no rows found in the CSV for these sample_ids:")
        for sid in missing:
            print(f"    - {sid}")
        print()


def run_downloads(matches, outdir: Path, dry_run: bool):
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(matches)} unique slide series to download.\n")

    for i, row in enumerate(matches, start=1):
        sid = row["sample_id"]
        cancer = row.get("cancer", "")
        cmd_str = row["idc_download_cmd"]

        # idc_download_cmd looks like: "idc download <StudyInstanceUID>"
        # We append an output directory so files land in a predictable place.
        cmd = cmd_str.split() #+ ["--outdir", str(outdir)]

        print(f"[{i}/{len(matches)}] sample_id={sid} cancer={cancer}")
        print(f"    running: {' '.join(cmd)}")

        if dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            sys.exit(
                "ERROR: 'idc' command not found. Install the IDC client with:\n"
                "    pip install idc-index\n"
                "then re-run this script."
            )
        except subprocess.CalledProcessError as e:
            print(f"    !! download failed for {sid} (exit code {e.returncode}), continuing...\n")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Download CPTAC slide images for a fixed sample_id list via IDC.")
    parser.add_argument("--csv", default="cptac_slide_index.csv", help="Path to the slide index CSV")
    parser.add_argument("--outdir", default="idc_downloads", help="Directory to save downloaded slides into")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV file not found at {csv_path}")

    matches = find_matching_rows(csv_path, SAMPLE_IDS)
    report_missing(matches, SAMPLE_IDS)

    if not matches:
        sys.exit("No matching rows found. Nothing to download.")

    run_downloads(matches, Path(args.outdir), args.dry_run)


if __name__ == "__main__":
    main()
