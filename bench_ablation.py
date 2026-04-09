#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path("/workspace/WINMOL-Analyzer")
PYTHON = "python3"
RUNNER = ROOT / "winmol_run.py"

MODEL = "/workspace/WINMOL-Analyzer/standalone/model/model_UNet_GenDS_512_2023-02-27_211141.hdf5"
INPUT = "/data/in/raster_7_roi.tif"
PROCESS_TYPE = "Nodes"

OUT_ROOT = pathlib.Path("/data/out/ablation_runs")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        "name": "stripe_binary",
        "env": {
            "WINMOL_STREAM_PREDICTION": "1",
        },
        "config": {
            "stripe_pipeline": True,
            "grid_pipeline": True,
            "grid_dense_split": False,
        },
    },
    {
        "name": "grid_binary",
        "env": {
            "WINMOL_STREAM_PREDICTION": "1",
            "WINMOL_DISABLE_STRIPE_PIPELINE": "1",
        },
        "config": {
            "stripe_pipeline": False,
            "grid_pipeline": True,
            "grid_dense_split": False,
        },
    },
    {
        "name": "stream_tiled_vector",
        "env": {
            "WINMOL_STREAM_PREDICTION": "1",
            "WINMOL_TILED_VECTOR_PROCESSING": "1",
            "WINMOL_DISABLE_STRIPE_PIPELINE": "1",
            "WINMOL_DISABLE_GRID_PIPELINE": "1",
        },
        "config": {
            "stripe_pipeline": False,
            "grid_pipeline": False,
            "vector_backend": "tiled",
        },
    },
    {
        "name": "stream_global_vector",
        "env": {
            "WINMOL_STREAM_PREDICTION": "1",
            "WINMOL_DISABLE_STRIPE_PIPELINE": "1",
            "WINMOL_DISABLE_GRID_PIPELINE": "1",
        },
        "config": {
            "stripe_pipeline": False,
            "grid_pipeline": False,
            "vector_backend": "global",
        },
    },
]

STEMS_RE = re.compile(r"Total stems written:\s+(\d+)")
LIVE_STEMS_RE = re.compile(r"Detected stems(?: total)?:\s+(\d+)")
READ_RE = re.compile(r"avg read_data\s+([0-9.]+)s")


def run_case(case: dict) -> dict:
    case_dir = OUT_ROOT / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    stem_map = case_dir / "stem_map.tif"
    output_prefix = case_dir / "vectors"
    log_path = case_dir / "run.log"

    env = os.environ.copy()
    env.update(case.get("env", {}))
    env["WINMOL_CONFIG_OVERRIDES_JSON"] = json.dumps(case.get("config", {}))

    cmd = [
        PYTHON,
        "-u",
        str(RUNNER),
        MODEL,
        INPUT,
        str(stem_map),
        str(output_prefix),
        PROCESS_TYPE,
    ]

    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    wall_s = time.monotonic() - t0

    text = log_path.read_text(encoding="utf-8", errors="replace")

    final_stems = None
    m = STEMS_RE.findall(text)
    if m:
        final_stems = int(m[-1])

    live_stems = None
    m = LIVE_STEMS_RE.findall(text)
    if m:
        live_stems = int(m[-1])

    last_avg_read_data_s = None
    m = READ_RE.findall(text)
    if m:
        last_avg_read_data_s = float(m[-1])

    return {
        "case": case["name"],
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 3),
        "final_stems_written": final_stems,
        "live_stems_detected": live_stems,
        "last_avg_read_data_s": last_avg_read_data_s,
        "log_path": str(log_path),
    }


def main():
    results = []
    for case in CASES:
        print(f"Running {case['name']} ...", flush=True)
        result = run_case(case)
        results.append(result)
        print(result, flush=True)

    csv_path = OUT_ROOT / "ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "returncode",
                "wall_s",
                "final_stems_written",
                "live_stems_detected",
                "last_avg_read_data_s",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSorted by wall time:")
    for row in sorted(results, key=lambda x: (x["returncode"] != 0, x["wall_s"])):
        print(
            f"{row['case']:24} "
            f"rc={row['returncode']} "
            f"wall={row['wall_s']:.3f}s "
            f"final_stems={row['final_stems_written']} "
            f"live_stems={row['live_stems_detected']} "
            f"avg_read_data={row['last_avg_read_data_s']}"
        )

    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()
