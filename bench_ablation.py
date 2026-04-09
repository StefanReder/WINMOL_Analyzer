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

MODEL = os.environ.get(
    "WINMOL_ABLATION_MODEL",
    "/workspace/WINMOL-Analyzer/standalone/model/model_UNet_GenDS_512_2023-02-27_211141.hdf5",
)
INPUT = os.environ.get("WINMOL_ABLATION_INPUT", "/data/in/raster_7_roi.tif")
PROCESS_TYPE = os.environ.get("WINMOL_ABLATION_PROCESS_TYPE", "Nodes")

OUT_ROOT = pathlib.Path(
    os.environ.get("WINMOL_ABLATION_OUT_ROOT", "/data/out/ablation_runs")
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MULTI_GPU_VISIBLE = os.environ.get(
    "WINMOL_ABLATION_MULTI_GPU_VISIBLE_DEVICES",
    "0,1,2,3",
)

PIPELINES = [
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

MODES = [
    {
        "name": "cpu",
        "env": {
            "CUDA_VISIBLE_DEVICES": "",
        },
        "config": {
            "prediction_backend": "cpu",
            "execution_mode": "stream",
        },
    },
    {
        "name": "single_gpu",
        "env": {
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "config": {
            "prediction_backend": "single_gpu",
            "execution_mode": "stream",
        },
    },
    {
        "name": "multi_gpu",
        "env": {
            "CUDA_VISIBLE_DEVICES": MULTI_GPU_VISIBLE,
        },
        "config": {
            "prediction_backend": "multi_gpu",
            "execution_mode": "tiled",
        },
    },
]

STEMS_RE = re.compile(r"Total stems written:\s+(\d+)")
LIVE_STEMS_RE = re.compile(r"(?:final number of stems|Detected stems(?: total)?)\s*[: ]\s*(\d+)")
AVG_READ_RE = re.compile(r"avg read(?:_data)?\s+([0-9.]+)s")
AVG_INFER_RE = re.compile(r"avg infer\s+([0-9.]+)s")
VECTOR_JOBS_RE = re.compile(r"Vector jobs completed:\s+(\d+)")
ELAPSED_RE = re.compile(r"Elapsed time:\s+([0-9.]+)\s+seconds")


def build_cases():
    cases = []
    for mode in MODES:
        for pipeline in PIPELINES:
            env = {}
            env.update(mode.get("env", {}))
            env.update(pipeline.get("env", {}))

            config = {}
            config.update(mode.get("config", {}))
            config.update(pipeline.get("config", {}))

            cases.append(
                {
                    "mode": mode["name"],
                    "pipeline": pipeline["name"],
                    "name": f"{mode['name']}__{pipeline['name']}",
                    "env": env,
                    "config": config,
                }
            )
    return cases


def extract_last(pattern: re.Pattern[str], text: str, cast):
    matches = pattern.findall(text)
    if not matches:
        return None
    return cast(matches[-1])


def run_case(case: dict) -> dict:
    case_dir = OUT_ROOT / case["mode"] / case["pipeline"]
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

    final_stems_written = extract_last(STEMS_RE, text, int)
    live_stems_detected = extract_last(LIVE_STEMS_RE, text, int)
    last_avg_read_s = extract_last(AVG_READ_RE, text, float)
    last_avg_infer_s = extract_last(AVG_INFER_RE, text, float)
    vector_jobs_completed = extract_last(VECTOR_JOBS_RE, text, int)
    reported_elapsed_s = extract_last(ELAPSED_RE, text, float)

    return {
        "name": case["name"],
        "mode": case["mode"],
        "pipeline": case["pipeline"],
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 3),
        "reported_elapsed_s": reported_elapsed_s,
        "final_stems_written": final_stems_written,
        "live_stems_detected": live_stems_detected,
        "last_avg_read_s": last_avg_read_s,
        "last_avg_infer_s": last_avg_infer_s,
        "vector_jobs_completed": vector_jobs_completed,
        "log_path": str(log_path),
    }


def main():
    results = []
    cases = build_cases()

    for case in cases:
        print(f"Running {case['name']} ...", flush=True)
        result = run_case(case)
        results.append(result)
        print(result, flush=True)

    csv_path = OUT_ROOT / "ablation_results.csv"
    fieldnames = [
        "name",
        "mode",
        "pipeline",
        "returncode",
        "wall_s",
        "reported_elapsed_s",
        "final_stems_written",
        "live_stems_detected",
        "last_avg_read_s",
        "last_avg_infer_s",
        "vector_jobs_completed",
        "log_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nSorted by wall time:")
    for row in sorted(results, key=lambda x: (x["returncode"] != 0, x["wall_s"])):
        print(
            f"{row['name']:32} "
            f"rc={row['returncode']} "
            f"wall={row['wall_s']:.3f}s "
            f"elapsed={row['reported_elapsed_s']} "
            f"stems_written={row['final_stems_written']} "
            f"stems_live={row['live_stems_detected']} "
            f"avg_read={row['last_avg_read_s']} "
            f"avg_infer={row['last_avg_infer_s']} "
            f"vector_jobs={row['vector_jobs_completed']}"
        )

    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()
