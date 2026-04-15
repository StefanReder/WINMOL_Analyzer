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
INPUT = os.environ.get("WINMOL_ABLATION_INPUT", "/data/in/Barnekow_6.tif")
PROCESS_TYPE = os.environ.get("WINMOL_ABLATION_PROCESS_TYPE", "Nodes")

OUT_ROOT = pathlib.Path(
    os.environ.get("WINMOL_ABLATION_OUT_ROOT", "/data/out/ablation_runs")
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MULTI_GPU_VISIBLE = os.environ.get(
    "WINMOL_ABLATION_MULTI_GPU_VISIBLE_DEVICES",
    "0,1,2,3",
)

PIPELINE = {
    "name": "stream_tiled_vector",
    "runtime_path": "stream_prediction_then_tiled_vector",
    "config": {},
}

MODES = [
    {
        "name": "cpu",
        "env": {
            "CUDA_VISIBLE_DEVICES": "",
        },
        "config": {
            "prediction_backend": "cpu",
        },
    },
    {
        "name": "single_gpu",
        "env": {
            "CUDA_VISIBLE_DEVICES": "0",
        },
        "config": {
            "prediction_backend": "single_gpu",
        },
    },
    {
        "name": "multi_gpu",
        "env": {
            "CUDA_VISIBLE_DEVICES": MULTI_GPU_VISIBLE,
        },
        "config": {
            "prediction_backend": "multi_gpu",
        },
    },
]

STEMS_RE = re.compile(r"Total stems written:\s+(\d+)")
LIVE_STEMS_RE = re.compile(
    r"(?:final number of stems|Detected stems(?: total)?)\s*[: ]\s*(\d+)")
AVG_READ_RE = re.compile(r"avg read(?:_data)?\s+([0-9.]+)s")
AVG_INFER_RE = re.compile(r"avg infer\s+([0-9.]+)s")
VECTOR_JOBS_RE = re.compile(r"Vector jobs completed:\s+(\d+)")
ELAPSED_RE = re.compile(r"Elapsed time:\s+([0-9.]+)\s+seconds")
VECTOR_PREP_RE = re.compile(r"VECTOR PREP \| tiles (\d+) \| halo_px (\d+) \| prep_s ([0-9.]+)")
VECTOR_SCHED_RE = re.compile(r"VECTOR SCHEDULER \| tiles (\d+) \| outer_workers (\d+) \| inner_workers (\d+) \| cpu_budget (\d+)")
VECTOR_TOTALS_RE = re.compile(r"VECTOR STEP TOTALS \| skel_s ([0-9.]+) \| restore_s ([0-9.]+) \| build_s ([0-9.]+) \| connect_s ([0-9.]+) \| quant_s ([0-9.]+) \| write_s ([0-9.]+) \| total_s ([0-9.]+)")
VECTOR_AVGS_RE = re.compile(r"VECTOR STEP AVGS \| skel_s ([0-9.]+) \| restore_s ([0-9.]+) \| build_s ([0-9.]+) \| connect_s ([0-9.]+) \| quant_s ([0-9.]+) \| write_s ([0-9.]+) \| total_s ([0-9.]+)")
VECTOR_WALL_RE = re.compile(r"VECTOR WALL \| outer_workers (\d+) \| inner_workers (\d+) \| wall_s ([0-9.]+)")
VECTOR_PIPELINE_RE = re.compile(r"VECTOR PIPELINE TOTAL \| prep_s ([0-9.]+) \| tile_phase_s ([0-9.]+) \| merge_s ([0-9.]+) \| total_s ([0-9.]+)")
MERGE_TIMINGS_RE = re.compile(r"MERGE TIMINGS \| tile_scan_s ([0-9.]+) \| recon_s ([0-9.]+) \| write_s ([0-9.]+) \| total_s ([0-9.]+)")


def build_cases():
    cases = []
    for mode in MODES:
        env = {}
        env.update(mode.get("env", {}))

        config = {}
        config.update(PIPELINE.get("config", {}))
        config.update(mode.get("config", {}))

        cases.append(
            {
                "mode": mode["name"],
                "pipeline": PIPELINE["name"],
                "runtime_path": PIPELINE["runtime_path"],
                "name": f"{mode['name']}__{PIPELINE['name']}",
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


def extract_last_groups(pattern: re.Pattern[str], text: str, casts):
    matches = pattern.findall(text)
    if not matches:
        return None
    values = matches[-1]
    if not isinstance(values, tuple):
        values = (values,)
    return tuple(c(v) for c, v in zip(casts, values))


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

    vector_prep = extract_last_groups(VECTOR_PREP_RE, text, (int, int, float))
    vector_sched = extract_last_groups(VECTOR_SCHED_RE, text, (int, int, int, int))
    vector_totals = extract_last_groups(VECTOR_TOTALS_RE, text, (float, float, float, float, float, float, float))
    vector_avgs = extract_last_groups(VECTOR_AVGS_RE, text, (float, float, float, float, float, float, float))
    vector_wall = extract_last_groups(VECTOR_WALL_RE, text, (int, int, float))
    vector_pipeline = extract_last_groups(VECTOR_PIPELINE_RE, text, (float, float, float, float))
    merge_timings = extract_last_groups(MERGE_TIMINGS_RE, text, (float, float, float, float))

    return {
        "name": case["name"],
        "mode": case["mode"],
        "pipeline": case["pipeline"],
        "runtime_path": case["runtime_path"],
        "returncode": proc.returncode,
        "wall_s": round(wall_s, 3),
        "reported_elapsed_s": reported_elapsed_s,
        "final_stems_written": final_stems_written,
        "live_stems_detected": live_stems_detected,
        "last_avg_read_s": last_avg_read_s,
        "last_avg_infer_s": last_avg_infer_s,
        "vector_jobs_completed": vector_jobs_completed,
        "vector_tiles": None if vector_prep is None else vector_prep[0],
        "vector_halo_px": None if vector_prep is None else vector_prep[1],
        "vector_prep_s": None if vector_prep is None else vector_prep[2],
        "vector_outer_workers": None if vector_sched is None else vector_sched[1],
        "vector_inner_workers": None if vector_sched is None else vector_sched[2],
        "vector_cpu_budget": None if vector_sched is None else vector_sched[3],
        "vector_skel_total_s": None if vector_totals is None else vector_totals[0],
        "vector_restore_total_s": None if vector_totals is None else vector_totals[1],
        "vector_build_total_s": None if vector_totals is None else vector_totals[2],
        "vector_connect_total_s": None if vector_totals is None else vector_totals[3],
        "vector_quant_total_s": None if vector_totals is None else vector_totals[4],
        "vector_write_total_s": None if vector_totals is None else vector_totals[5],
        "vector_tile_total_s": None if vector_totals is None else vector_totals[6],
        "vector_tile_wall_s": None if vector_wall is None else vector_wall[2],
        "vector_avg_tile_s": None if vector_avgs is None else vector_avgs[6],
        "vector_pipeline_prep_s": None if vector_pipeline is None else vector_pipeline[0],
        "vector_pipeline_tile_phase_s": None if vector_pipeline is None else vector_pipeline[1],
        "vector_pipeline_merge_s": None if vector_pipeline is None else vector_pipeline[2],
        "vector_pipeline_total_s": None if vector_pipeline is None else vector_pipeline[3],
        "merge_tile_scan_s": None if merge_timings is None else merge_timings[0],
        "merge_recon_s": None if merge_timings is None else merge_timings[1],
        "merge_write_s": None if merge_timings is None else merge_timings[2],
        "merge_total_s": None if merge_timings is None else merge_timings[3],
        "log_path": str(log_path),
    }


def main():
    results = []
    cases = build_cases()

    print("Benchmark cases:")
    for case in cases:
        print(
            f"  {case['name']:34} -> {case['runtime_path']} "
            f"(backend={case['config']['prediction_backend']})"
        )

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
        "runtime_path",
        "returncode",
        "wall_s",
        "reported_elapsed_s",
        "final_stems_written",
        "live_stems_detected",
        "last_avg_read_s",
        "last_avg_infer_s",
        "vector_jobs_completed",
        "vector_tiles",
        "vector_halo_px",
        "vector_prep_s",
        "vector_outer_workers",
        "vector_inner_workers",
        "vector_cpu_budget",
        "vector_skel_total_s",
        "vector_restore_total_s",
        "vector_build_total_s",
        "vector_connect_total_s",
        "vector_quant_total_s",
        "vector_write_total_s",
        "vector_tile_total_s",
        "vector_tile_wall_s",
        "vector_avg_tile_s",
        "vector_pipeline_prep_s",
        "vector_pipeline_tile_phase_s",
        "vector_pipeline_merge_s",
        "vector_pipeline_total_s",
        "merge_tile_scan_s",
        "merge_recon_s",
        "merge_write_s",
        "merge_total_s",
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
            f"path={row['runtime_path']:36} "
            f"rc={row['returncode']} "
            f"wall={row['wall_s']:.3f}s "
            f"elapsed={row['reported_elapsed_s']} "
            f"stems_written={row['final_stems_written']} "
            f"stems_live={row['live_stems_detected']} "
            f"avg_read={row['last_avg_read_s']} "
            f"avg_infer={row['last_avg_infer_s']} "
            f"vec_tile_wall={row['vector_tile_wall_s']} "
            f"vec_connect={row['vector_connect_total_s']} "
            f"vec_quant={row['vector_quant_total_s']} "
            f"merge={row['merge_total_s']} "
            f"outer/inner={row['vector_outer_workers']}/{row['vector_inner_workers']}"
        )

    print("\nVector bottleneck summary:")
    for row in results:
        tile_phase = row.get('vector_pipeline_tile_phase_s') or row.get('vector_tile_wall_s')
        merge_phase = row.get('vector_pipeline_merge_s') or row.get('merge_total_s')
        total_vec = row.get('vector_pipeline_total_s')
        if total_vec in (None, 0):
            total_vec = None
        vec_share = None if total_vec is None else 100.0 * ((tile_phase or 0.0) + (merge_phase or 0.0)) / max(total_vec, 1e-9)
        print(
            f"  {row['name']:32} tile_phase={tile_phase} merge={merge_phase} "
            f"connect={row.get('vector_connect_total_s')} quant={row.get('vector_quant_total_s')} "
            f"vec_share={None if vec_share is None else round(vec_share, 1)}%"
        )

    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()
