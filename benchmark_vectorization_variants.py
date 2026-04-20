#!/usr/bin/env python3
"""Benchmark Vectorization variants by temporarily swapping utils/Vectorization.py.

Expected files in project utils/:
- Vectorization.py                    (current active version)
- Vectorization_release.py            (release candidate)
- Vectorization_stream_tiled_cleanup.py (other candidate)

This script:
- backs up the current active Vectorization.py
- activates each variant one by one by copying it to utils/Vectorization.py
- optionally clears __pycache__ entries for Vectorization
- runs bench_ablation.py in a subprocess
- stores stdout/stderr and copies ablation_results.csv into a per-variant folder
- restores the original Vectorization.py at the end, even on failure

Usage example:
  python3 benchmark_vectorization_variants.py \
      --project /workspace/WINMOL-Analyzer \
      --variants current release cleanup

Optional environment forwarding example:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark_vectorization_variants.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict


VARIANT_MAP: Dict[str, str] = {
    "current": "Vectorization.py",
    "release": "Vectorization_release.py",
    "cleanup": "Vectorization_stream_tiled_cleanup.py",
}


def clear_vectorization_pyc(utils_dir: Path) -> None:
    pycache = utils_dir / "__pycache__"
    if not pycache.exists():
        return
    for p in pycache.glob("Vectorization*.pyc"):
        try:
            p.unlink()
        except OSError:
            pass


def copy_active(src: Path, active: Path) -> None:
    shutil.copy2(src, active)
    clear_vectorization_pyc(active.parent)


def run_variant(project_dir: Path, variant_name: str, src_file: Path, out_root: Path) -> int:
    active_file = project_dir / "utils" / "Vectorization.py"
    variant_dir = out_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Activating variant: {variant_name} ===")
    print(f"Source: {src_file}")
    copy_active(src_file, active_file)

    cmd = [sys.executable, "-u", "bench_ablation.py"]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(project_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    (variant_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (variant_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    (variant_dir / "meta.txt").write_text(
        f"variant={variant_name}\n"
        f"source={src_file}\n"
        f"returncode={proc.returncode}\n"
        f"wall_s={elapsed:.3f}\n",
        encoding="utf-8",
    )

    csv_src = project_dir / "data" / "out" / "ablation_runs" / "ablation_results.csv"
    # preferred path from your logs is /data/out/ablation_runs inside container,
    # which is normally mounted there; also try relative project path if present.
    alt_csv_src = Path("/data/out/ablation_runs/ablation_results.csv")

    if alt_csv_src.exists():
        shutil.copy2(alt_csv_src, variant_dir / "ablation_results.csv")
    elif csv_src.exists():
        shutil.copy2(csv_src, variant_dir / "ablation_results.csv")

    print(proc.stdout)
    if proc.stderr.strip():
        print("--- stderr ---")
        print(proc.stderr)

    print(f"=== Finished {variant_name} | rc={proc.returncode} | wall={elapsed:.1f}s ===")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default="/workspace/WINMOL-Analyzer",
        help="Path to the WINMOL project root",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["current", "release", "cleanup"],
        choices=sorted(VARIANT_MAP.keys()),
        help="Which variants to benchmark",
    )
    parser.add_argument(
        "--out-dir",
        default="/data/out/vectorization_variant_benchmarks",
        help="Directory for per-variant logs/results",
    )
    args = parser.parse_args()

    project_dir = Path(args.project).resolve()
    utils_dir = project_dir / "utils"
    active_file = utils_dir / "Vectorization.py"
    backup_file = utils_dir / "Vectorization.py.benchmark_backup"
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not active_file.exists():
        raise FileNotFoundError(f"Missing active file: {active_file}")

    for name in args.variants:
        src = utils_dir / VARIANT_MAP[name]
        if not src.exists():
            raise FileNotFoundError(f"Missing variant file for '{name}': {src}")

    print(f"Project: {project_dir}")
    print(f"Output:  {out_root}")
    print(f"Variants: {', '.join(args.variants)}")

    shutil.copy2(active_file, backup_file)
    results = {}
    try:
        for name in args.variants:
            src = utils_dir / VARIANT_MAP[name]
            rc = run_variant(project_dir, name, src, out_root)
            results[name] = rc
    finally:
        print("\n=== Restoring original Vectorization.py ===")
        if backup_file.exists():
            shutil.copy2(backup_file, active_file)
            clear_vectorization_pyc(utils_dir)
            try:
                backup_file.unlink()
            except OSError:
                pass

    print("\nSummary:")
    for name in args.variants:
        print(f"  {name}: rc={results.get(name, 'not-run')}")
    print(f"\nSaved logs/results to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
