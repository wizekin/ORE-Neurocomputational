#!/usr/bin/env python3
"""Run ORE's pytest suite and randomized Workbench evaluation on cloud notebooks.

This runner intentionally uses the CPU for ordinary pytest and Python-level
randomized tests. A detected GPU is reported, but is not claimed as an
accelerator unless the tested code itself uses a GPU-enabled library.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def _run_pytest(root: Path, target: str, log_path: Path) -> dict:
    command = [sys.executable, "-m", "pytest", "-v", target]
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    return {
        "command": command,
        "exit_code": process.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log": str(log_path),
        "status": "passed" if process.returncode == 0 else "failed",
    }


def _gpu_diagnostics() -> dict:
    result = {"available": False, "details": "No NVIDIA GPU detected by nvidia-smi."}
    executable = shutil.which("nvidia-smi")
    if not executable:
        return result
    process = subprocess.run(
        [executable], capture_output=True, text=True, check=False
    )
    if process.returncode == 0:
        result.update(available=True, details=process.stdout.strip())
    else:
        result["details"] = (process.stderr or process.stdout).strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pytest-target", default="tests")
    parser.add_argument("--trials-per-callable", type=int, default=5)
    parser.add_argument("--max-callables-per-package", type=int, default=0,
                        help="0 means all discovered eligible callables.")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-randomized", action="store_true")
    args = parser.parse_args()

    if args.trials_per_callable < 1:
        parser.error("--trials-per-callable must be at least 1")
    if args.max_callables_per_package < 0:
        parser.error("--max-callables-per-package cannot be negative")

    root = args.repo.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error(f"Repository root does not contain pyproject.toml: {root}")
    output_dir = (args.output_dir or root / "ore_reports" / "cloud_runs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    run_dir = output_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema": "ore.cloud_test_run.v1",
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "python": sys.version,
        "platform": sys.platform,
        "gpu": _gpu_diagnostics(),
        "execution_note": (
            "pytest and the current randomized Workbench runner execute Python code on CPU. "
            "GPU availability alone does not accelerate them."
        ),
        "pytest": None,
        "randomized_evaluation": None,
        "runner_errors": [],
    }

    if not args.skip_pytest:
        print("\n=== Running repository pytest suite ===", flush=True)
        try:
            summary["pytest"] = _run_pytest(root, args.pytest_target, run_dir / "pytest.log")
            print(f"pytest: {summary['pytest']['status']} "
                  f"(exit {summary['pytest']['exit_code']})", flush=True)
        except Exception:
            summary["runner_errors"].append("pytest runner error:\n" + traceback.format_exc())

    if not args.skip_randomized:
        print("\n=== Running ORE randomized Workbench evaluation ===", flush=True)
        try:
            workbench_dir = root / "experiment_workbench"
            if not (workbench_dir / "auto_test.py").is_file():
                raise FileNotFoundError(f"Missing {workbench_dir / 'auto_test.py'}")
            sys.path.insert(0, str(workbench_dir))
            from auto_test import run_auto_test_suite

            last_report = {"time": 0.0}

            def progress(payload: dict) -> None:
                now = time.monotonic()
                if now - last_report["time"] >= 10 or payload.get("phase", "").startswith("Reports"):
                    last_report["time"] = now
                    print(
                        "[ORE] "
                        f"{payload.get('package', 'all')} | "
                        f"{payload.get('callable', '')} | "
                        f"trial {payload.get('trial_index', 0)}/"
                        f"{payload.get('trial_count', 0)} | "
                        f"{payload.get('completed', 0)}/"
                        f"{payload.get('total', 0)} | {payload.get('phase', '')}",
                        flush=True,
                    )

            randomized = run_auto_test_suite(
                root,
                progress_callback=progress,
                trials_per_callable=args.trials_per_callable,
                max_callables_per_package=args.max_callables_per_package,
                random_seed=args.random_seed,
            )
            randomized_archive = Path(randomized["archive_path"])
            copied_archive = run_dir / randomized_archive.name
            if randomized_archive.is_file():
                shutil.copy2(randomized_archive, copied_archive)
            summary["randomized_evaluation"] = {
                "run_id": randomized["run_id"],
                "random_seed": randomized["random_seed"],
                "summary": randomized["summary"],
                "report_directory": randomized["report_directory"],
                "archive": str(copied_archive) if copied_archive.exists() else None,
            }
            print(
                "Randomized evaluation completed: "
                f"{randomized['summary']['completed_trials']}/"
                f"{randomized['summary']['planned_trials']} trials",
                flush=True,
            )
        except Exception:
            summary["runner_errors"].append(
                "Randomized evaluation runner error:\n" + traceback.format_exc()
            )

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary_path = run_dir / "cloud_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    archive_path = run_dir / "ORE-cloud-test-results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_file() and file_path != archive_path:
                archive.write(file_path, arcname=file_path.relative_to(run_dir).as_posix())

    print(f"\nResults folder: {run_dir}", flush=True)
    print(f"Downloadable archive: {archive_path}", flush=True)
    pytest_failed = bool(summary["pytest"] and summary["pytest"]["exit_code"] != 0)
    runner_failed = bool(summary["runner_errors"])
    return 1 if pytest_failed or runner_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
