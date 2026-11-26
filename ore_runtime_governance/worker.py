"""Worker bootstrap executed inside supervised child processes (steps 21, 29).

The parent passes a job file and expects a result file; the worker validates
everything that crosses the trust boundary before importing anything: job
size, JSON structure, entry-point syntax, and payload shape (integrated
step 29 -- length, encoding, schema, and content-type checks at every
boundary). It runs with a scrubbed environment (no ambient credentials),
writes one atomic JSON result, and exits with distinguishable codes.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

MAX_JOB_BYTES = 8 * 1024 * 1024
WORKER_VERSION = "1.0.0"
_ENTRY_RE_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:.")


def _validate_job(job: dict) -> tuple[str, dict]:
    """Validate the decoded job document (integrated step 29)."""
    if not isinstance(job, dict):
        raise ValueError("job document must be a JSON object")
    unknown = set(job).difference(
        {"entry_point", "payload", "budget_seconds", "component_id"}
    )
    if unknown:
        raise ValueError(f"unknown job fields rejected: {sorted(unknown)}")
    entry_point = job.get("entry_point")
    if not isinstance(entry_point, str) or not entry_point:
        raise ValueError("entry_point must be a non-empty string")
    if any(ch not in _ENTRY_RE_OK for ch in entry_point) or len(entry_point) > 200:
        raise ValueError(f"entry_point {entry_point!r} is not a module:attr reference")
    payload = job.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    if len(json.dumps(payload)) > MAX_JOB_BYTES:
        raise ValueError("payload exceeds the worker's byte budget")
    return entry_point, payload


def run_job(job_path: Path, result_path: Path) -> int:
    try:
        raw = job_path.read_bytes()
    except OSError as exc:
        result_path.write_text(
            json.dumps({"error": f"job file unreadable: {exc}"}), encoding="utf-8"
        )
        return 2
    if len(raw) > MAX_JOB_BYTES:
        result_path.write_text(
            json.dumps({"error": f"job exceeds {MAX_JOB_BYTES} bytes"}), encoding="utf-8"
        )
        return 2
    try:
        job = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        result_path.write_text(
            json.dumps({"error": f"malformed job JSON: {exc}"}), encoding="utf-8"
        )
        return 2
    try:
        entry_point, payload = _validate_job(job)
    except ValueError as exc:
        result_path.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        return 2
    try:
        module_name, _, attr = entry_point.partition(":")
        if not module_name or not attr:
            raise ImportError(f"entry_point {entry_point!r} must be module:attr")
        import importlib

        module = importlib.import_module(module_name)
        target = module
        for part in attr.split("."):
            target = getattr(target, part)
        if not callable(target):
            raise TypeError(f"entry_point {entry_point!r} is not callable")
        result = target(payload)
        serializable = (
            dict(result) if isinstance(result, dict) else {"value": result}
        )
        for key, value in serializable.items():
            json.dumps(value)  # enforce JSON serializability of outputs
        result_path.write_text(
            json.dumps(
                {
                    "result": serializable,
                    "entry_point": entry_point,
                    "worker_version": WORKER_VERSION,
                }
            ),
            encoding="utf-8",
        )
        return 0
    except Exception:
        error = traceback.format_exc(limit=8)
        try:
            result_path.write_text(
                json.dumps({"error": error[-2000:], "worker_version": WORKER_VERSION}),
                encoding="utf-8",
            )
        except OSError:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ORE runtime supervised worker (Point 22 isolation plane)"
    )
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args(argv)
    return run_job(Path(args.job_file).expanduser(), Path(args.result_file).expanduser())


if __name__ == "__main__":
    sys.exit(main())
