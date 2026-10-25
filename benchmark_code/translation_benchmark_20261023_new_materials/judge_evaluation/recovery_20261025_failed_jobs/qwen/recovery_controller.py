#!/usr/bin/env python3
"""Single guarded RAM-wait controller for one authorized Qwen retry."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/benchmark")
RUNNER = ROOT / "translation_benchmark_20261023_new_materials/judge_evaluation/local_runner"
DRIVER_PATH = RUNNER / "judge_driver.py"
RECOVERY_ROOT = ROOT / "translation_benchmark_20261023_new_materials/judge_evaluation/recovery_20261025_failed_jobs/qwen"
RETRY_SCRIPT = RECOVERY_ROOT / "retry_failed_job.py"
GUARD_PATH = RECOVERY_ROOT / "controller.guard.json"
STATUS_PATH = RECOVERY_ROOT / "controller_status.json"
PROGRESS_PATH = RECOVERY_ROOT / "progress.md"
CHECKPOINT_PATH = RECOVERY_ROOT / "checkpoint.json"
MODEL_ID = "qwen3.8-flash-next-xhigh"
JOB_ID = "pw-430e98de99b3"
SESSION = os.environ.get("RECOVERY_CONTROLLER_SESSION", "persistent-recovery-controller")
MARGIN_GIB = 0 if os.environ.get("QWEN_RECOVERY_FORCE_START") == "1" else 32


def load_driver() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_judge_driver", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(driver: Any, path: Path, value: Any) -> None:
    driver.atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_guard() -> dict[str, Any]:
    payload = {
        "schema_version": "single-recovery-controller-guard-v1",
        "pid": os.getpid(),
        "started_at": stamp(),
        "session": SESSION,
        "job_id": JOB_ID,
        "model_id": MODEL_ID,
    }
    GUARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(GUARD_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            return payload
        except FileExistsError:
            try:
                existing = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", 0))
            except Exception as exc:
                raise RuntimeError(f"guard exists but cannot read it: {exc}") from exc
            if existing_pid and pid_alive(existing_pid):
                raise RuntimeError(f"duplicate recovery controller refused; existing PID {existing_pid}")
            GUARD_PATH.unlink()


def release_guard(guard: dict[str, Any]) -> None:
    try:
        current = json.loads(GUARD_PATH.read_text(encoding="utf-8"))
        if int(current.get("pid", -1)) == int(guard["pid"]):
            GUARD_PATH.unlink()
    except FileNotFoundError:
        return


def old_preflight_provenance() -> list[str]:
    paths = sorted(RECOVERY_ROOT.glob("*/batch_result.json"))
    return [str(path) for path in paths]


def write_state(
    driver: Any,
    status: str,
    gate: dict[str, Any],
    guard: dict[str, Any],
    checkpoint_sha: str,
    **extra: Any,
) -> None:
    checked = gate.get("checked_at", stamp())
    state = {
        "schema_version": "qwen-failed-job-recovery-checkpoint-v1",
        "updated_at": stamp(),
        "checked_at": checked,
        "status": status,
        "session": SESSION,
        "controller_pid": os.getpid(),
        "guard_path": str(GUARD_PATH),
        "job_id": JOB_ID,
        "model_id": MODEL_ID,
        "available_bytes_conservative": gate.get("available_bytes_conservative"),
        "available_gib_conservative": gate.get("available_gib_conservative"),
        "required_bytes": gate.get("required_bytes"),
        "required_gib": gate.get("required_gib"),
        "load_estimate_bytes": gate.get("load_estimate_bytes"),
        "load_estimate_gib": gate.get("load_estimate_gib"),
        "ram_gate_pass": gate.get("pass"),
        "reserved_margin_gib": MARGIN_GIB,
        "user_authorized_force_start": MARGIN_GIB == 0,
        "original_checkpoint_sha256": checkpoint_sha,
        "old_preflight_abort_provenance": old_preflight_provenance(),
        "old_preflight_abort_is_not_new_judge_attempt": True,
        "new_model_attempt_started": extra.get("new_model_attempt_started", False),
        "new_model_response_received": extra.get("new_model_response_received", False),
        "quality_retry": False,
        "format_repair": False,
    }
    state.update(extra)
    write_json(driver, CHECKPOINT_PATH, state)
    write_json(driver, STATUS_PATH, state)
    lines = [
        "# Qwen failed-job recovery",
        "",
        f"- Updated: {state['updated_at']}",
        f"- Status: {status}",
        f"- Session: {SESSION}",
        f"- Controller PID: {os.getpid()}",
        f"- Checked at: {checked}",
        f"- Available conservative: {gate.get('available_gib_conservative')} GiB",
        f"- Required (model+mmproj+{MARGIN_GIB} GiB): {gate.get('required_gib')} GiB",
        f"- RAM gate pass: {gate.get('pass')}",
        f"- Job: {JOB_ID}",
        "- Old preflight abort is provenance only; it is not a new judge attempt or final result.",
        "- No original checkpoint/results/aggregate writes are permitted.",
    ]
    if extra:
        lines.append(f"- Extra: {json.dumps(extra, ensure_ascii=False, sort_keys=True)}")
    driver.atomic_write(PROGRESS_PATH, "\n".join(lines) + "\n")


def main() -> int:
    driver = load_driver()
    guard = acquire_guard()
    checkpoint_path = driver.checkpoint_path(MODEL_ID)
    checkpoint_sha = driver.sha256_file(checkpoint_path)
    checkpoint = driver.read_json(checkpoint_path)
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("completed Qwen checkpoint has no model config")
    inventory = driver.model_inventory(model)
    if inventory.get("errors"):
        raise RuntimeError("model inventory failed: " + "; ".join(inventory["errors"]))
    required = int(inventory["load_estimate_bytes"]) + MARGIN_GIB * driver.GIB
    try:
        write_state(driver, "WAITING_FOR_RAM", driver.vm_headroom(required, "recovery_wait", inventory["load_estimate_bytes"]), guard, checkpoint_sha)
        while True:
            if driver.sha256_file(checkpoint_path) != checkpoint_sha:
                gate = driver.vm_headroom(required, "identity_abort", inventory["load_estimate_bytes"])
                write_state(driver, "ABORTED_CHECKPOINT_CHANGED", gate, guard, checkpoint_sha)
                return 3
            gate = driver.vm_headroom(required, "recovery_wait", inventory["load_estimate_bytes"])
            if not gate.get("pass"):
                write_state(driver, "WAITING_FOR_RAM", gate, guard, checkpoint_sha)
                time.sleep(60)
                continue
            listeners = driver.lsof_pids(int(model["port"]))
            if listeners or driver.port_open(int(model["port"])):
                write_state(driver, "ABORTED_PORT_NOT_FREE", gate, guard, checkpoint_sha, listeners=listeners)
                return 4
            write_state(driver, "STARTING_ONE_AUTHORIZED_RETRY", gate, guard, checkpoint_sha, new_model_attempt_started=True)
            completed = subprocess.run(
                [sys.executable, str(RETRY_SCRIPT)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            result_paths = sorted(RECOVERY_ROOT.glob("*/batch_result.json"))
            result_path = result_paths[-1] if result_paths else None
            result: dict[str, Any] = {}
            if result_path and result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    result = {}
            response_received = bool(result.get("response_metadata", {}).get("http_status") == 200)
            final_status = "RETRY_COMPLETE_VALID" if result.get("status") == "ok" else "RETRY_COMPLETE_FAILED"
            write_state(
                driver,
                final_status,
                gate,
                guard,
                checkpoint_sha,
                new_model_attempt_started=True,
                new_model_response_received=response_received,
                retry_exit_code=completed.returncode,
                retry_result_path=str(result_path) if result_path else None,
                retry_stdout_tail=completed.stdout[-2000:],
                retry_stderr_tail=completed.stderr[-2000:],
                retry_raw_path=result.get("raw_path"),
                retry_raw_sha256=result.get("raw_sha256"),
            )
            return 0 if result.get("status") == "ok" else 2
    finally:
        release_guard(guard)


if __name__ == "__main__":
    raise SystemExit(main())
