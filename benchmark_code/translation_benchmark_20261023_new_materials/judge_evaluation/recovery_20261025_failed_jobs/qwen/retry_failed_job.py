#!/usr/bin/env python3
"""One isolated, user-authorized retry of the frozen Qwen failed job."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import traceback
from pathlib import Path
from typing import Any


ROOT = Path("/benchmark")
RUNNER = ROOT / "translation_benchmark_20261023_new_materials/judge_evaluation/local_runner"
DRIVER_PATH = RUNNER / "judge_driver.py"
RECOVERY_ROOT = ROOT / "translation_benchmark_20261023_new_materials/judge_evaluation/recovery_20261025_failed_jobs/qwen"
RECOVERY_RAW_ROOT = RECOVERY_ROOT / "raw"
JOB_ID = "pw-430e98de99b3"
MODEL_ID = "qwen3.8-flash-next-xhigh"


def load_driver() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_judge_driver", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fhash(driver: Any, path: Path) -> str | None:
    return driver.sha256_file(path) if path.is_file() else None


def main() -> int:
    driver = load_driver()
    force_start = os.environ.get("QWEN_RECOVERY_FORCE_START") == "1"
    if force_start:
        original_vm_headroom = driver.vm_headroom
        def authorized_headroom(required_bytes: int, phase: str, load_estimate_bytes: int) -> dict[str, Any]:
            adjusted = max(0, required_bytes - 32 * driver.GIB)
            evidence = original_vm_headroom(adjusted, phase, load_estimate_bytes)
            evidence["user_authorized_margin_override"] = True
            evidence["original_required_bytes"] = required_bytes
            evidence["reserved_margin_gib"] = 0
            return evidence
        driver.vm_headroom = authorized_headroom
    batch_id = f"{MODEL_ID}-{JOB_ID}-user_authorized_retry-{stamp()}"
    batch_dir = RECOVERY_ROOT / batch_id
    raw_dir = RECOVERY_RAW_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    raw_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = batch_dir / "batch_manifest.json"
    result_path = batch_dir / "batch_result.json"
    progress_path = batch_dir / "progress.md"
    handoff_path = batch_dir / "handoff.json"

    parent_raw = driver.RAW_ROOT / MODEL_ID / "requests" / f"{JOB_ID}.json"
    checkpoint_path = driver.checkpoint_path(MODEL_ID)
    results_path = driver.results_path(MODEL_ID)
    parent_raw_sha = fhash(driver, parent_raw)
    checkpoint_sha_before = fhash(driver, checkpoint_path)
    results_sha_before = fhash(driver, results_path)
    if parent_raw_sha is None:
        raise FileNotFoundError(f"parent raw missing: {parent_raw}")
    parent = driver.read_json(parent_raw)
    checkpoint = driver.read_json(checkpoint_path)
    records = [item for item in checkpoint.get("records", []) if item.get("task_id") == JOB_ID]
    if len(records) != 1:
        raise RuntimeError(f"expected one checkpoint record, got {len(records)}")
    if parent.get("status") != "invalid" or parent.get("invalid_reason") != "subscores_sum_mismatch":
        raise RuntimeError("parent is not the authorized invalid job")
    if records[0].get("status") != "invalid" or records[0].get("invalid_reason") != "subscores_sum_mismatch":
        raise RuntimeError("checkpoint is not the authorized invalid job")
    parent_task = records[0].get("task")
    if not isinstance(parent_task, dict):
        raise RuntimeError("parent task missing")

    manifest_path_frozen, manifest, discover_errors = driver.discover_interface()
    if manifest_path_frozen is None or not isinstance(manifest, dict):
        raise RuntimeError(f"frozen manifest unavailable: {discover_errors}")
    packet_path, packet, packet_error = driver.load_slot_packet(manifest_path_frozen, manifest, "judge-01")
    if packet_error or packet is None:
        raise RuntimeError(f"frozen packet unavailable: {packet_error}")
    interface = driver.normalize_interface(manifest_path_frozen, manifest, packet)
    tasks = [item for item in interface["pointwise"] if item.get("task_id") == JOB_ID]
    if len(tasks) != 1:
        raise RuntimeError(f"expected one frozen task, got {len(tasks)}")
    task = tasks[0]
    if driver.sha256_json(parent_task) != driver.sha256_json(task):
        raise RuntimeError("parent task differs from frozen packet task")
    instructions_path, instructions, instructions_error = driver.load_instructions(manifest_path_frozen, manifest)
    if instructions_error or instructions is None:
        raise RuntimeError(f"frozen instructions unavailable: {instructions_error}")

    model_default = next(item for item in driver.default_model_configs() if item["id"] == MODEL_ID)
    model = checkpoint.get("model")
    if not isinstance(model, dict) or driver.sha256_json(model) != driver.sha256_json(model_default):
        raise RuntimeError("checkpoint model differs from frozen model config")
    mode_sha = driver.mode_sha256(model)
    config_sha = fhash(driver, driver.CONFIG_PATH)
    manifest_sha = fhash(driver, manifest_path_frozen)
    if checkpoint.get("mode_sha256") != mode_sha:
        raise RuntimeError("mode hash mismatch")
    if checkpoint.get("config_sha256") != config_sha:
        raise RuntimeError("config hash changed")
    if checkpoint.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("manifest hash changed")
    approval = driver.approval_state(manifest_path_frozen, manifest)
    if not approval.get("approved"):
        raise RuntimeError("ROOT_APPROVED is not valid for frozen manifest")
    inventory = driver.model_inventory(model)
    if inventory.get("errors"):
        raise RuntimeError("model inventory failed: " + "; ".join(inventory["errors"]))

    batch_manifest = {
        "schema_version": "user-authorized-failed-job-retry-v1",
        "batch_id": batch_id,
        "created_at": driver.now(),
        "authorization": "user explicitly authorized retry of pw-430e98de99b3",
        "memory_margin_override": {"enabled": force_start, "reserved_gib": 0 if force_start else 32, "authorization": "2026-10-25 user: 請強制開始；你自己做就好" if force_start else None},
        "job_id": JOB_ID,
        "attempt_index": 2,
        "parent_attempt_index": 1,
        "parent_raw_path": str(parent_raw),
        "parent_raw_sha256": parent_raw_sha,
        "parent_invalid_reason": parent.get("invalid_reason"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "results_path": str(results_path),
        "results_sha256_before": results_sha_before,
        "frozen_manifest_path": str(manifest_path_frozen),
        "frozen_manifest_sha256": manifest_sha,
        "packet_path": str(packet_path) if packet_path else None,
        "instructions_path": str(instructions_path) if instructions_path else None,
        "task_sha256": driver.sha256_json(task),
        "task": task,
        "model": model,
        "model_inventory": inventory,
        "config_sha256": config_sha,
        "mode_sha256": mode_sha,
        "driver_path": str(DRIVER_PATH),
        "driver_sha256": fhash(driver, DRIVER_PATH),
        "approval": approval,
        "raw_root": str(raw_dir),
        "fixed_generation": {
            "ctx_size": 32768,
            "temperature": 0,
            "top_p": 1,
            "seed": driver.SEED,
            "max_tokens": 16384,
            "stream": False,
            "parallel": 1,
            "cache_prompt": False,
            "speculative": "off",
            "reasoning_budget": 8192,
            "reasoning_effort": "xhigh",
            "enable_thinking": True,
            "preserve_thinking": False,
        },
        "quality_retry": False,
        "format_repair": False,
        "status": "RUNNING",
    }
    driver.write_json_immutable(manifest_path, driver.redact(batch_manifest))
    progress_path.write_text(
        f"batch={batch_id}\nstatus=STARTING_SERVER\njob_id={JOB_ID}\n"
        f"parent_raw_sha256={parent_raw_sha}\ncontroller_pid={os.getpid()}\n",
        encoding="utf-8",
    )

    # LocalServer and its preflight/cleanup artifacts are redirected only to
    # this recovery raw root.  No original checkpoint/results/raw path is used.
    driver.RAW_ROOT = raw_dir
    server = driver.LocalServer(model, batch_id)
    attempt: dict[str, Any] | None = None
    server_meta: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    attempt_path = raw_dir / "requests" / f"{JOB_ID}__attempt2_user_authorized_failed_job_retry.json"
    try:
        server_meta = server.start()
        driver.write_json(handoff_path, {
            "schema_version": "user-authorized-failed-job-retry-handoff-v1",
            "batch_id": batch_id,
            "controller_pid": os.getpid(),
            "server_pid": server_meta.get("pid"),
            "server_port": server_meta.get("port"),
            "model_id": MODEL_ID,
            "status": "ONE_RETRY",
            "updated_at": driver.now(),
        })
        progress_path.write_text(
            f"batch={batch_id}\nstatus=GENERATING_ONE_RETRY\njob_id={JOB_ID}\n"
            f"controller_pid={os.getpid()}\nserver_pid={server_meta.get('pid')}\n"
            f"server_port={server_meta.get('port')}\n",
            encoding="utf-8",
        )
        # Exactly one fresh generation. No format repair or infrastructure
        # retry is attempted by this user-authorized recovery batch.
        attempt, _info = driver.task_payload(
            server, model, task, instructions, manifest_path_frozen, repair=False
        )
        attempt["recovery_context"] = {
            "batch_id": batch_id,
            "attempt_index": 2,
            "parent_raw_path": str(parent_raw),
            "parent_raw_sha256": parent_raw_sha,
            "user_authorized": True,
            "quality_retry": False,
            "format_repair": False,
            "infrastructure_retry": False,
        }
        driver.write_json_immutable(attempt_path, driver.redact(attempt))
    except Exception as exc:
        attempt = {
            "schema_version": "user-authorized-failed-job-retry-attempt-v1",
            "model_id": MODEL_ID,
            "track": "pointwise",
            "task_id": JOB_ID,
            "status": "invalid",
            "invalid_reason": f"recovery_exception:{type(exc).__name__}:{exc}",
            "task": task,
            "recovery_context": {
                "batch_id": batch_id,
                "attempt_index": 2,
                "parent_raw_path": str(parent_raw),
                "parent_raw_sha256": parent_raw_sha,
                "user_authorized": True,
            },
        }
        driver.write_json_immutable(attempt_path, driver.redact(attempt))
        progress_path.write_text(
            f"batch={batch_id}\nstatus=FAILED\njob_id={JOB_ID}\n"
            f"error={attempt['invalid_reason']}\n",
            encoding="utf-8",
        )
    finally:
        cleanup = server.stop(reason="user_authorized_failed_job_retry_complete")

    preflight_path = raw_dir / f"{batch_id}.server_preflight.json"
    cleanup_path = raw_dir / f"{batch_id}.server_cleanup.json"
    checkpoint_sha_after = fhash(driver, checkpoint_path)
    results_sha_after = fhash(driver, results_path)
    result = {
        "schema_version": "user-authorized-failed-job-retry-result-v1",
        "batch_id": batch_id,
        "completed_at": driver.now(),
        "job_id": JOB_ID,
        "attempt_index": 2,
        "status": attempt.get("status") if attempt else "invalid",
        "invalid_reason": attempt.get("invalid_reason") if attempt else "missing_attempt",
        "raw_path": str(attempt_path),
        "raw_sha256": fhash(driver, attempt_path),
        "preflight_path": str(preflight_path) if preflight_path.is_file() else None,
        "preflight_sha256": fhash(driver, preflight_path),
        "cleanup_path": str(cleanup_path) if cleanup_path.is_file() else None,
        "cleanup_sha256": fhash(driver, cleanup_path),
        "server_meta": driver.redact(server_meta),
        "cleanup": driver.redact(cleanup),
        "response_metadata": attempt.get("response_metadata"),
        "actual_template_render": attempt.get("actual_template_render"),
        "parent_raw_path": str(parent_raw),
        "parent_raw_sha256": parent_raw_sha,
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "results_sha256_before": results_sha_before,
        "results_sha256_after": results_sha_after,
        "original_checkpoint_unchanged": checkpoint_sha_before == checkpoint_sha_after,
        "original_results_unchanged": results_sha_before == results_sha_after,
        "quality_retry": False,
        "aggregate_modified": False,
        "first_valid_only": True,
    }
    driver.write_json_immutable(result_path, driver.redact(result))
    progress_path.write_text(
        f"batch={batch_id}\nstatus={result['status']}\njob_id={JOB_ID}\n"
        f"raw_path={attempt_path}\nraw_sha256={result['raw_sha256']}\n"
        f"cleanup_path={result['cleanup_path']}\n"
        f"original_checkpoint_unchanged={result['original_checkpoint_unchanged']}\n"
        f"original_results_unchanged={result['original_results_unchanged']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "batch_id": batch_id,
        "result_path": str(result_path),
        "raw_path": str(attempt_path),
        "status": result["status"],
        "invalid_reason": result["invalid_reason"],
        "raw_sha256": result["raw_sha256"],
        "cleanup_path": result["cleanup_path"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
