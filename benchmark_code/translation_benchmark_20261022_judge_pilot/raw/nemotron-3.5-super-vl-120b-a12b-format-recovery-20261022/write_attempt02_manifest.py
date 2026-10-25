#!/usr/bin/env python3
"""Write the auditable single-case recovery bundle and manifest."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
PARENT_DIR = RUN_DIR.parents[1]
PACKET = PARENT_DIR / "judge_packet.json"
CASE = RUN_DIR / "recovery_case_attempt-02.json"
SMOKE = RUN_DIR / "recovery_smoke_attempt-02.json"
TEMPLATE = RUN_DIR / "recovery_template_evidence_attempt-02.json"
PREFLIGHT = RUN_DIR / "preflight_attempt-02.json"
START = RUN_DIR / "server_start_attempt-02.json"
CLEANUP = RUN_DIR / "recovery_cleanup_evidence_attempt-02.json"
VALIDATION = RUN_DIR / "recovery_validation_attempt-02.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mtime(path: Path) -> float:
    return path.stat().st_mtime


def main() -> int:
    case = read(CASE)
    smoke = read(SMOKE)
    template = read(TEMPLATE)
    preflight = read(PREFLIGHT)
    start = read(START)
    cleanup = read(CLEANUP)
    validation = read(VALIDATION) if VALIDATION.exists() else {}
    packet = read(PACKET)
    row = case["record"]
    event = case["raw_event"]
    elapsed = float(case["metrics"].get("api_elapsed_s") or 0.0)
    finished = mtime(CASE)
    started = finished - elapsed
    case_status = row.get("status") == "ok" and case.get("metrics", {}).get("strict_json_pass") is True and case.get("metrics", {}).get("score_shape_reason") == ""
    bundle = {
        "schema_version": "judge-pilot-live-bundle-v1",
        "status": "COMPLETE_RECOVERY_SUBSET",
        "complete": True,
        "packet_id": case.get("packet_id"),
        "packet_sha256": case.get("packet_sha256"),
        "instructions_sha256": case.get("instructions_sha256"),
        "judge": {"judge_id": case.get("judge_id"), "family": "nvidia-nemotron", "provider": "llama.cpp-local-isolated", "model": case.get("settings", {}).get("model"), "base_url": case.get("settings", {}).get("base_url"), "auth_env": None, "auth_present": False, "ui_model_version": None},
        "run_id": case.get("run_id"),
        "settings": {**case.get("settings", {}), "strict_json_output": True, "max_concurrency": 1, "chat_template_owner": "endpoint/llama.cpp native template", "extra_body_keys": ["chat_template_kwargs", "reasoning_budget_tokens", "reasoning_format"], "response_format_requested": "json_object", "checkpoint_enabled": False},
        "capability_checks": [{"check_id": "recovery-smoke-attempt-02", "kind": "request_format_smoke", "status": "ok" if smoke.get("overall_status") == "PASS" else "failed", "parsed": smoke.get("parsed_json"), "elapsed_s": smoke.get("elapsed_s"), "reason": smoke.get("error"), "template_verification": "attempt-02"}],
        "template_verification": {"status": "PASS" if template.get("checks_passed") == template.get("checks_total") and template.get("runtime", {}).get("template_hash_matches_prior") else "FAIL", "evidence": str(TEMPLATE), "runtime_template_sha256": template.get("runtime", {}).get("runtime_template_sha256"), "checks_passed": template.get("checks_passed"), "checks_total": template.get("checks_total"), "caps": template.get("runtime", {}).get("chat_template_caps")},
        "records": [row],
        "raw_events": [event],
        "completed_request_ids": [case.get("request_id")],
        "counts": {"records": 1, "ok": 1 if row.get("status") == "ok" else 0, "failed": 1 if row.get("status") == "failed" else 0, "qualification": 0, "pointwise": 1, "pairwise": 0},
        "expected_tracks": {"qualification": 12, "pointwise": 21, "pairwise": 6},
        "incomplete_expected_tracks": True,
        "subset_metadata": {"intentional": True, "kind": "single_format_recovery", "target": "src-03/cand-02", "attempt": 2, "formal_request_count": 1, "smoke_request_count": 1, "prior_server_start_preserved": str(RUN_DIR / "server_start.json"), "prior_raw_immutable": True, "not_a_sixth_judge": True, "not_a_uniform_39_request_run": True},
    }
    bundle_path = RUN_DIR / "recovery_bundle_attempt-02.json"
    write(bundle_path, bundle)
    overlap_observations = [
        {"observed_at_unix": 1792683223.864634, "nemo_task": 68, "nemo_is_processing": True, "nemo_n_decoded": 846, "mistral_task": 34142, "mistral_is_processing": True, "mistral_n_decoded": 984, "memory_free_percent": 19, "vm_memory_pressure": 0},
        {"observed_at_unix": 1792683331.078051, "nemo_task": 68, "nemo_is_processing": True, "nemo_n_decoded": 1980, "mistral_task": 34142, "mistral_is_processing": True, "mistral_n_decoded": 1747, "memory_free_percent": 19, "vm_memory_pressure": 0},
        {"observed_at_unix": 1792683612.640856, "nemo_task": 68, "nemo_is_processing": True, "nemo_n_decoded": 4927, "mistral_task": 34142, "mistral_is_processing": True, "mistral_n_decoded": 3732, "memory_free_percent": 19, "vm_memory_pressure": 0},
        {"observed_at_unix": 1792683752.8378198, "nemo_task": 68, "nemo_is_processing": True, "nemo_n_decoded": 6284, "mistral_task": 38228, "mistral_is_processing": True, "mistral_n_decoded": 551, "memory_free_percent": 19, "vm_memory_pressure": 0},
        {"observed_at_unix": 1792683824.3056571, "nemo_task": 68, "nemo_is_processing": False, "nemo_n_decoded": 0, "mistral_task": 38228, "mistral_is_processing": True, "mistral_n_decoded": 1064, "memory_free_percent": 19, "vm_memory_pressure": 0},
    ]
    manifest = {
        "schema_version": "nemotron-format-recovery-manifest-v2",
        "status": "COMPLETE_RECOVERY_DIAGNOSTIC" if case_status and cleanup.get("cleanup_pass") is True else "INCOMPLETE_OR_FAILED",
        "attempt": 2,
        "run_id": case.get("run_id"),
        "judge_id": case.get("judge_id"),
        "packet": {"path": str(PACKET), "packet_id": packet.get("packet_id"), "packet_sha256": case.get("packet_sha256"), "instructions_sha256": case.get("instructions_sha256"), "parent_raw_immutable": True},
        "exact_model": {"name": "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL", "model": start.get("model"), "mmproj": start.get("mmproj"), "native_template": True, "prior_hash_evidence": "ALL_PASS; not rehashed in this recovery"},
        "protocol": {"formal_case_count": 1, "smoke_count": 1, "retry_count": 0, "same_blind_target": "src-03/cand-02", "temperature": 0, "top_p": 1, "seed": 20261022, "reasoning_budget_tokens": 8192, "max_tokens": 16384, "context_size": 32768, "timeout_s": 1800, "enable_thinking": True, "preserve_reasoning": False, "reasoning_format": "deepseek", "response_format": "json_object", "strict_json_validation": True, "supports_reasoning_effort": False},
        "template": {"evidence": str(TEMPLATE), "status": bundle["template_verification"]["status"], "runtime_template_sha256": bundle["template_verification"]["runtime_template_sha256"], "checks_passed": bundle["template_verification"]["checks_passed"], "checks_total": bundle["template_verification"]["checks_total"]},
        "smoke": {"path": str(SMOKE), "status": smoke.get("overall_status"), "http_status": smoke.get("http_status"), "finish_reason": smoke.get("finish_reason"), "json_shape_pass": smoke.get("json_shape_pass"), "reasoning_content_present": smoke.get("reasoning_nonempty"), "reasoning_chars": smoke.get("reasoning_chars"), "elapsed_s": smoke.get("elapsed_s"), "usage": smoke.get("usage"), "timings": smoke.get("timings"), "truncation": smoke.get("truncation")},
        "formal": {"path": str(CASE), "bundle_path": str(bundle_path), "status": case.get("status"), "http_status": case.get("metrics", {}).get("http_status"), "finish_reason": case.get("metrics", {}).get("finish_reason"), "strict_json_pass": case.get("metrics", {}).get("strict_json_pass"), "score_shape_pass": case_status, "score": row.get("pointwise", {}).get("score_0_100"), "subscores": row.get("pointwise", {}).get("subscores"), "api_elapsed_s": elapsed, "usage": case.get("metrics", {}).get("usage"), "server_timings": case.get("metrics", {}).get("server_timings"), "reasoning_content_present": case.get("metrics", {}).get("reasoning_content_present"), "reasoning_chars": case.get("metrics", {}).get("reasoning_chars"), "reasoning_token_count": case.get("metrics", {}).get("reasoning_token_audit", {}).get("token_count"), "reasoning_budget_hit_exact": case.get("metrics", {}).get("reasoning_token_audit", {}).get("budget_hit_exact"), "truncation": case.get("metrics", {}).get("truncation")},
        "validation": {"path": str(VALIDATION), "status": validation.get("status"), "score_shape": validation.get("score_shape", {}).get("status"), "track_eligibility": validation.get("track_eligibility")},
        "resource_gate": {"path": str(PREFLIGHT), "gate_pass": preflight.get("gate", {}).get("pass"), "free_percent_pre_start": preflight.get("gate", {}).get("free_percent"), "free_gib_estimate_pre_start": preflight.get("gate", {}).get("free_gib_estimate"), "expected_post_start_margin_gib": preflight.get("gate", {}).get("expected_post_start_margin_gib"), "memory_pressure_non_warning": preflight.get("gate", {}).get("memory_pressure_non_warning"), "mistral_decode_active_pre_start": preflight.get("gate", {}).get("mistral_decode_active")},
        "overlap": {"active_mistral_pid": 71868, "active_mistral_runner_pid": 72376, "server_pid": start.get("pid"), "server_started_at_unix": start.get("started_at_unix"), "formal_started_estimate_unix": started, "formal_finished_estimate_unix": finished, "formal_overlap_seconds": elapsed, "observations": overlap_observations, "interpretation": "Concurrent native slot progress and 19% free memory were observed; latency is diagnostic context only and does not alter score selection."},
        "cleanup": {"path": str(CLEANUP), "status": cleanup.get("cleanup_pass"), "recovery_pid": cleanup.get("recovery_pid"), "recovery_health_after": cleanup.get("recovery_health_after_stop", {}).get("stdout"), "mistral_health_after": cleanup.get("mistral_18080_after_stop", {}).get("stdout"), "formal_health_after": cleanup.get("formal_8080_after_stop", {}).get("stdout"), "formal_models_after": cleanup.get("formal_8080_models_after_stop", {}).get("stdout"), "memory_after": cleanup.get("memory_after_stop", {}).get("stdout")},
        "composition": {"prior_uniform_run_valid_pointwise": 20, "recovery_valid_if_strict": 1 if case_status else 0, "formal_eligibility": "diagnostic-only; not a uniform-run replacement", "not_a_sixth_judge": True},
        "artifacts_sha256": {name: sha256_file(path) for name, path in (("case", CASE), ("bundle", bundle_path), ("smoke", SMOKE), ("template", TEMPLATE), ("preflight", PREFLIGHT), ("start", START), ("cleanup", CLEANUP), ("validation", VALIDATION)) if path.exists()},
        "created_at_unix": time.time(),
    }
    write(RUN_DIR / "recovery_manifest_attempt-02.json", manifest)
    print(json.dumps({"manifest_status": manifest["status"], "bundle": str(bundle_path), "score": manifest["formal"]["score"], "api_elapsed_s": elapsed, "reasoning_tokens": manifest["formal"]["reasoning_token_count"], "cleanup_pass": cleanup.get("cleanup_pass")}, ensure_ascii=False))
    return 0 if manifest["status"] == "COMPLETE_RECOVERY_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())
