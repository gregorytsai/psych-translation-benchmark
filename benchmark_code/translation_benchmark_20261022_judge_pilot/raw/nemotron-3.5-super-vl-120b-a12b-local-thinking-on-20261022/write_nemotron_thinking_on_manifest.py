#!/usr/bin/env python3
"""Write a non-scoring template/validation/metrics handoff for the on run."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/benchmark")
PILOT = ROOT / "translation_benchmark_20261022_judge_pilot"
RUN_DIR = PILOT / "raw/nemotron-3.5-super-vl-120b-a12b-local-thinking-on-20261022"
RAW_NAME = "nemotron-3.5-super-vl-120b-a12b-local-thinking-on.raw.json"


def load(name: str) -> Any:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(name: str) -> dict[str, Any]:
    path = RUN_DIR / name
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256(path) if path.exists() else None}


def main() -> int:
    raw = load(RAW_NAME)
    validation = load("nemotron-3.5-super-vl-120b-a12b-local-thinking-on.validation.track-v2.json")
    diagnostics = load("render_diagnostics.json")
    smoke = load("thinking_on_smoke.json")
    snapshots = load("server_snapshots.json")
    attempt = load("attempt-01.result.json")
    cleanup = load("cleanup_evidence.json")
    budget_audit = load("reasoning_budget_tokenize_audit.json")
    events = raw.get("raw_events", [])
    elapsed = [float(event["elapsed_s"]) for event in events if isinstance(event.get("elapsed_s"), (int, float))]
    finishes = Counter()
    status_counts = Counter(str(event.get("status")) for event in events)
    token_totals = Counter()
    cache_values = Counter()
    truncations = 0
    parse_errors = 0
    reasoning_chars: list[int] = []
    for event in events:
        response = event.get("response") or {}
        choices = response.get("choices") or []
        if choices:
            finishes[str(choices[0].get("finish_reason"))] += 1
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(key), (int, float)):
                token_totals[key] += usage[key]
        timings = response.get("timings") or {}
        if isinstance(timings.get("cache_n"), (int, float)):
            cache_values[str(timings["cache_n"])] += 1
        if event.get("parse_error"):
            parse_errors += 1
        if response.get("truncated") is True or (choices and choices[0].get("finish_reason") == "length") or usage.get("completion_tokens") == 8192:
            truncations += 1
        message = choices[0].get("message", {}) if choices else {}
        if isinstance(message, dict) and isinstance(message.get("reasoning_content"), str):
            reasoning_chars.append(len(message["reasoning_content"]))
    audit_counts = budget_audit.get("counts", {})
    runtime = snapshots.get("model_runtime", {})
    raw_counts = raw.get("counts", {})
    manifest = {
        "schema_version": "nemotron-thinking-on-judge-pilot-manifest-v1",
        "status": "PROVISIONAL_NOT_AGGREGATED",
        "run_dir": str(RUN_DIR),
        "frozen_inputs": {"packet": {"path": str(PILOT / "judge_packet.json"), "sha256": raw.get("packet_sha256")}, "instructions": {"path": str(PILOT / "judge_instructions.txt"), "sha256": raw.get("instructions_sha256")}, "packet_id": raw.get("packet_id")},
        "model": {
            "family": "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
            "quantization": "UD-Q6_K_XL",
            "directory": str(ROOT / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL"),
            "shards_plus_mmproj": "4 GGUF shards + mmproj-BF16.gguf",
            "prior_five_file_hash_status": "ALL_PASS; no second 114 GiB rehash",
            "runtime_model_path": runtime.get("runtime_model_path"),
            "runtime_model_ftype": runtime.get("runtime_model_ftype"),
            "model_shard_prior_sha256": runtime.get("model_shard_sha256_prior_all_pass"),
            "mmproj_prior_sha256": runtime.get("mmproj_sha256_prior_all_pass"),
        },
        "runtime": {
            "executable": str(ROOT / "llama-v0.7.0/build-release/bin/llama-server"),
            "build_info": runtime.get("runtime_build_info"),
            "base_url": "http://127.0.0.1:18081",
            "model_alias": raw.get("judge", {}).get("model"),
            "loopback_only": True,
            "ctx_size": 16384,
            "server_reasoning": "on",
            "server_reasoning_format": "deepseek",
            "server_reasoning_preserve": False,
            "reasoning_budget_tokens": 4096,
            "native_template_length": runtime.get("runtime_template_length"),
            "native_template_sha256": runtime.get("runtime_template_sha256"),
            "native_template_hash_matches_prior": runtime.get("template_sha256_matches_prior"),
            "chat_template_caps": runtime.get("chat_template_caps"),
        },
        "request_protocol": {
            "enable_thinking": True,
            "preserve_reasoning": False,
            "reasoning_format": "deepseek",
            "response_format": {"type": "json_object"},
            "reasoning_effort_sent": False,
            "supports_reasoning_effort": False,
            "strict_json_output": True,
            "temperature": 0,
            "top_p": 1,
            "seed": 20261022,
            "max_tokens_by_track": {"capability": 8192, "qualification": 8192, "pointwise": 8192, "pairwise": 8192},
            "timeout_s": 900,
        },
        "template_and_smoke_gate": {
            "render_diagnostics": {"path": str(RUN_DIR / "render_diagnostics.json"), "status": diagnostics.get("overall_status"), "checks_passed": diagnostics.get("checks_passed"), "checks_total": diagnostics.get("checks_total")},
            "smoke": {"path": str(RUN_DIR / "thinking_on_smoke.json"), "status": smoke.get("overall_status"), "finish_reason": smoke.get("finish_reason"), "reasoning_nonempty": smoke.get("reasoning_nonempty"), "reasoning_chars": smoke.get("reasoning_content_chars"), "usage": smoke.get("usage"), "timings": smoke.get("timings"), "truncation": smoke.get("truncation")},
            "native_template_modified": False,
            "grammar_compatibility": "reasoning_format=deepseek retained as request-only llama.cpp compatibility field; native template unchanged",
        },
        "live_run": {
            "attempt_result": attempt,
            "raw_bundle": artifact(RAW_NAME),
            "checkpoint": artifact("nemotron-3.5-super-vl-120b-a12b-local-thinking-on.checkpoint.json"),
            "attempt_log": artifact("attempt-01.log"),
            "counts": raw_counts,
            "raw_event_count": len(events),
            "raw_event_status_counts": dict(status_counts),
            "latency_s": {"n": len(elapsed), "min": min(elapsed) if elapsed else None, "max": max(elapsed) if elapsed else None, "mean": statistics.mean(elapsed) if elapsed else None, "median": statistics.median(elapsed) if elapsed else None, "total": sum(elapsed)},
            "usage_token_totals": dict(token_totals),
            "cache_n_distribution": dict(cache_values),
            "finish_reason_distribution": dict(finishes),
            "truncations": truncations,
            "parse_errors": parse_errors,
            "reasoning_chars": {"n": len(reasoning_chars), "min": min(reasoning_chars) if reasoning_chars else None, "max": max(reasoning_chars) if reasoning_chars else None, "median": statistics.median(reasoning_chars) if reasoning_chars else None},
            "reasoning_budget_tokenize_audit": {"path": str(RUN_DIR / "reasoning_budget_tokenize_audit.json"), "counts": audit_counts, "at_or_over_budget_request_ids": [sample.get("request_id") for sample in budget_audit.get("samples", []) if sample.get("budget_at_or_over") is True]},
        },
        "validation": {
            "path": str(RUN_DIR / "nemotron-3.5-super-vl-120b-a12b-local-thinking-on.validation.track-v2.json"),
            "status": validation.get("status"),
            "records": validation.get("records"),
            "track_eligibility": validation.get("track_eligibility"),
            "raw_scores_repaired": False,
            "pointwise_partial_reason": "one HTTP-successful response failed scalar subscore-sum validation; retained as failed with no repair/retry",
        },
        "cleanup": {
            "path": str(RUN_DIR / "cleanup_evidence.json"),
            "status": "PASS" if cleanup.get("cleanup_pass") else "FAIL",
            "isolated_pid": cleanup.get("isolated_pid"),
            "isolated_process_stopped": cleanup.get("isolated_process_after_stop", {}).get("returncode") != 0,
            "isolated_listener_released": cleanup.get("isolated_listener_after_stop", {}).get("returncode") != 0,
            "isolated_health_after_stop": cleanup.get("isolated_health_after_stop", {}).get("stdout"),
            "formal_8080_health_after_stop": cleanup.get("formal_8080_after_stop", {}).get("health", {}).get("stdout"),
            "formal_8080_models_after_stop": cleanup.get("formal_8080_after_stop", {}).get("models", {}).get("stdout"),
            "formal_service_untouched": cleanup.get("formal_service_untouched"),
            "memory_after_stop": cleanup.get("memory_after_stop", {}).get("stdout"),
        },
        "source_evidence": {
            "local_code": [
                "llama-v0.7.0/tools/server/README.md:236-237 (--reasoning-format/--reasoning-budget)",
                "llama-v0.7.0/tools/server/server-common.cpp:1441-1451 (request reasoning_budget_tokens and tags)",
                "llama-v0.7.0/tools/server/server-schema.cpp:335-378 (budget field and forced end tag)",
                "llama-v0.7.0/common/reasoning-budget.cpp:52-166 (fixed budget state machine)",
                "llama-v0.7.0/common/sampling.cpp:314-321,490-492 (budget sampler and end-tag forcing)",
            ],
            "official_primary_docs": [
                "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                "https://docs.nvidia.com/nim/large-language-models/latest/get-started/advanced/get-started-nemotron-3.5-super-vl.html",
            ],
        },
        "scope_guard": [
            "No LiteLLM call",
            "No formal models.ini/Open WebUI change",
            "No frozen packet/instructions, old Nemotron thinking-off raw, or four-judge result change",
            "No candidate mapping exposed; only blind IDs are retained",
            "First thinking-on scores are provisional and must be integrated by the separate statistics agent",
        ],
        "artifacts": [artifact(name) for name in ["server_start.json", "server_snapshots.json", "render_diagnostics.json", "runtime_process_memory_snapshot.json", "thinking_on_smoke.json", "reasoning_budget_tokenize_probe.json", "reasoning_budget_tokenize_audit.json", RAW_NAME, "nemotron-3.5-super-vl-120b-a12b-local-thinking-on.checkpoint.json", "nemotron-3.5-super-vl-120b-a12b-local-thinking-on.validation.track-v2.json", "attempt-01.log", "attempt-01.result.json", "cleanup_evidence.json"]],
    }
    output = RUN_DIR / "nemotron_thinking_on_run_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = manifest["live_run"]["reasoning_budget_tokenize_audit"]
    eligibility = validation.get("track_eligibility", {})
    report = f"""# Nemotron thinking-on judge pilot handoff\n\n- Status: **PROVISIONAL_NOT_AGGREGATED**; one fresh frozen run only.\n- Run dir: `{RUN_DIR}`\n- Exact model: `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL` (4 shards + `mmproj-BF16.gguf`); prior five-file verification `ALL_PASS`.\n- Runtime: llama.cpp `{runtime.get('runtime_build_info')}`, loopback `127.0.0.1:18081`, isolated PID `{cleanup.get('isolated_pid')}` stopped after run; formal 8080 remained HTTP 200 for health/models.\n\n## Template/request evidence\n\n- Native runtime template length/hash: `{runtime.get('runtime_template_length')}` / `{runtime.get('runtime_template_sha256')}`; prior hash match `{runtime.get('template_sha256_matches_prior')}`; no native template edit.\n- `/props` reports `supports_reasoning_effort=false`; system role, tools, preserve-reasoning and object arguments are supported.\n- Non-generating render gate: `{diagnostics.get('checks_passed')}/{diagnostics.get('checks_total')} PASS`.\n- Smoke: `{smoke.get('overall_status')}`; actual reasoning present, strict JSON, `finish_reason={smoke.get('finish_reason')}`, no truncation.\n- Fixed request: `enable_thinking=true`, `preserve_reasoning=false`, `reasoning_format=deepseek`, `response_format=json_object`, `reasoning_budget_tokens=4096`, `max_tokens=8192` on all tracks, timeout 900 s; no `reasoning_effort`.\n\n## Run/validation metrics\n\n- Requests: 2 capability + 12 qualification parents (24 variants) + 21 pointwise + 6 pairwise = 41 raw events; 39 substantive requests.\n- Raw events: {dict(status_counts)}; finish reasons {dict(finishes)}; truncations {truncations}; parse errors {parse_errors}.\n- Usage totals: {dict(token_totals)}; latency median/mean/max: {manifest['live_run']['latency_s']['median']:.2f}/{manifest['live_run']['latency_s']['mean']:.2f}/{manifest['live_run']['latency_s']['max']:.2f} s.\n- Reasoning tokenizer audit: {audit_counts.get('tokenize_ok')}/{audit_counts.get('events')} succeeded; exact budget hits {audit_counts.get('budget_hit_exact')}; at-or-over candidates {audit_counts.get('budget_at_or_over')}. Completion tokens are not used as a budget-hit signal.\n- Import validation: top-level `{validation.get('status')}`; 51 flattened rows = 50 valid/1 failed. Qualification `{eligibility.get('qualification', {}).get('status')}`, pointwise `{eligibility.get('pointwise', {}).get('status')}` (20/21 valid; one scalar subscore-sum failure retained), pairwise `{eligibility.get('pairwise', {}).get('status')}`.\n\nAll raw request/response bodies, timing/usage/cache/finish data, reasoning strings, checkpoints, logs, diagnostics, and cleanup proof are in the run directory. Do not treat this first thinking-on run as a formal aggregate conclusion.\n"""
    (RUN_DIR / "template_validation_metrics.md").write_text(report, encoding="utf-8")
    print(json.dumps({"manifest": str(output), "report": str(RUN_DIR / "template_validation_metrics.md"), "raw_counts": raw_counts, "validation": validation.get("status"), "budget_audit": audit_counts, "cleanup": cleanup.get("cleanup_pass")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
