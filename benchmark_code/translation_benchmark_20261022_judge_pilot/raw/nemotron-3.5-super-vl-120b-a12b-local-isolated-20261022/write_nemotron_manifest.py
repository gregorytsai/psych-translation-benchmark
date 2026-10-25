#!/usr/bin/env python3
"""Assemble a non-scoring manifest for the isolated Nemotron pilot artifacts."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/benchmark")
RUN_DIR = ROOT / "translation_benchmark_20261022_judge_pilot/raw/nemotron-3.5-super-vl-120b-a12b-local-isolated-20261022"
PILOT_DIR = ROOT / "translation_benchmark_20261022_judge_pilot"


def load(name: str) -> Any:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(name: str) -> dict[str, Any]:
    path = RUN_DIR / name
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256(path) if path.exists() else None}


def main() -> int:
    raw_name = "nemotron-3.5-super-vl-120b-a12b-local-isolated.raw.json"
    raw = load(raw_name)
    validation = load("nemotron-3.5-super-vl-120b-a12b-local-isolated.validation.track-v2.json")
    diagnostics = load("render_diagnostics.json")
    compat_smoke = load("smoke_compat_reasoning_format_deepseek.json")
    failed_smoke = load("smoke.json")
    snapshots = load("server_snapshots.json")
    attempt = load("attempt-01.result.json")
    cleanup = load("cleanup_evidence.json")

    events = raw.get("raw_events", [])
    elapsed = [float(event["elapsed_s"]) for event in events if isinstance(event.get("elapsed_s"), (int, float))]
    finishes = Counter()
    usage_totals = Counter()
    cache_values = Counter()
    parse_errors = 0
    for event in events:
        response = event.get("response") or {}
        choices = response.get("choices") or []
        if choices:
            finishes[str(choices[0].get("finish_reason"))] += 1
        usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(key), (int, float)):
                usage_totals[key] += usage[key]
        timings = response.get("timings") or {}
        if isinstance(timings.get("cache_n"), (int, float)):
            cache_values[str(timings["cache_n"])] += 1
        if event.get("parse_error"):
            parse_errors += 1

    props = snapshots.get("model_runtime", {})
    manifest = {
        "schema_version": "nemotron-local-judge-pilot-manifest-v1",
        "status": "PROVISIONAL_NOT_AGGREGATED",
        "captured_for": "translation_benchmark_20261022_judge_pilot",
        "run_dir": str(RUN_DIR),
        "frozen_inputs": {
            "packet": {"path": str(PILOT_DIR / "judge_packet.json"), "sha256": raw.get("packet_sha256")},
            "instructions": {"path": str(PILOT_DIR / "judge_instructions.txt"), "sha256": raw.get("instructions_sha256")},
            "packet_id": raw.get("packet_id"),
        },
        "model": {
            "model_family": "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
            "quantization": "UD-Q6_K_XL",
            "directory": str(ROOT / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL"),
            "shards_plus_mmproj": "4 GGUF shards + mmproj-BF16.gguf",
            "prior_download_verification": "ALL_PASS (recorded manifest; no second whole-payload hash pass)",
            "runtime_model_path": props.get("runtime_model_path"),
            "runtime_model_ftype": props.get("runtime_model_ftype"),
            "model_size_bytes_from_runtime": (snapshots.get("snapshots", {}).get("v1_models", {}).get("response", {}).get("data", [{}])[0].get("meta", {}) or {}).get("size"),
            "parameter_count_from_runtime": (snapshots.get("snapshots", {}).get("v1_models", {}).get("response", {}).get("data", [{}])[0].get("meta", {}) or {}).get("n_params"),
        },
        "runtime": {
            "executable": str(ROOT / "llama-v0.7.0/build-release/bin/llama-server"),
            "build_info": props.get("runtime_build_info"),
            "base_url": "http://127.0.0.1:18081",
            "model_alias": raw.get("judge", {}).get("model"),
            "loopback_only": True,
            "startup_args_recorded_in": str(RUN_DIR / "runtime_process_memory_snapshot.json"),
            "memory_preflight": "256 GiB unified memory; 51% free before run; Nemotron RSS approximately 112-114 GiB; existing formal/local processes unchanged",
            "chat_template_length": props.get("runtime_template_length"),
            "chat_template_sha256": props.get("runtime_template_sha256"),
            "chat_template_caps": props.get("chat_template_caps"),
            "modalities": props.get("modalities"),
        },
        "template_and_request_gate": {
            "render_diagnostics": {"path": str(RUN_DIR / "render_diagnostics.json"), "status": diagnostics.get("overall_status"), "checks_passed": diagnostics.get("checks_passed"), "checks_total": diagnostics.get("checks_total")},
            "failed_initial_smoke_preserved": {"path": str(RUN_DIR / "smoke.json"), "status": failed_smoke.get("overall_status"), "diagnosis": "response_format=json_object auto-grammar prefill rejected the native <think></think> generation suffix"},
            "compatibility_smoke": {"path": str(RUN_DIR / "smoke_compat_reasoning_format_deepseek.json"), "status": compat_smoke.get("overall_status"), "finish_reason": compat_smoke.get("finish_reason"), "usage": compat_smoke.get("usage"), "elapsed_s": compat_smoke.get("elapsed_s")},
            "generation_request": {"response_format": "json_object", "reasoning_format": "deepseek"},
            "chat_template_kwargs": {"enable_thinking": False, "preserve_reasoning": False},
            "reasoning_effort": "not sent; runtime caps report supports_reasoning_effort=false",
            "note": "reasoning_format=deepseek is a request-only llama.cpp compatibility field; the embedded/native template was not replaced or modified",
        },
        "live_run": {
            "attempt_result": attempt,
            "output_bundle": artifact(raw_name),
            "checkpoint": artifact("nemotron-3.5-super-vl-120b-a12b-local-isolated.checkpoint.json"),
            "attempt_log": artifact("attempt-01.log"),
            "counts": raw.get("counts"),
            "raw_event_count": len(events),
            "raw_event_status_counts": dict(Counter(str(event.get("status")) for event in events)),
            "latency_s": {"n": len(elapsed), "min": min(elapsed) if elapsed else None, "max": max(elapsed) if elapsed else None, "mean": statistics.mean(elapsed) if elapsed else None, "median": statistics.median(elapsed) if elapsed else None},
            "usage_token_totals": dict(usage_totals),
            "cache_n_distribution": dict(cache_values),
            "finish_reason_distribution": dict(finishes),
            "parse_errors": parse_errors,
        },
        "validation": {
            "path": str(RUN_DIR / "nemotron-3.5-super-vl-120b-a12b-local-isolated.validation.track-v2.json"),
            "status": validation.get("status"),
            "records": validation.get("records"),
            "score_shape": validation.get("score_shape"),
            "track_eligibility": validation.get("track_eligibility"),
            "raw_scores_repaired": False,
        },
        "cleanup": {
            "path": str(RUN_DIR / "cleanup_evidence.json"),
            "isolated_process_stopped": cleanup.get("isolated_process_after_stop", {}).get("returncode") == 1,
            "isolated_port_released": cleanup.get("isolated_listener_after_stop", {}).get("returncode") == 1,
            "isolated_health_after_stop": cleanup.get("isolated_health_after_stop", {}).get("stdout"),
            "formal_service_untouched": True,
            "formal_service_state_observed": "not running before and after; no launchctl start/stop/restart issued",
            "formal_8080_health_http": cleanup.get("formal_8080_health_after_stop", {}).get("stdout"),
            "formal_8080_models_http": cleanup.get("formal_8080_models_after_stop", {}).get("stdout"),
            "existing_llama_processes_snapshot": cleanup.get("existing_processes_after_stop", {}).get("stdout"),
        },
        "source_evidence": {
            "local_server_docs": [
                "llama-v0.7.0/tools/server/README.md POST /apply-template and response_format/reasoning-format documentation",
                "llama-v0.7.0/tools/server/server-common.cpp request parsing for response_format, chat_template_kwargs, enable_thinking, and reasoning_format",
                "llama-v0.7.0/common/chat.cpp preserve_reasoning -> preserve_thinking/truncate_history_thinking context mapping",
                "llama-v0.7.0/common/sampling.cpp output-format grammar prefill diagnostics",
                "llama-v0.7.0/tools/mtmd/models/nemotron-omni.cpp and model metadata for Nemotron Omni projector support",
            ],
            "official_primary_docs": [
                "https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                "https://docs.nvidia.com/nim/large-language-models/latest/get-started/advanced/get-started-nemotron-3.5-super-vl.html",
            ],
        },
        "scope_guard": [
            "No LiteLLM call",
            "No formal models.ini or Open WebUI change",
            "No frozen packet, instructions, old raw, or four-judge result change",
            "No candidate mapping exposed; output remains blind IDs",
            "First Nemotron scores are provisional and must be integrated by the separate statistics agent",
        ],
        "artifacts": [artifact(name) for name in [
            "server_snapshots.json", "render_diagnostics.json", "runtime_process_memory_snapshot.json", "smoke.json", "smoke_compat_reasoning_format_deepseek.json", raw_name, "nemotron-3.5-super-vl-120b-a12b-local-isolated.checkpoint.json", "nemotron-3.5-super-vl-120b-a12b-local-isolated.validation.track-v2.json", "attempt-01.log", "attempt-01.result.json", "cleanup_evidence.json",
        ]],
    }
    output = RUN_DIR / "nemotron_run_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "validation": validation.get("status"), "counts": raw.get("counts"), "event_status": manifest["live_run"]["raw_event_status_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
