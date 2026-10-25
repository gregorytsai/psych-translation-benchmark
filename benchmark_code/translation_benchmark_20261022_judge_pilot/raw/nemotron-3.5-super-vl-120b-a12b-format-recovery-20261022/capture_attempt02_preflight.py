#!/usr/bin/env python3
"""Capture the bounded-overlap resource gate without changing services."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
TOTAL_GIB = 256.0
NEMOTRON_RESIDENT_ESTIMATE_GIB = 112.0
REQUIRED_POST_START_MARGIN_GIB = 32.0


def run(args: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"args": args, "returncode": result.returncode, "stdout": result.stdout[:20000], "stderr": result.stderr[:4000]}
    except Exception as exc:  # noqa: BLE001 - preserve diagnostics
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc)}


def curl(path: str, port: int, timeout: int = 15) -> dict[str, Any]:
    return run(["/usr/bin/curl", "--noproxy", "*", "--max-time", str(timeout), "-sS", f"http://127.0.0.1:{port}{path}"])


def http_code(path: str, port: int) -> dict[str, Any]:
    return run(["/usr/bin/curl", "--noproxy", "*", "--max-time", "15", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{port}{path}"])


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_free_percent(memory_output: str) -> float | None:
    match = re.search(r"System-wide memory free percentage:\s*([0-9.]+)%", memory_output)
    return float(match.group(1)) if match else None


def selected_slots(value: str) -> dict[str, Any]:
    try:
        slots = json.loads(value)
    except json.JSONDecodeError:
        return {"parse": "failed", "raw_sha256": sha256_text(value)}
    if not isinstance(slots, list):
        return {"parse": "unexpected_type", "raw_sha256": sha256_text(value)}
    selected: list[dict[str, Any]] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        params = slot.get("params") if isinstance(slot.get("params"), dict) else {}
        next_token = slot.get("next_token")
        next_token = next_token[0] if isinstance(next_token, list) and next_token and isinstance(next_token[0], dict) else {}
        selected.append({"id": slot.get("id"), "is_processing": slot.get("is_processing"), "id_task": slot.get("id_task"), "n_ctx": slot.get("n_ctx"), "n_prompt_tokens": slot.get("n_prompt_tokens"), "n_prompt_tokens_processed": slot.get("n_prompt_tokens_processed"), "max_tokens": params.get("max_tokens"), "n_predict": params.get("n_predict"), "n_remain": next_token.get("n_remain"), "n_decoded": next_token.get("n_decoded"), "reasoning_format": params.get("reasoning_format")})
    return {"parse": "ok", "slots": selected, "raw_sha256": sha256_text(value)}


def selected_props(value: str) -> dict[str, Any]:
    try:
        props = json.loads(value)
    except json.JSONDecodeError:
        return {"parse": "failed", "raw_sha256": sha256_text(value)}
    if not isinstance(props, dict):
        return {"parse": "unexpected_type", "raw_sha256": sha256_text(value)}
    return {"parse": "ok", "raw_sha256": sha256_text(value), "model_alias": props.get("model_alias"), "model_ftype": props.get("model_ftype"), "n_ctx": props.get("n_ctx"), "total_slots": props.get("total_slots"), "chat_template_caps": props.get("chat_template_caps"), "build_info": props.get("build_info"), "is_sleeping": props.get("is_sleeping")}


def main() -> int:
    memory = run(["/usr/bin/memory_pressure", "-Q"])
    sysctl = run(["/usr/sbin/sysctl", "kern.memorystatus_level", "vm.memory_pressure", "vm.swapusage", "vm_pressure_level_transition_threshold"])
    vm_stat = run(["/usr/bin/vm_stat"])
    swap = run(["/usr/sbin/sysctl", "vm.swapusage"])
    footprint = run(["/usr/bin/footprint", "-p", "71868"], timeout=60)
    health = http_code("/health", 18080)
    slots = curl("/slots", 18080)
    props = curl("/props", 18080, timeout=30)
    formal = http_code("/health", 8080)
    recovery_port = run(["/usr/sbin/lsof", "-nP", "-iTCP:18081", "-sTCP:LISTEN"])
    processes = run(["/bin/ps", "-o", "pid,ppid,etime,%cpu,rss,command", "-p", "71868,72376"])
    free_percent = parse_free_percent(str(memory.get("stdout", "")))
    free_gib = (TOTAL_GIB * free_percent / 100.0) if free_percent is not None else None
    expected_post_start_margin = (free_gib - NEMOTRON_RESIDENT_ESTIMATE_GIB) if free_gib is not None else None
    sysctl_text = str(sysctl.get("stdout", ""))
    non_warning = "vm.memory_pressure: 0" in sysctl_text and ("kern.memorystatus_level: 63" in sysctl_text or "kern.memorystatus_level: 0" in sysctl_text)
    decode_active = selected_slots(str(slots.get("stdout", "")))
    process_ok = processes.get("returncode") == 0 and "71868" in str(processes.get("stdout", "")) and "72376" in str(processes.get("stdout", ""))
    gate = bool(health.get("stdout") == "200" and formal.get("stdout") == "200" and free_percent is not None and free_percent >= 30 and expected_post_start_margin is not None and expected_post_start_margin >= REQUIRED_POST_START_MARGIN_GIB and non_warning and process_ok and decode_active.get("parse") == "ok" and any(slot.get("is_processing") is True for slot in decode_active.get("slots", [])) and recovery_port.get("returncode") != 0)
    evidence = {
        "schema_version": "nemotron-format-recovery-preflight-v2",
        "captured_at_unix": time.time(),
        "attempt": 2,
        "gate": {"pass": gate, "total_memory_gib": TOTAL_GIB, "free_percent": free_percent, "free_gib_estimate": free_gib, "nemotron_resident_estimate_gib": NEMOTRON_RESIDENT_ESTIMATE_GIB, "expected_post_start_margin_gib": expected_post_start_margin, "required_post_start_margin_gib": REQUIRED_POST_START_MARGIN_GIB, "memory_pressure_non_warning": non_warning, "mistral_decode_active": any(slot.get("is_processing") is True for slot in decode_active.get("slots", [])), "formal_health_200": formal.get("stdout") == "200", "recovery_port_free": recovery_port.get("returncode") != 0},
        "memory_pressure": memory,
        "sysctl_pressure_swap": sysctl,
        "vm_stat": vm_stat,
        "swap": swap,
        "mistral_footprint": {"record": footprint, "raw_sha256": sha256_text(str(footprint.get("stdout", "")))},
        "mistral_health": health,
        "mistral_slots": {"record": slots, "selected": decode_active},
        "mistral_props": {"record_sha256": sha256_text(str(props.get("stdout", ""))), "selected": selected_props(str(props.get("stdout", "")))},
        "formal_health": formal,
        "recovery_port": recovery_port,
        "processes": processes,
        "model_and_runtime": {"model": "/benchmark/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/UD-Q6_K_XL/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL-00001-of-00004.gguf", "mmproj": "/benchmark/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/mmproj-BF16.gguf", "native_template": True, "ctx_size": 32768, "reasoning_budget_tokens": 8192, "max_tokens": 16384, "formal_case_count": 1, "prior_server_start_preserved": str(RUN_DIR / "server_start.json")},
    }
    output = RUN_DIR / "preflight_attempt-02.json"
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate_pass": gate, "free_percent": free_percent, "free_gib_estimate": free_gib, "expected_post_start_margin_gib": expected_post_start_margin, "memory_pressure_non_warning": non_warning, "mistral_decode_active": evidence["gate"]["mistral_decode_active"], "mistral_health": health.get("stdout"), "formal_health": formal.get("stdout"), "recovery_port_free": recovery_port.get("returncode") != 0}, ensure_ascii=False))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
