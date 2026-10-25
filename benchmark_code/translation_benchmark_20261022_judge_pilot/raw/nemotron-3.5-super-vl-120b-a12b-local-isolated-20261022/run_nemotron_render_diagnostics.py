#!/usr/bin/env python3
"""Capture local Nemotron runtime metadata and non-generating template evidence.

This script intentionally performs no model generation.  It records the exact
request/response bodies sent to /apply-template, plus health/models/props/slots
snapshots, so the subsequent generation run has an independently reviewable
template and architecture gate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:18081"
MODEL_ALIAS = "nemotron-3.5-super-vl-120b-a12b-local-isolated"
MODEL_SHARD = Path(
    "/benchmark/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/UD-Q6_K_XL/"
    "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL-00001-of-00004.gguf"
)
MM_PROJ = Path(
    "/benchmark/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/mmproj-BF16.gguf"
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=payload, headers=headers, method=method)
    started = time.monotonic()
    result: dict[str, Any] = {
        "method": method,
        "path": path,
        "request": body,
    }
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
            elapsed = time.monotonic() - started
            text = raw.decode("utf-8", errors="replace")
            result.update(
                {
                    "status": "ok",
                    "http_status": response.status,
                    "elapsed_s": elapsed,
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "response": json.loads(text),
                }
            )
            return result
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed = time.monotonic() - started
        result.update(
            {
                "status": "failed",
                "http_status": exc.code,
                "elapsed_s": elapsed,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "error": raw.decode("utf-8", errors="replace")[:8000],
            }
        )
        return result
    except Exception as exc:  # preserve diagnostics as data
        result.update({"status": "failed", "elapsed_s": time.monotonic() - started, "error": repr(exc)})
        return result


def text_of(record: dict[str, Any]) -> str:
    response = record.get("response")
    return response.get("prompt", "") if isinstance(response, dict) else ""


def check(record: dict[str, Any], *, contains: list[str] = (), absent: list[str] = (), suffix: str | None = None) -> dict[str, Any]:
    prompt = text_of(record)
    checks = {
        "http_ok": record.get("status") == "ok" and record.get("http_status") == 200,
        "contains": {needle: needle in prompt for needle in contains},
        "absent": {needle: needle not in prompt for needle in absent},
    }
    if suffix is not None:
        checks["suffix"] = {"expected": suffix, "actual": prompt[-len(suffix) :] if suffix else "", "match": prompt.endswith(suffix)}
    checks["pass"] = bool(checks["http_ok"]) and all(checks["contains"].values()) and all(checks["absent"].values()) and ("suffix" not in checks or checks["suffix"]["match"])
    return checks


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = {name: request("GET", path) for name, path in {
        "health": "/health",
        "v1_models": "/v1/models",
        "props": "/props",
        "slots": "/slots",
    }.items()}
    props = snapshots["props"].get("response", {})
    template = props.get("chat_template") if isinstance(props, dict) else None
    model_meta = {
        "alias": MODEL_ALIAS,
        "model_shard": str(MODEL_SHARD),
        "model_shard_exists": MODEL_SHARD.exists(),
        "model_shard_size_bytes": MODEL_SHARD.stat().st_size if MODEL_SHARD.exists() else None,
        "model_shard_sha256": "ec945391f2b6de496cc888d2523a8004101caf0982055273dd081dd4f3803a2d",
        "mmproj": str(MM_PROJ),
        "mmproj_exists": MM_PROJ.exists(),
        "mmproj_size_bytes": MM_PROJ.stat().st_size if MM_PROJ.exists() else None,
        "mmproj_sha256": "9c09381a5744d0ea9a30048c94922f6f25ee0d63842de9c45779eca4fb5924d9",
        "runtime_template_length": len(template) if isinstance(template, str) else None,
        "runtime_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None,
        "runtime_build_info": props.get("build_info") if isinstance(props, dict) else None,
        "runtime_model_path": props.get("model_path") if isinstance(props, dict) else None,
        "runtime_model_ftype": props.get("model_ftype") if isinstance(props, dict) else None,
        "chat_template_caps": props.get("chat_template_caps") if isinstance(props, dict) else None,
        "modalities": {key: props.get(key) for key in ("modalities", "vision", "audio", "video") if isinstance(props, dict) and key in props},
    }
    write_json(RUN_DIR / "server_snapshots.json", {"base_url": BASE_URL, "snapshots": snapshots, "model_runtime": model_meta})

    common_messages = [
        {"role": "system", "content": "SYS_SENTINEL_9f7d"},
        {"role": "user", "content": "USR_SENTINEL_4c2a"},
    ]
    render_cases: list[tuple[str, dict[str, Any]]] = [
        (
            "system_user_thinking_on",
            {"messages": common_messages, "add_generation_prompt": True},
        ),
        (
            "system_user_thinking_off",
            {"messages": common_messages, "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}},
        ),
        (
            "history_preserve_on",
            {
                "messages": [
                    {"role": "system", "content": "SYS_HISTORY_1a"},
                    {"role": "user", "content": "USR_HISTORY_2b"},
                    {"role": "assistant", "reasoning_content": "PRIOR_REASONING_MARKER", "content": "PRIOR_ANSWER_MARKER"},
                    {"role": "user", "content": "USR_NEXT_3c"},
                ],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": True},
            },
        ),
        (
            "history_preserve_off",
            {
                "messages": [
                    {"role": "system", "content": "SYS_HISTORY_1a"},
                    {"role": "user", "content": "USR_HISTORY_2b"},
                    {"role": "assistant", "reasoning_content": "PRIOR_REASONING_MARKER", "content": "PRIOR_ANSWER_MARKER"},
                    {"role": "user", "content": "USR_NEXT_3c"},
                ],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False},
            },
        ),
        (
            "tool_role_and_tools",
            {
                "messages": [
                    {"role": "system", "content": "SYS_TOOL_61aa"},
                    {"role": "user", "content": "USR_TOOL_72bb"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup_rate", "arguments": {"currency": "TWD"}}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "TOOL_RESULT_83cc"},
                ],
                "tools": [{"type": "function", "function": {"name": "lookup_rate", "description": "TOOL_DESC_SENTINEL", "parameters": {"type": "object", "properties": {"currency": {"type": "string"}}, "required": ["currency"]}}}],
                "tool_choice": "auto",
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False, "preserve_reasoning": False},
            },
        ),
        (
            "json_request_shape_non_generating",
            {
                "model": MODEL_ALIAS,
                "messages": common_messages,
                "temperature": 0,
                "top_p": 1,
                "seed": 20261022,
                "max_tokens": 120,
                "stream": False,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False, "preserve_reasoning": False},
                "add_generation_prompt": True,
            },
        ),
    ]
    render_records: list[dict[str, Any]] = []
    for name, body in render_cases:
        record = request("POST", "/apply-template", body)
        record["case"] = name
        if name == "system_user_thinking_on":
            record["checks"] = check(record, contains=["SYS_SENTINEL_9f7d", "USR_SENTINEL_4c2a"], suffix="<|im_start|>assistant\n<think>\n")
        elif name == "system_user_thinking_off":
            record["checks"] = check(record, contains=["SYS_SENTINEL_9f7d", "USR_SENTINEL_4c2a"], suffix="<|im_start|>assistant\n<think></think>")
        elif name == "history_preserve_on":
            record["checks"] = check(record, contains=["PRIOR_REASONING_MARKER", "PRIOR_ANSWER_MARKER", "USR_NEXT_3c"])
        elif name == "history_preserve_off":
            record["checks"] = check(record, contains=["PRIOR_ANSWER_MARKER", "USR_NEXT_3c"], absent=["PRIOR_REASONING_MARKER"])
        elif name == "tool_role_and_tools":
            record["checks"] = check(record, contains=["TOOL_DESC_SENTINEL", "TOOL_RESULT_83cc", "SYS_TOOL_61aa", "USR_TOOL_72bb"])
        elif name == "json_request_shape_non_generating":
            record["checks"] = check(record, contains=["SYS_SENTINEL_9f7d", "USR_SENTINEL_4c2a"], suffix="<|im_start|>assistant\n<think></think>")
        render_records.append(record)

    passes = sum(1 for record in render_records if record.get("checks", {}).get("pass"))
    diagnostics = {
        "schema_version": "nemotron-template-diagnostics-v1",
        "endpoint": BASE_URL,
        "model_alias": MODEL_ALIAS,
        "non_generating": True,
        "render_records": render_records,
        "checks_passed": passes,
        "checks_total": len(render_records),
        "overall_status": "PASS" if passes == len(render_records) and len(render_records) > 0 else "FAIL",
        "notes": [
            "The runtime chat_template is captured verbatim in server_snapshots.json; its SHA-256 is recorded in model_runtime.",
            "The history cases use assistant.reasoning_content to test llama.cpp's preserve_reasoning -> template preserve_thinking/truncate_history_thinking mapping.",
            "The JSON request case is still non-generating; response_format and generation controls are carried through the request body for parser-shape evidence only.",
        ],
    }
    write_json(RUN_DIR / "render_diagnostics.json", diagnostics)

    ps = subprocess.run(["/bin/ps", "-axo", "pid,ppid,etime,%cpu,rss,command"], capture_output=True, text=True, check=False)
    memory = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True, text=True, check=False)
    vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=False)
    write_json(
        RUN_DIR / "runtime_process_memory_snapshot.json",
        {
            "captured_at_unix": time.time(),
            "target_process_lines": [line for line in ps.stdout.splitlines() if MODEL_ALIAS in line or "llama-server --models-preset" in line],
            "memory_pressure_q": memory.stdout,
            "vm_stat": vm.stdout,
            "formal_launchd_query": subprocess.run(["/bin/launchctl", "print", "system/com.scc.llama-server"], capture_output=True, text=True, check=False).stdout[:12000],
        },
    )
    print(json.dumps({"overall_status": diagnostics["overall_status"], "checks_passed": passes, "checks_total": len(render_records), "template_sha256": model_meta["runtime_template_sha256"]}, ensure_ascii=False))
    return 0 if diagnostics["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
