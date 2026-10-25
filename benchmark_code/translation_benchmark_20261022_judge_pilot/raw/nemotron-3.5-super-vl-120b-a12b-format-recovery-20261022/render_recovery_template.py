#!/usr/bin/env python3
"""Non-generating native-template checks for the recovery server."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
EVIDENCE_NAME = os.environ.get("NEMO_TEMPLATE_EVIDENCE_NAME", "recovery_template_evidence.json")
BASE = "http://127.0.0.1:18081"
ALIAS = "nemotron-3.5-super-vl-120b-a12b-format-recovery"
EXPECTED_TEMPLATE_SHA = "39a6aba148cde4e253d6935ce035fa9c4489df867ac2509ce278894d7c4f6177"


def call(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=payload, headers=headers, method=method)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
        return {"status": "ok", "http_status": response.status, "elapsed_s": time.monotonic() - started, "request": body, "response": parsed}
    except urllib.error.HTTPError as exc:
        return {"status": "failed", "http_status": exc.code, "elapsed_s": time.monotonic() - started, "request": body, "error": exc.read().decode("utf-8", errors="replace")[:12000]}
    except Exception as exc:
        return {"status": "failed", "elapsed_s": time.monotonic() - started, "request": body, "error": repr(exc)}


def check(record: dict[str, Any], needles: list[str], suffix: str) -> dict[str, Any]:
    prompt = (record.get("response") or {}).get("prompt", "")
    return {"http_ok": record.get("status") == "ok" and record.get("http_status") == 200, "needles": {x: x in prompt for x in needles}, "suffix": {"expected": suffix, "actual": prompt[-len(suffix):], "match": prompt.endswith(suffix)}, "pass": record.get("status") == "ok" and record.get("http_status") == 200 and all(x in prompt for x in needles) and prompt.endswith(suffix)}


def run(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"args": args, "returncode": result.returncode, "stdout": result.stdout[:12000], "stderr": result.stderr[:4000]}


def main() -> int:
    snapshots = {name: call("GET", path) for name, path in {"health": "/health", "models": "/v1/models", "props": "/props", "slots": "/slots"}.items()}
    props = snapshots["props"].get("response", {})
    template = props.get("chat_template") if isinstance(props, dict) else None
    messages = [{"role": "system", "content": "SYS_RECOVERY_SENTINEL"}, {"role": "user", "content": "USR_RECOVERY_SENTINEL"}]
    cases = [
        ("thinking_on_roles", {"messages": messages, "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False}}, ["SYS_RECOVERY_SENTINEL", "USR_RECOVERY_SENTINEL"], "<|im_start|>assistant\n<think>\n"),
        ("json_high_budget_shape", {"model": ALIAS, "messages": messages, "temperature": 0, "top_p": 1, "seed": 20261022, "max_tokens": 16384, "stream": False, "response_format": {"type": "json_object"}, "reasoning_format": "deepseek", "reasoning_budget_tokens": 8192, "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False}, "add_generation_prompt": True}, ["SYS_RECOVERY_SENTINEL", "USR_RECOVERY_SENTINEL"], "<|im_start|>assistant\n<think>\n"),
    ]
    records = []
    for name, body, needles, suffix in cases:
        record = call("POST", "/apply-template", body)
        record["case"] = name
        record["checks"] = check(record, needles, suffix)
        records.append(record)
    runtime = {
        "alias": ALIAS,
        "runtime_template_length": len(template) if isinstance(template, str) else None,
        "runtime_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None,
        "expected_template_sha256": EXPECTED_TEMPLATE_SHA,
        "template_hash_matches_prior": isinstance(template, str) and hashlib.sha256(template.encode("utf-8")).hexdigest() == EXPECTED_TEMPLATE_SHA,
        "build_info": props.get("build_info") if isinstance(props, dict) else None,
        "model_path": props.get("model_path") if isinstance(props, dict) else None,
        "model_ftype": props.get("model_ftype") if isinstance(props, dict) else None,
        "chat_template_caps": props.get("chat_template_caps") if isinstance(props, dict) else None,
    }
    ps = run(["/bin/ps", "-axo", "pid,ppid,etime,%cpu,rss,command"])
    evidence = {"schema_version": "nemotron-format-recovery-template-v1", "non_generating": True, "base_url": BASE, "runtime": runtime, "snapshots": snapshots, "render_records": records, "checks_passed": sum(x["checks"]["pass"] for x in records), "checks_total": len(records), "process_snapshot": ps, "memory_pressure": run(["/usr/bin/memory_pressure", "-Q"]), "formal_health": call("GET", "http://127.0.0.1:8080/health") if False else None, "notes": ["Native template is captured from recovery /props and not modified.", "The JSON body case is non-generating and records high-budget request shape only."]}
    (RUN_DIR / EVIDENCE_NAME).write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checks_passed": evidence["checks_passed"], "checks_total": evidence["checks_total"], "template_sha256": runtime["runtime_template_sha256"], "caps": runtime["chat_template_caps"]}, ensure_ascii=False))
    return 0 if evidence["checks_passed"] == evidence["checks_total"] and runtime["template_hash_matches_prior"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
