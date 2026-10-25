#!/usr/bin/env python3
"""Capture non-generating Mistral /props and /apply-template provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "template_audit_20261023"
BASE_URL = "http://127.0.0.1:18080"
PID_FILE = ROOT / "raw/mistral-medium-3.5-128b-local-thinking-high-targeted-20261023/server.pid"
COMMAND_FILE = ROOT / "raw/mistral-medium-3.5-128b-local-thinking-high-targeted-20261023/server.command.txt"
MODEL = "/benchmark/Mistral-Medium-3.5-128B/Mistral-Medium-3.5-128B-UD-Q4_K_XL-00001-of-00003.gguf"
ALIAS = "mistral-medium-3.5-128b-targeted-thinking-high-20261023"
BUILD = "b524-620d2225c"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def now_stamps() -> dict[str, str]:
    current = datetime.now(timezone.utc)
    return {
        "utc": current.isoformat().replace("+00:00", "Z"),
        "taipei": current.astimezone(ZoneInfo("Asia/Taipei")).isoformat(),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def lsof_snapshot() -> str:
    proc = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-iTCP:18080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.stdout[:12000]


def call(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_request = b"" if body is None else json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=None if body is None else raw_request,
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if body is not None else {})},
        method=method,
    )
    started = time.monotonic()
    record: dict[str, Any] = {
        "method": method,
        "endpoint": BASE_URL + path,
        "path": path,
        "timestamp": now_stamps(),
        "request_json": body,
        "request_sha256": sha256_bytes(raw_request),
        "request_bytes": len(raw_request),
        "command_class": f"{method} {path} (non-generating endpoint)",
        "status": "failed",
    }
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_response = response.read()
            record.update(
                {
                    "status": "ok",
                    "http_status": response.status,
                    "elapsed_s": time.monotonic() - started,
                    "response_sha256": sha256_bytes(raw_response),
                    "response_bytes": len(raw_response),
                }
            )
        try:
            parsed = json.loads(raw_response.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            parsed = None
        record["response_json"] = parsed
        if path == "/apply-template" and isinstance(parsed, dict):
            prompt = parsed.get("prompt")
            if isinstance(prompt, str):
                record["prompt_chars"] = len(prompt)
                record["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return record
    except urllib.error.HTTPError as exc:
        raw_response = exc.read()
        record.update(
            {
                "http_status": exc.code,
                "elapsed_s": time.monotonic() - started,
                "response_sha256": sha256_bytes(raw_response),
                "response_bytes": len(raw_response),
                "response_text": raw_response.decode("utf-8", errors="replace")[:12000],
            }
        )
        return record
    except Exception as exc:  # noqa: BLE001 - retain endpoint diagnostics
        record.update({"elapsed_s": time.monotonic() - started, "error": repr(exc)})
        return record


def messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def common_body(system: str, user: str, effort: str, *, preserve: bool = False, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": ALIAS,
        "messages": messages(system, user),
        "reasoning_effort": effort,
        "reasoning_format": "deepseek",
        "reasoning_budget_tokens": 8192,
        "chat_template_kwargs": {"preserve_reasoning": preserve},
        "temperature": 0,
        "top_p": 1,
        "seed": 20261022,
        "max_tokens": 16384,
        "stream": False,
        "response_format": {"type": "json_object"},
        "add_generation_prompt": True,
    }
    body.update(extra)
    return body


def render_checks() -> list[dict[str, Any]]:
    cases: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "system_user_none",
            common_body("SYS_TEMPLATE_TARGETED_NONE", "USR_TEMPLATE_TARGETED_NONE", "none"),
            {
                "must": ["SYS_TEMPLATE_TARGETED_NONE", "USR_TEMPLATE_TARGETED_NONE", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]", '[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]'],
                "must_not": ["[THINK]", "reasoning_effort=high"],
            },
        ),
        (
            "system_user_high",
            common_body("SYS_TEMPLATE_TARGETED_HIGH", "USR_TEMPLATE_TARGETED_HIGH", "high"),
            {
                "must": ["SYS_TEMPLATE_TARGETED_HIGH", "USR_TEMPLATE_TARGETED_HIGH", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]", '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]'],
                "must_not": ["[THINK]", '"reasoning_effort": "none"'],
            },
        ),
        (
            "assistant_history_high",
            {
                **common_body("SYS_TEMPLATE_TARGETED_HISTORY", "USR_TEMPLATE_TARGETED_FINAL", "high", preserve=True),
                "messages": [
                    {"role": "system", "content": "SYS_TEMPLATE_TARGETED_HISTORY"},
                    {"role": "user", "content": "USR_TEMPLATE_TARGETED_FIRST"},
                    {"role": "assistant", "content": "AST_TEMPLATE_TARGETED_HISTORY", "reasoning_content": "HIST_REASONING_TARGETED"},
                    {"role": "user", "content": "USR_TEMPLATE_TARGETED_FINAL"},
                ],
            },
            {
                "must": ["SYS_TEMPLATE_TARGETED_HISTORY", "USR_TEMPLATE_TARGETED_FIRST", "AST_TEMPLATE_TARGETED_HISTORY", "USR_TEMPLATE_TARGETED_FINAL", "HIST_REASONING_TARGETED", "[THINK]HIST_REASONING_TARGETED[/THINK]", "</s>", '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]'],
                "must_not": [],
            },
        ),
        (
            "tool_high",
            common_body(
                "SYS_TEMPLATE_TARGETED_TOOL",
                "USR_TEMPLATE_TARGETED_TOOL",
                "high",
                tools=[{"type": "function", "function": {"name": "lookup_targeted", "description": "TOOL_DESCRIPTION_TARGETED", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}}],
                tool_choice="auto",
            ),
            {
                "must": ["SYS_TEMPLATE_TARGETED_TOOL", "USR_TEMPLATE_TARGETED_TOOL", "TOOL_DESCRIPTION_TARGETED", "[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]", '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]'],
                "must_not": ["[THINK]"],
            },
        ),
        (
            "actual_high_request_shape",
            common_body("SYS_TEMPLATE_TARGETED_BODY", "USR_TEMPLATE_TARGETED_BODY", "high"),
            {
                "must": ["SYS_TEMPLATE_TARGETED_BODY", "USR_TEMPLATE_TARGETED_BODY", '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]'],
                "must_not": ["[THINK]"],
            },
        ),
    ]
    records: list[dict[str, Any]] = []
    for name, body, checks in cases:
        record = call("POST", "/apply-template", body)
        response = record.get("response_json")
        prompt = response.get("prompt") if isinstance(response, dict) else None
        pass_http = record.get("status") == "ok" and record.get("http_status") == 200
        prompt_only = isinstance(response, dict) and set(response) == {"prompt"} and isinstance(prompt, str)
        must = {needle: isinstance(prompt, str) and prompt.count(needle) for needle in checks["must"]}
        must_not = {needle: isinstance(prompt, str) and prompt.count(needle) for needle in checks["must_not"]}
        record.update(
            {
                "case": name,
                "documented_non_generating": True,
                "checks": {
                    "http_200": pass_http,
                    "prompt_only_response": prompt_only,
                    "must_counts": must,
                    "must_not_counts": must_not,
                    "pass": pass_http and prompt_only and all(value == 1 for value in must.values()) and all(value == 0 for value in must_not.values()),
                },
            }
        )
        records.append(record)
    return records


def main() -> int:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = {name: call("GET", path) for name, path in {"health": "/health", "models": "/v1/models", "props": "/props", "slots": "/slots"}.items()}
    props = snapshots["props"].get("response_json") if isinstance(snapshots["props"], dict) else {}
    props = props if isinstance(props, dict) else {}
    template = props.get("chat_template") if isinstance(props.get("chat_template"), str) else ""
    pid = PID_FILE.read_text(encoding="utf-8").strip() if PID_FILE.exists() else None
    command = COMMAND_FILE.read_text(encoding="utf-8") if COMMAND_FILE.exists() else ""
    records = render_checks()
    for name, snapshot in snapshots.items():
        write_json(AUDIT_DIR / f"{name}.json", snapshot)
    write_json(AUDIT_DIR / "render_records.json", records)
    evidence = {
        "schema_version": "mistral-targeted-template-provenance-v1",
        "status": "PASS_NON_GENERATING_AUDIT" if all(record.get("checks", {}).get("pass") for record in records) and snapshots["props"].get("status") == "ok" else "FAIL",
        "non_generating": True,
        "generation_endpoint_used": False,
        "base_url": BASE_URL,
        "endpoint_contract": {"render_endpoint": "/apply-template", "completion_endpoints_used": [], "documented_behavior": "prompt-only formatting; no inference"},
        "server_binding": {"pid": pid, "port": 18080, "model_path": props.get("model_path"), "model_alias": props.get("model_alias"), "build_info": props.get("build_info"), "command_file": str(COMMAND_FILE), "command_sha256": sha256_bytes(command.encode("utf-8")), "listener_snapshot": lsof_snapshot()},
        "props": {"model_alias": props.get("model_alias"), "model_path": props.get("model_path"), "build_info": props.get("build_info"), "default_context": (props.get("default_generation_settings") or {}).get("n_ctx"), "effective_template_length": len(template), "effective_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(), "chat_template_caps": props.get("chat_template_caps"), "reasoning_effort_supported": (props.get("chat_template_caps") or {}).get("supports_reasoning_effort"), "preserve_reasoning_supported": (props.get("chat_template_caps") or {}).get("supports_preserve_reasoning")},
        "render_records": records,
        "control_semantics": {"native_template_selector": "reasoning_effort", "runtime_parser_fields": ["reasoning_format=deepseek", "reasoning_budget_tokens=8192"], "qwen_enable_thinking_sent": False, "reasoning_mode_sent": False, "high_model_settings_occurrences": [record.get("checks", {}).get("must_counts", {}).get('[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]', 0) for record in records]},
        "notes": ["All render bodies use synthetic markers only.", "The /apply-template responses are prompt-only and are not generation evidence.", "The new audit is not retroactive evidence for the completed 39-request raw bundle."],
    }
    write_json(AUDIT_DIR / "template_provenance.json", evidence)
    print(json.dumps({"status": evidence["status"], "checks_passed": sum(record.get("checks", {}).get("pass", False) for record in records), "checks_total": len(records), "template_sha256": evidence["props"]["effective_template_sha256"], "context": evidence["props"]["default_context"]}, ensure_ascii=False))
    return 0 if evidence["status"] == "PASS_NON_GENERATING_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
