#!/usr/bin/env python3
"""Run one request-format compatibility smoke with llama.cpp reasoning parsing."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
URL = "http://127.0.0.1:18081/v1/chat/completions"
MODEL = "nemotron-3.5-super-vl-120b-a12b-local-isolated"


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "<REDACTED>" if any(word in str(key).lower() for word in ("authorization", "api_key", "apikey", "secret", "password", "bearer")) else redact(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <REDACTED>", value)
        return re.sub(r"(?i)\bsk-[A-Za-z0-9._-]{12,}", "sk-<REDACTED>", value)
    return value


def main() -> int:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a JSON smoke-test responder. Output only the requested JSON object."},
            {"role": "user", "content": 'Return exactly this JSON object: {"ok":true,"marker":"nemotron-local-smoke-compat"}'},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 20261022,
        "max_tokens": 120,
        "stream": False,
        "response_format": {"type": "json_object"},
        "reasoning_format": "deepseek",
        "chat_template_kwargs": {"enable_thinking": False, "preserve_reasoning": False},
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=encoded, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    record: dict[str, Any] = {"schema_version": "nemotron-smoke-compat-v1", "endpoint": URL, "request": redact(payload), "request_sha256": hashlib.sha256(encoded).hexdigest(), "status": "failed"}
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            body = response.read()
            parsed = json.loads(body.decode("utf-8", errors="replace"))
            elapsed = time.monotonic() - started
            record.update({"status": "ok", "http_status": response.status, "elapsed_s": elapsed, "response_sha256": hashlib.sha256(body).hexdigest(), "response": redact(parsed)})
            choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                content = message.get("reasoning_content", "") if isinstance(message, dict) else ""
            record["assistant_content"] = content
            try:
                record["parsed_json"] = json.loads(content.strip())
                record["json_shape_pass"] = record["parsed_json"] == {"ok": True, "marker": "nemotron-local-smoke-compat"}
            except json.JSONDecodeError as exc:
                record.update({"parsed_json": None, "json_shape_pass": False, "parse_error": str(exc)})
            record.update({"finish_reason": choices[0].get("finish_reason") if choices else None, "usage": parsed.get("usage") if isinstance(parsed, dict) else None, "timings": parsed.get("timings") if isinstance(parsed, dict) else None})
    except urllib.error.HTTPError as exc:
        body = exc.read()
        record.update({"http_status": exc.code, "elapsed_s": time.monotonic() - started, "error": body.decode("utf-8", errors="replace")[:8000], "response_sha256": hashlib.sha256(body).hexdigest()})
    except Exception as exc:
        record.update({"elapsed_s": time.monotonic() - started, "error": repr(exc)})
    record["overall_status"] = "PASS" if record.get("status") == "ok" and record.get("json_shape_pass") is True else "FAIL"
    (RUN_DIR / "smoke_compat_reasoning_format_deepseek.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ("overall_status", "http_status", "elapsed_s", "finish_reason", "usage", "timings", "json_shape_pass", "error")}, ensure_ascii=False))
    return 0 if record["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
