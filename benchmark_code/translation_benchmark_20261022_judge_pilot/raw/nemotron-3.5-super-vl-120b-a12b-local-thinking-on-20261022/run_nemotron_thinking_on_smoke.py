#!/usr/bin/env python3
"""Run the one final-body thinking-on smoke gate and preserve full redacted evidence."""

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
MODEL = "nemotron-3.5-super-vl-120b-a12b-local-thinking-on"
EXPECTED = {"ok": True, "marker": "nemotron-thinking-on-smoke"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"authorization", "api_key", "apikey", "password", "secret", "access_token", "bearer"} or lowered.endswith("_token") and lowered not in {"reasoning_budget_tokens", "thinking_budget_tokens"}:
                out[key] = "<REDACTED>"
            else:
                out[key] = redact(child)
        return out
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
            {"role": "system", "content": "You are a JSON smoke-test responder. Think through the answer, then output only the requested JSON object."},
            {"role": "user", "content": 'Return exactly this JSON object: {"ok":true,"marker":"nemotron-thinking-on-smoke"}'},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 20261022,
        "max_tokens": 8192,
        "stream": False,
        "response_format": {"type": "json_object"},
        "reasoning_format": "deepseek",
        "reasoning_budget_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False},
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=encoded, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    record: dict[str, Any] = {
        "schema_version": "nemotron-thinking-on-smoke-v1",
        "endpoint": URL,
        "request": redact(payload),
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "timeout_s": 900,
        "status": "failed",
    }
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            body = response.read()
            parsed = json.loads(body.decode("utf-8", errors="replace"))
            elapsed = time.monotonic() - started
            record.update({"status": "ok", "http_status": response.status, "elapsed_s": elapsed, "response_sha256": hashlib.sha256(body).hexdigest(), "response": redact(parsed)})
            choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
            record["assistant_content"] = content
            record["reasoning_content"] = reasoning
            record["assistant_content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest() if isinstance(content, str) else None
            record["reasoning_content_sha256"] = hashlib.sha256(reasoning.encode("utf-8")).hexdigest() if isinstance(reasoning, str) else None
            record["assistant_content_chars"] = len(content) if isinstance(content, str) else None
            record["reasoning_content_chars"] = len(reasoning) if isinstance(reasoning, str) else 0
            record["reasoning_nonempty"] = isinstance(reasoning, str) and bool(reasoning.strip())
            record["finish_reason"] = choice.get("finish_reason")
            record["usage"] = parsed.get("usage") if isinstance(parsed, dict) else None
            record["timings"] = parsed.get("timings") if isinstance(parsed, dict) else None
            try:
                parsed_json = json.loads(content.strip()) if isinstance(content, str) else None
                record["parsed_json"] = parsed_json
                record["json_shape_pass"] = parsed_json == EXPECTED
            except (json.JSONDecodeError, AttributeError) as exc:
                record["parsed_json"] = None
                record["json_shape_pass"] = False
                record["parse_error"] = str(exc)
            usage = record.get("usage") or {}
            record["truncation"] = record.get("finish_reason") == "length" or (isinstance(usage, dict) and usage.get("completion_tokens") == payload["max_tokens"])
    except urllib.error.HTTPError as exc:
        body = exc.read()
        record.update({"http_status": exc.code, "elapsed_s": time.monotonic() - started, "error": body.decode("utf-8", errors="replace")[:12000], "response_sha256": hashlib.sha256(body).hexdigest()})
    except Exception as exc:
        record.update({"elapsed_s": time.monotonic() - started, "error": repr(exc)})

    record["overall_status"] = "PASS" if (
        record.get("status") == "ok"
        and record.get("http_status") == 200
        and record.get("reasoning_nonempty") is True
        and record.get("json_shape_pass") is True
        and record.get("finish_reason") == "stop"
        and record.get("truncation") is False
    ) else "FAIL"
    out = RUN_DIR / "thinking_on_smoke.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ("overall_status", "http_status", "elapsed_s", "finish_reason", "reasoning_nonempty", "reasoning_content_chars", "usage", "timings", "json_shape_pass", "truncation", "error")}, ensure_ascii=False))
    return 0 if record["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
