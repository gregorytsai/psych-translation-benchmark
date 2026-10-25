#!/usr/bin/env python3
"""Run exactly one local JSON chat-completion smoke and preserve its raw evidence."""

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
        result = {}
        for key, child in value.items():
            if any(word in str(key).lower() for word in ("authorization", "api_key", "apikey", "token", "secret", "password", "bearer")):
                result[key] = "<REDACTED>"
            else:
                result[key] = redact(child)
        return result
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
            {"role": "user", "content": 'Return exactly this JSON object: {"ok":true,"marker":"nemotron-local-smoke"}'},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 20261022,
        "max_tokens": 120,
        "stream": False,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False, "preserve_reasoning": False},
    }
    raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=raw_body, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    record: dict[str, Any] = {
        "schema_version": "nemotron-smoke-v1",
        "endpoint": URL,
        "request": redact(payload),
        "request_sha256": hashlib.sha256(raw_body).hexdigest(),
        "status": "failed",
    }
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            response_bytes = response.read()
            response_text = response_bytes.decode("utf-8", errors="replace")
            parsed = json.loads(response_text)
            elapsed = time.monotonic() - started
            record.update({"status": "ok", "http_status": response.status, "elapsed_s": elapsed, "response_sha256": hashlib.sha256(response_bytes).hexdigest(), "response": redact(parsed)})
            choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
            message = choices[0].get("message", {}) if choices else {}
            content = message.get("content") if isinstance(message, dict) else None
            reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                content = reasoning if isinstance(reasoning, str) else ""
            candidate = content.strip()
            try:
                decoded = json.loads(candidate)
                record["parsed_json"] = decoded
                record["json_shape_pass"] = decoded == {"ok": True, "marker": "nemotron-local-smoke"}
            except json.JSONDecodeError as exc:
                record.update({"parsed_json": None, "json_shape_pass": False, "parse_error": str(exc)})
            record["content_sha256"] = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
            record["finish_reason"] = choices[0].get("finish_reason") if choices else None
            record["usage"] = parsed.get("usage") if isinstance(parsed, dict) else None
    except urllib.error.HTTPError as exc:
        body = exc.read()
        record.update({"http_status": exc.code, "elapsed_s": time.monotonic() - started, "error": body.decode("utf-8", errors="replace")[:8000], "response_sha256": hashlib.sha256(body).hexdigest()})
    except Exception as exc:  # preserve failure as raw diagnostic
        record.update({"elapsed_s": time.monotonic() - started, "error": repr(exc)})
    record["overall_status"] = "PASS" if record.get("status") == "ok" and record.get("json_shape_pass") is True else "FAIL"
    (RUN_DIR / "smoke.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ("overall_status", "http_status", "elapsed_s", "finish_reason", "usage", "json_shape_pass", "error")}, ensure_ascii=False))
    return 0 if record["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
