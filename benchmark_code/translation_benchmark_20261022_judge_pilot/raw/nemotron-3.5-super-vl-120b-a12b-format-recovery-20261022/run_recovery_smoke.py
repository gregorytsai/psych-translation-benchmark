#!/usr/bin/env python3
"""One non-judge high-budget smoke for the recovery server; not a scored case."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
SMOKE_NAME = os.environ.get("NEMO_SMOKE_NAME", "recovery_smoke.json")
URL = "http://127.0.0.1:18081/v1/chat/completions"
MODEL = "nemotron-3.5-super-vl-120b-a12b-format-recovery"
EXPECTED = {"ok": True, "marker": "nemotron-format-recovery-smoke"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "<REDACTED>" if str(key).lower() in {"authorization", "api_key", "apikey", "password", "secret", "access_token", "bearer"} else redact(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <REDACTED>", value)
        return re.sub(r"(?i)\bsk-[A-Za-z0-9._-]{12,}", "sk-<REDACTED>", value)
    return value


def main() -> int:
    payload = {"model": MODEL, "messages": [{"role": "system", "content": "You are a JSON smoke-test responder. Think through the answer, then output only the requested JSON object."}, {"role": "user", "content": 'Return exactly this JSON object: {"ok":true,"marker":"nemotron-format-recovery-smoke"}'}], "temperature": 0, "top_p": 1, "seed": 20261022, "max_tokens": 16384, "stream": False, "response_format": {"type": "json_object"}, "reasoning_format": "deepseek", "reasoning_budget_tokens": 8192, "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False}}
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=encoded, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    record: dict[str, Any] = {"schema_version": "nemotron-format-recovery-smoke-v1", "endpoint": URL, "request": redact(payload), "request_sha256": hashlib.sha256(encoded).hexdigest(), "timeout_s": 1800, "status": "failed"}
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            body = response.read()
            parsed = json.loads(body.decode("utf-8", errors="replace"))
        record.update({"status": "ok", "http_status": response.status, "elapsed_s": time.monotonic() - started, "response_sha256": hashlib.sha256(body).hexdigest(), "response": redact(parsed)})
        choice = (parsed.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        record.update({"assistant_content": content, "reasoning_content": reasoning, "reasoning_nonempty": isinstance(reasoning, str) and bool(reasoning.strip()), "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0, "finish_reason": choice.get("finish_reason"), "usage": parsed.get("usage"), "timings": parsed.get("timings")})
        try:
            record["parsed_json"] = json.loads(content.strip()) if isinstance(content, str) else None
            record["json_shape_pass"] = record["parsed_json"] == EXPECTED
        except (json.JSONDecodeError, AttributeError) as exc:
            record.update({"parsed_json": None, "json_shape_pass": False, "parse_error": str(exc)})
        record["truncation"] = choice.get("finish_reason") == "length" or (parsed.get("usage") or {}).get("completion_tokens") == payload["max_tokens"]
    except urllib.error.HTTPError as exc:
        body = exc.read()
        record.update({"http_status": exc.code, "elapsed_s": time.monotonic() - started, "error": body.decode("utf-8", errors="replace")[:12000], "response_sha256": hashlib.sha256(body).hexdigest()})
    except Exception as exc:
        record.update({"elapsed_s": time.monotonic() - started, "error": repr(exc)})
    record["overall_status"] = "PASS" if record.get("status") == "ok" and record.get("http_status") == 200 and record.get("reasoning_nonempty") is True and record.get("json_shape_pass") is True and record.get("finish_reason") == "stop" and record.get("truncation") is False else "FAIL"
    (RUN_DIR / SMOKE_NAME).write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in ("overall_status", "http_status", "elapsed_s", "reasoning_nonempty", "reasoning_chars", "finish_reason", "usage", "timings", "json_shape_pass", "truncation", "error")}, ensure_ascii=False))
    return 0 if record["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
