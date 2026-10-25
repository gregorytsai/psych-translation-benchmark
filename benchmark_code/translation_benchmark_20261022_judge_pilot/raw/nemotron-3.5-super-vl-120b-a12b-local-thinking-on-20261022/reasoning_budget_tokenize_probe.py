#!/usr/bin/env python3
"""Non-generating tokenizer probe for retained reasoning strings (no score changes)."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
CHECKPOINT = RUN_DIR / "nemotron-3.5-super-vl-120b-a12b-local-thinking-on.checkpoint.json"
URL = "http://127.0.0.1:18081/tokenize"
BUDGET = 4096


def call(text: str) -> dict[str, Any]:
    body = json.dumps({"content": text, "add_special": False, "parse_special": True}, ensure_ascii=False).encode("utf-8")
    started = time.monotonic()
    request = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    tokens = parsed.get("tokens", []) if isinstance(parsed, dict) else []
    return {
        "http_status": response.status,
        "elapsed_s": time.monotonic() - started,
        "reasoning_chars": len(text),
        "reasoning_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": len(tokens) if isinstance(tokens, list) else None,
        "budget": BUDGET,
        "token_count_at_or_over_budget": isinstance(tokens, list) and len(tokens) >= BUDGET,
    }


def main() -> int:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    wanted = {"qualification-q-07", "pointwise-src-02-cand-06"}
    samples: list[dict[str, Any]] = []
    for event in checkpoint.get("raw_events", []):
        if event.get("request_id") not in wanted:
            continue
        response = event.get("response") or {}
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        if not isinstance(reasoning, str):
            samples.append({"request_id": event.get("request_id"), "status": "missing_reasoning_content"})
            continue
        record = call(reasoning)
        record.update({"request_id": event.get("request_id"), "finish_reason": choices[0].get("finish_reason") if choices else None, "completion_tokens": (response.get("usage") or {}).get("completion_tokens")})
        record["status"] = "ok"
        samples.append(record)
    output = {"schema_version": "nemotron-reasoning-budget-tokenize-probe-v1", "endpoint": URL, "budget": BUDGET, "non_generating": True, "samples": samples, "interpretation": "token_count measures retained reasoning_content only; completion_tokens includes reasoning plus final JSON and is not itself a budget-hit signal"}
    (RUN_DIR / "reasoning_budget_tokenize_probe.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"samples": [{k: v for k, v in sample.items() if k not in {"reasoning_sha256"}} for sample in samples]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
