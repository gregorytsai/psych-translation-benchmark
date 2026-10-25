#!/usr/bin/env python3
"""Tokenize retained reasoning_content for every generated event, no generation."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
RAW = RUN_DIR / "nemotron-3.5-super-vl-120b-a12b-local-thinking-on.raw.json"
URL = "http://127.0.0.1:18081/tokenize"
BUDGET = 4096


def tokenize(text: str) -> dict[str, Any]:
    body = json.dumps({"content": text, "add_special": False, "parse_special": True}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        tokens = parsed.get("tokens", []) if isinstance(parsed, dict) else []
        return {"status": "ok", "http_status": response.status, "elapsed_s": time.monotonic() - started, "token_count": len(tokens) if isinstance(tokens, list) else None}
    except urllib.error.HTTPError as exc:
        return {"status": "failed", "http_status": exc.code, "elapsed_s": time.monotonic() - started, "error": exc.read().decode("utf-8", errors="replace")[:1000]}
    except Exception as exc:
        return {"status": "failed", "elapsed_s": time.monotonic() - started, "error": repr(exc)}


def main() -> int:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for event in raw.get("raw_events", []):
        response = event.get("response") or {}
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        base = {
            "request_id": event.get("request_id"),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None,
            "reasoning_sha256": hashlib.sha256(reasoning.encode("utf-8")).hexdigest() if isinstance(reasoning, str) else None,
            "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
            "total_tokens": (response.get("usage") or {}).get("total_tokens"),
        }
        if not isinstance(reasoning, str):
            base.update({"status": "missing_reasoning_content", "token_count": None})
        else:
            base.update(tokenize(reasoning))
            base["budget"] = BUDGET
            base["budget_hit_exact"] = base.get("token_count") == BUDGET
            base["budget_at_or_over"] = isinstance(base.get("token_count"), int) and base["token_count"] >= BUDGET
        samples.append(base)
    counts = {"events": len(samples), "tokenize_ok": sum(x.get("status") == "ok" for x in samples), "tokenize_failed": sum(x.get("status") != "ok" for x in samples), "budget_hit_exact": sum(x.get("budget_hit_exact") is True for x in samples), "budget_at_or_over": sum(x.get("budget_at_or_over") is True for x in samples)}
    output = {"schema_version": "nemotron-reasoning-budget-tokenize-audit-v1", "endpoint": URL, "budget": BUDGET, "non_generating": True, "raw_bundle": str(RAW), "counts": counts, "samples": samples, "interpretation": "token_count measures retained reasoning_content only; total/completion usage includes the final JSON and must not be compared directly with the reasoning budget. Exact budget hits are force-end candidates under the local sampler source semantics."}
    (RUN_DIR / "reasoning_budget_tokenize_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "exact_hits": [x["request_id"] for x in samples if x.get("budget_hit_exact") is True]}, ensure_ascii=False))
    return 0 if counts["tokenize_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
