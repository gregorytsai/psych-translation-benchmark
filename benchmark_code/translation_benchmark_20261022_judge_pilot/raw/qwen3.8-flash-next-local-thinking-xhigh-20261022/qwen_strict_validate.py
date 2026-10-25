#!/usr/bin/env python3
"""Strict local-run validation for the frozen Qwen thinking-on judge run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter
from typing import Any


PACKET_SHA = "38b10f7ebc6b8166bfd5581e6afb69d385a13212aea1a1d6c88b05281fb88dda"
INSTRUCTIONS_SHA = "3141f0cd8191713ab5ee79844b22b34856a87c5ddb29bb9708a1d8d6fc5d8eba"
EXPECTED_IDS = {
    *(f"qualification-q-{i:02d}" for i in range(1, 13)),
    *(f"pointwise-src-{src:02d}-cand-{cand:02d}" for src in range(1, 4) for cand in range(1, 8)),
    *(f"pairwise-p-{pair:02d}-{side}" for pair in range(1, 4) for side in ("AB", "BA")),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    args = ap.parse_args()
    raw_bytes = args.raw.read_bytes()
    d: dict[str, Any] = json.loads(raw_bytes)
    errors: list[str] = []
    warnings: list[str] = []
    events = d.get("raw_events") or []
    substantive = [e for e in events if e.get("request_id") not in {"capability-01", "request-format-02"}]
    event_ids = [e.get("request_id") for e in substantive]
    expected_capability = {"capability-01", "request-format-02"}
    capability = [e for e in events if e.get("request_id") in expected_capability]
    if d.get("packet_sha256") != PACKET_SHA:
        errors.append("top-level packet_sha256 mismatch")
    if d.get("instructions_sha256") != INSTRUCTIONS_SHA:
        errors.append("top-level instructions_sha256 mismatch")
    if not d.get("complete") or d.get("status") != "COMPLETE":
        errors.append("bundle is not COMPLETE")
    counts = d.get("counts") or {}
    if counts != {"failed": 0, "ok": 39, "pairwise": 6, "pointwise": 21, "qualification": 12, "records": 39}:
        errors.append(f"top-level counts mismatch: {counts!r}")
    if len(events) != 41 or len(substantive) != 39 or len(capability) != 2:
        errors.append(f"event cardinality mismatch: events={len(events)} substantive={len(substantive)} capability={len(capability)}")
    if set(event_ids) != EXPECTED_IDS or len(event_ids) != len(set(event_ids)):
        errors.append("substantive request-id set/uniqueness mismatch")
    settings = d.get("settings") or {}
    expected_settings = {
        "temperature": 0,
        "top_p": 1,
        "seed": 20261022,
        "strict_json_output": True,
        "response_format_requested": "json_object",
        "max_tokens_by_track": {"capability": 16384, "pairwise": 16384, "pointwise": 16384, "qualification": 16384},
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            errors.append(f"bundle settings mismatch for {key}: {settings.get(key)!r}")

    req_invariants = Counter()
    response_invariants = Counter()
    json_shapes: Counter[str] = Counter()
    for event in events:
        rid = event.get("request_id")
        req = event.get("request") or {}
        resp = event.get("response") or {}
        if event.get("status") != "ok" or event.get("http_status") != 200:
            errors.append(f"{rid}: status/http mismatch")
        if event.get("endpoint") != "http://127.0.0.1:18082/v1":
            errors.append(f"{rid}: endpoint mismatch")
        expected_request = {
            "model": "qwen3.8-flash-next-isolated",
            "temperature": 0,
            "top_p": 1,
            "seed": 20261022,
            "max_tokens": 16384,
            "response_format": {"type": "json_object"},
            "reasoning_effort": "xhigh",
            "reasoning_format": "deepseek",
            "reasoning_budget_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
            "stream": False,
        }
        for key, expected in expected_request.items():
            if req.get(key) != expected:
                errors.append(f"{rid}: request {key} mismatch: {req.get(key)!r}")
        choices = resp.get("choices") or []
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            errors.append(f"{rid}: choice/finish mismatch")
        message = choices[0].get("message") if choices else {}
        message = message or {}
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        if not reasoning:
            errors.append(f"{rid}: missing reasoning_content")
        if not content:
            errors.append(f"{rid}: missing content")
        try:
            parsed = json.loads(content)
            json_shapes[type(parsed).__name__] += 1
        except Exception as exc:  # noqa: BLE001 - retained as validation evidence
            errors.append(f"{rid}: content is not strict JSON: {exc}")
        usage = resp.get("usage") or {}
        if usage.get("completion_tokens", 0) > 16384:
            errors.append(f"{rid}: API completion exceeds max_tokens")
        req_invariants.update(
            (
                req.get("temperature"), req.get("top_p"), req.get("seed"), req.get("max_tokens"),
                req.get("reasoning_effort"), req.get("reasoning_format"), req.get("reasoning_budget_tokens"),
                json.dumps(req.get("chat_template_kwargs"), sort_keys=True),
            ),
        )
        response_invariants.update((event.get("http_status"), choices[0].get("finish_reason") if choices else None))

    audit = {
        "schema_version": "qwen-thinking-strict-validation-v1",
        "status": "PASS" if not errors else "FAIL",
        "raw_path": str(args.raw),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "packet_sha256": d.get("packet_sha256"),
        "instructions_sha256": d.get("instructions_sha256"),
        "events": {"total": len(events), "capability": len(capability), "substantive": len(substantive)},
        "request_ids": {"expected": len(EXPECTED_IDS), "unique": len(set(event_ids)), "all_expected": set(event_ids) == EXPECTED_IDS},
        "settings_exact": not any("bundle settings mismatch" in e for e in errors),
        "request_invariants_exact": not any("request " in e for e in errors),
        "response_invariants": {"status_http": dict(response_invariants), "json_shapes": dict(json_shapes)},
        "errors": errors,
        "warnings": warnings,
        "note": "Judge context was restricted to frozen packet/instructions; owner map, hidden qualification key, and prior scores were not supplied.",
    }
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": audit["status"], "errors": len(errors)}, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
