#!/usr/bin/env python3
"""Run exactly one blind format-recovery pointwise request.

This is deliberately separate from the uniform 39-request runner: the only
target is the original packet's src-03/cand-02 row, and no prior score,
subscore, error, answer key, or candidate mapping is placed in the prompt.
The packet and instruction hashes remain those of the frozen parent files.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PARENT_DIR = ROOT / "translation_benchmark_20261022_judge_pilot"
PACKET_PATH = PARENT_DIR / "judge_packet.json"
INSTRUCTIONS_PATH = PARENT_DIR / "judge_instructions.txt"
URL = "http://127.0.0.1:18081/v1/chat/completions"
TOKENIZE_URL = "http://127.0.0.1:18081/tokenize"
MODEL = "nemotron-3.5-super-vl-120b-a12b-format-recovery"
JUDGE_ID = "nemotron-3.5-super-vl-120b-format-recovery-20261022"
ATTEMPT_TAG = os.environ.get("NEMO_RECOVERY_ATTEMPT_TAG", "base")
RUN_ID = "nemotron-3.5-super-vl-120b-format-recovery-20261022-run1" if ATTEMPT_TAG == "base" else f"nemotron-3.5-super-vl-120b-format-recovery-20261022-{ATTEMPT_TAG}"
REQUEST_ID = "pointwise-src-03-cand-02"
SEED = 20261022
REASONING_BUDGET = 8192
MAX_TOKENS = 16384
TIMEOUT_S = 1800.0

sys.path.insert(0, str(PARENT_DIR))
from run_live_judge import (  # noqa: E402
    build_request_messages,
    deterministic,
    normalized_errors,
    redact,
    score_shape,
    sha256_text,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_target(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for item in packet.get("pointwise", {}).get("items", []):
        if item.get("item_id") != "src-03":
            continue
        for candidate in item.get("candidates", []):
            if candidate.get("blind_id") == "cand-02":
                return item, candidate
    raise RuntimeError("frozen packet target src-03/cand-02 was not found")


def tokenize_reasoning(reasoning: str | None) -> dict[str, Any]:
    """Count retained reasoning tokens through llama.cpp; this does not generate."""

    result: dict[str, Any] = {
        "status": "missing_reasoning_content" if not isinstance(reasoning, str) else "failed",
        "endpoint": TOKENIZE_URL,
        "budget": REASONING_BUDGET,
        "token_count": None,
    }
    if not isinstance(reasoning, str):
        return result
    encoded = json.dumps({"content": reasoning, "add_special": False, "parse_special": True}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(TOKENIZE_URL, data=encoded, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
            parsed = json.loads(body.decode("utf-8"))
        tokens = parsed.get("tokens", []) if isinstance(parsed, dict) else []
        result.update({"status": "ok", "http_status": response.status, "elapsed_s": time.monotonic() - started, "token_count": len(tokens) if isinstance(tokens, list) else None})
    except urllib.error.HTTPError as exc:
        result.update({"status": "failed", "http_status": exc.code, "elapsed_s": time.monotonic() - started, "error": redact(exc.read().decode("utf-8", errors="replace")[:1000])})
    except Exception as exc:  # noqa: BLE001 - preserve diagnostic evidence
        result.update({"status": "failed", "elapsed_s": time.monotonic() - started, "error": redact(repr(exc))})
    count = result.get("token_count")
    result["budget_hit_exact"] = count == REASONING_BUDGET
    result["budget_at_or_over"] = isinstance(count, int) and count >= REASONING_BUDGET
    return result


def main() -> int:
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    instructions = INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    item, candidate = find_target(packet)
    source = item["source_en"]
    target = candidate["target_zh_tw"]
    packet_sha = hashlib.sha256(PACKET_PATH.read_bytes()).hexdigest()
    instructions_sha = sha256_text(instructions)

    body = "Evaluate this one anonymous candidate and return the pointwise JSON contract.\n\n" + f"{candidate['blind_id']}:\n{target}"
    messages = build_request_messages(instructions, "pointwise", source, body)
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_object"},
        "reasoning_format": "deepseek",
        "reasoning_budget_tokens": REASONING_BUDGET,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False},
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    event: dict[str, Any] = {
        "request_id": REQUEST_ID,
        "endpoint": URL,
        "auth_env": None,
        "auth_present": False,
        "status": "failed",
        "request": redact(payload),
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    response_data: dict[str, Any] | None = None
    started = time.monotonic()
    request = urllib.request.Request(URL, data=encoded, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw_body = response.read()
            response_data = json.loads(raw_body.decode("utf-8"))
        event.update({"status": "ok", "http_status": response.status, "elapsed_s": time.monotonic() - started, "response": redact(response_data), "response_sha256": hashlib.sha256(raw_body).hexdigest()})
    except urllib.error.HTTPError as exc:
        raw_body = exc.read()
        event.update({"status": "failed", "http_status": exc.code, "elapsed_s": time.monotonic() - started, "error": redact(raw_body.decode("utf-8", errors="replace")[:12000]), "response_sha256": hashlib.sha256(raw_body).hexdigest()})
    except Exception as exc:  # noqa: BLE001 - preserve provider/runtime errors as data
        event.update({"status": "failed", "elapsed_s": time.monotonic() - started, "error": redact(repr(exc))})

    choice: dict[str, Any] = {}
    message: dict[str, Any] = {}
    content: str | None = None
    reasoning: str | None = None
    parsed: Any = None
    parse_error: str | None = None
    shape_pass = False
    shape_reason = "no HTTP response"
    if isinstance(response_data, dict):
        choices = response_data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            raw_content = message.get("content")
            if isinstance(raw_content, list):
                content = "".join(part.get("text", "") for part in raw_content if isinstance(part, dict))
            elif isinstance(raw_content, str):
                content = raw_content
            reasoning = message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else None
            if isinstance(content, str) and content.strip():
                try:
                    parsed = json.loads(content.strip())
                    if not isinstance(parsed, dict):
                        shape_reason = "strict JSON value is not an object"
                    elif parsed.get("blind_id") not in {None, candidate["blind_id"]}:
                        shape_reason = "assistant blind_id disagrees with task"
                    else:
                        parsed["blind_id"] = candidate["blind_id"]
                        shape_pass, shape_reason = score_shape(parsed)
                except json.JSONDecodeError as exc:
                    parse_error = f"strict JSON parse failed: {exc}"
                    shape_reason = parse_error
            else:
                shape_reason = "assistant content is missing; reasoning is not accepted as JSON output"
        else:
            shape_reason = "response has no choices"

    usage = response_data.get("usage") if isinstance(response_data, dict) else None
    finish_reason = choice.get("finish_reason")
    reasoning_audit = tokenize_reasoning(reasoning)
    event["response_metadata"] = {
        "finish_reason": finish_reason,
        "usage": redact(usage),
        "timings": redact(response_data.get("timings") if isinstance(response_data, dict) else None),
        "reasoning_content_present": isinstance(reasoning, str) and bool(reasoning.strip()),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "reasoning_sha256": sha256_text(reasoning) if isinstance(reasoning, str) else None,
        "reasoning_token_audit": reasoning_audit,
        "strict_json_pass": shape_pass,
        "score_shape_reason": shape_reason,
        "truncation": finish_reason == "length" or reasoning_audit.get("budget_hit_exact") is True,
    }

    if shape_pass and isinstance(parsed, dict):
        pointwise = dict(parsed)
        if pointwise.get("abstain") is not True:
            pointwise["errors"] = normalized_errors(pointwise.get("errors", []), target)
        row: dict[str, Any] = {
            "status": "ok",
            "track": "pointwise",
            "run_id": RUN_ID,
            "item_id": item["item_id"],
            "source_id": item["item_id"],
            "source_text_sha256": sha256_text(source),
            "judge_id": JUDGE_ID,
            "family": "nvidia-nemotron",
            "candidate_id": candidate["blind_id"],
            "candidate": {"blind_id": candidate["blind_id"], "raw_translation_zh_tw": target, "target_text_sha256": sha256_text(target)},
            "item": {"item_id": item["item_id"], "domain": item.get("domain"), "input_type": item.get("input_type"), "source_en": source, "target_locale": item.get("target_locale")},
            "deterministic": deterministic(source, target),
            "pointwise": pointwise,
            "timings": {"api_elapsed_s": event.get("elapsed_s"), "timeout": False, "finish_reason": finish_reason, "usage": usage, "server_timings": response_data.get("timings") if isinstance(response_data, dict) else None, "reasoning": {"content_present": isinstance(reasoning, str) and bool(reasoning.strip()), "chars": len(reasoning) if isinstance(reasoning, str) else 0, "sha256": sha256_text(reasoning) if isinstance(reasoning, str) else None, "token_audit": reasoning_audit}},
        }
    else:
        row = {
            "status": "failed",
            "track": "pointwise",
            "run_id": RUN_ID,
            "item_id": item["item_id"],
            "source_id": item["item_id"],
            "source_text_sha256": sha256_text(source),
            "judge_id": JUDGE_ID,
            "family": "nvidia-nemotron",
            "candidate_id": candidate["blind_id"],
            "reason": shape_reason,
            "semantic_json_valid": shape_pass,
            "http_status": event.get("http_status"),
            "finish_reason": finish_reason,
        }

    record = {
        "schema_version": "nemotron-format-recovery-case-v1",
        "status": "COMPLETE" if event.get("status") == "ok" else "FAILED",
        "complete": True,
        "packet_id": packet.get("packet_id"),
        "packet_sha256": packet_sha,
        "instructions_sha256": instructions_sha,
        "run_id": RUN_ID,
        "judge_id": JUDGE_ID,
        "request_id": REQUEST_ID,
        "request_mode": {"kind": "single_pointwise_format_recovery", "native_template": True, "reasoning_format": "deepseek", "enable_thinking": True, "preserve_reasoning": False, "response_format": "json_object", "strict_json_validation": True, "no_retry": True},
        "source": {"item_id": item["item_id"], "source_en": source, "source_text_sha256": sha256_text(source)},
        "candidate": {"blind_id": candidate["blind_id"], "target_zh_tw": target, "target_text_sha256": sha256_text(target)},
        "settings": {"temperature": 0, "top_p": 1, "seed": SEED, "max_tokens": MAX_TOKENS, "reasoning_budget_tokens": REASONING_BUDGET, "context_size": 32768, "timeout_s": TIMEOUT_S, "model": MODEL, "base_url": URL.rsplit("/v1", 1)[0] + "/v1", "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False}},
        "record": row,
        "raw_event": event,
        "metrics": {"http_status": event.get("http_status"), "api_elapsed_s": event.get("elapsed_s"), "finish_reason": finish_reason, "usage": usage, "server_timings": response_data.get("timings") if isinstance(response_data, dict) else None, "reasoning_content_present": isinstance(reasoning, str) and bool(reasoning.strip()), "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0, "reasoning_sha256": sha256_text(reasoning) if isinstance(reasoning, str) else None, "reasoning_token_audit": reasoning_audit, "strict_json_pass": shape_pass, "score_shape_reason": shape_reason, "truncation": finish_reason == "length" or reasoning_audit.get("budget_hit_exact") is True},
        "parent_run_composition": {"prior_valid_pointwise_rows": 20, "recovery_row_if_valid": 1, "not_a_sixth_judge": True, "not_a_uniform_39_request_run": True, "source_parent_raw_immutable": True},
    }
    case_name = "recovery_case.json" if ATTEMPT_TAG == "base" else f"recovery_case_{ATTEMPT_TAG}.json"
    audit_name = "recovery_reasoning_tokenize_audit.json" if ATTEMPT_TAG == "base" else f"recovery_reasoning_tokenize_audit_{ATTEMPT_TAG}.json"
    write_json(RUN_DIR / case_name, record)
    write_json(RUN_DIR / audit_name, {"schema_version": "nemotron-recovery-reasoning-tokenize-v1", "non_generating": True, "request_id": REQUEST_ID, "budget": REASONING_BUDGET, "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0, "reasoning_sha256": sha256_text(reasoning) if isinstance(reasoning, str) else None, "audit": reasoning_audit})
    print(json.dumps({"status": record["status"], "http_status": event.get("http_status"), "elapsed_s": event.get("elapsed_s"), "finish_reason": finish_reason, "strict_json_pass": shape_pass, "score_shape_reason": shape_reason, "reasoning_content_present": record["metrics"]["reasoning_content_present"], "reasoning_token_count": reasoning_audit.get("token_count"), "budget_hit_exact": reasoning_audit.get("budget_hit_exact")}, ensure_ascii=False))
    return 0 if shape_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
