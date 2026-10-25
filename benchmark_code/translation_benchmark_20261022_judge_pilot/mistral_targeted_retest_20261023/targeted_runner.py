#!/usr/bin/env python3
"""Run only the root-approved Mistral targeted 12-request diagnostic set.

This imports request construction and validation helpers from the frozen
run_live_judge.py but never edits or invokes that runner's main(). The selected
keys are explicit so no prefix-based max-count selection can silently change
the registered subset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
COMMON = ROOT / "run_live_judge.py"
PACKET = ROOT / "judge_packet.json"
INSTRUCTIONS = ROOT / "judge_instructions.txt"
SEED = 20261022
TIMEOUT_S = 1800.0
MAX_TOKENS = 16384
REASONING_BUDGET = 8192
JUDGE_ID = "mistral-medium-3.5-128b-local-thinking-high-targeted-20261023"
FAMILY = "mistral"
MODEL = "mistral-medium-3.5-128b-targeted-thinking-high-20261023"
TARGETED_PREFIX = "targeted-20261023-"
QUALIFICATION_KEYS = ("q-01", "q-06", "q-12")
POINTWISE_KEYS = (("src-01", "cand-01"), ("src-02", "cand-04"), ("src-03", "cand-02"))
PAIRWISE_KEYS = (("p-01", "AB"), ("p-01", "BA"), ("p-03", "AB"), ("p-03", "BA"), ("p-02", "AB"), ("p-02", "BA"))


def load_common() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_live_judge", COMMON)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen runner: {COMMON}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = load_common()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_items(packet: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    qualifications = [item for item in packet["qualification"]["items"] if item.get("qualification_id") in QUALIFICATION_KEYS]
    qualifications.sort(key=lambda item: QUALIFICATION_KEYS.index(item["qualification_id"]))
    pointwise: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item_id, candidate_id in POINTWISE_KEYS:
        item = next((candidate_item for candidate_item in packet["pointwise"]["items"] if candidate_item.get("item_id") == item_id), None)
        if item is None:
            raise ValueError(f"missing pointwise item {item_id}")
        candidate = next((candidate for candidate in item.get("candidates", []) if candidate.get("blind_id") == candidate_id), None)
        if candidate is None:
            raise ValueError(f"missing pointwise candidate {item_id}/{candidate_id}")
        pointwise.append((item, candidate))
    pair_tasks: list[dict[str, Any]] = []
    for pair_id, orientation in PAIRWISE_KEYS:
        task = next((candidate_task for candidate_task in packet["pairwise"]["tasks"] if candidate_task.get("pair_id") == pair_id and candidate_task.get("orientation") == orientation), None)
        if task is None:
            raise ValueError(f"missing pairwise task {pair_id}/{orientation}")
        pair_tasks.append(task)
    if len(qualifications) != 3 or len(pointwise) != 3 or len(pair_tasks) != 6:
        raise ValueError(f"registered selection shape mismatch: q={len(qualifications)} point={len(pointwise)} pair={len(pair_tasks)}")
    return qualifications, pointwise, pair_tasks


def response_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response has no choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response choice has no message")
    return message


def event_metadata(result: dict[str, Any]) -> dict[str, Any]:
    event = result["event"]
    request = event.get("request") or {}
    response = event.get("response") or {}
    choices = response.get("choices") if isinstance(response, dict) else []
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    reasoning = message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    event["request_sha256"] = sha256_json(request)
    event["request_field_audit"] = {
        "reasoning_effort": request.get("reasoning_effort"),
        "reasoning_format": request.get("reasoning_format"),
        "reasoning_budget_tokens": request.get("reasoning_budget_tokens"),
        "max_tokens": request.get("max_tokens"),
        "temperature": request.get("temperature"),
        "top_p": request.get("top_p"),
        "seed": request.get("seed"),
        "chat_template_kwargs": request.get("chat_template_kwargs"),
        "response_format": request.get("response_format"),
        "enable_thinking_present": "enable_thinking" in request,
        "reasoning_mode_present": "reasoning_mode" in request,
    }
    event["response_metadata"] = {
        "finish_reason": choice.get("finish_reason"),
        "reasoning_content_present": bool(reasoning.strip()),
        "reasoning_chars": len(reasoning),
        "reasoning_sha256": hashlib.sha256(reasoning.encode("utf-8")).hexdigest(),
        "final_content_chars": len(content),
        "final_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "markdown_fence_present": content.strip().startswith("```") or content.strip().endswith("```") if content else False,
        "usage": response.get("usage") if isinstance(response, dict) else None,
    }
    return event


def strict_call(client: Any, instructions: str, track: str, request_id: str, source: str, body: str, events: list[dict[str, Any]]) -> tuple[Any | None, dict[str, Any], str | None]:
    messages = live.build_request_messages(instructions, track, source, body)
    result = client.call(messages, SEED, request_id, MAX_TOKENS)
    event = event_metadata(result)
    event["track"] = track
    event["targeted_request_id"] = request_id
    events.append(event)
    if not result.get("ok"):
        return None, result, result.get("error") or "HTTP/request failure"
    try:
        message = response_message(result["response"])
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("assistant final content is empty; reasoning_content is not a JSON fallback")
        parsed = live.extract_json_strict(content)
        result["assistant_content"] = live.redact(content)
        result["parsed_sha256"] = live.sha256_text(content)
        result["strict_json_pass"] = True
        return parsed, result, None
    except Exception as exc:  # noqa: BLE001 - retain exact contract failure
        result["strict_json_pass"] = False
        result["parse_error"] = live.redact(str(exc))
        return None, result, str(exc)


def checkpoint_bundle(packet: dict[str, Any], packet_sha: str, instructions_sha: str, run_id: str, settings: dict[str, Any], capability: list[dict[str, Any]], rows: list[dict[str, Any]], events: list[dict[str, Any]], completed: list[str], status: str, complete: bool, template_audit: str) -> dict[str, Any]:
    return {
        "schema_version": "mistral-targeted-12-bundle-v1",
        "status": status,
        "complete": complete,
        "packet_id": packet.get("packet_id"),
        "packet_sha256": packet_sha,
        "instructions_sha256": instructions_sha,
        "judge": {"judge_id": JUDGE_ID, "family": FAMILY, "provider": "llama.cpp-local-isolated-targeted", "model": MODEL, "base_url": settings["base_url"], "auth_env": None, "auth_present": False},
        "run_id": run_id,
        "selection": {"qualification": list(QUALIFICATION_KEYS), "pointwise": [f"{item}/{candidate}" for item, candidate in POINTWISE_KEYS], "pairwise": [f"{pair}/{orientation}" for pair, orientation in PAIRWISE_KEYS], "substantive_request_count": 12, "not_selected": ["D_conditional_hidden_critical_check", "E_conditional_clean_false_positive_check"]},
        "settings": settings,
        "template_audit": {"path": template_audit, "status": "PASS_NON_GENERATING_AUDIT"},
        "capability_checks": capability,
        "records": rows,
        "raw_events": events,
        "completed_request_ids": completed,
        "counts": {"records": len(rows), "ok": sum(row.get("status") == "ok" for row in rows), "failed": sum(row.get("status") == "failed" for row in rows), "qualification": sum(row.get("track") == "qualification" for row in rows), "pointwise": sum(row.get("track") == "pointwise" for row in rows), "pairwise": sum(row.get("track") == "pairwise" for row in rows)},
        "integrity": {"score_selected_retry": False, "valid_answer_retry": False, "raw_replacement": False, "cache": "disabled", "parallel": 1},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--template-audit", required=True, type=Path)
    args = parser.parse_args()
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    instructions = INSTRUCTIONS.read_text(encoding="utf-8")
    packet_sha = sha256_file(PACKET)
    instructions_sha = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    qualifications, pointwise, pair_tasks = selected_items(packet)
    run_id = f"{JUDGE_ID}-{packet.get('packet_id', 'packet')}-run1"
    extra_body = {"reasoning_effort": "high", "reasoning_format": "deepseek", "reasoning_budget_tokens": REASONING_BUDGET}
    protected = {"model", "messages", "stream"}
    if protected.intersection(extra_body):
        raise SystemExit("protected request keys attempted in fixed extra body")
    settings = {
        "base_url": args.base_url,
        "temperature": 0,
        "top_p": 1,
        "seed": SEED,
        "max_concurrency": 1,
        "timeout_s": TIMEOUT_S,
        "max_tokens_by_track": {"capability": MAX_TOKENS, "qualification": MAX_TOKENS, "pointwise": MAX_TOKENS, "pairwise": MAX_TOKENS},
        "strict_json_output": True,
        "reasoning_effort": "high",
        "reasoning_format": "deepseek",
        "reasoning_budget_tokens": REASONING_BUDGET,
        "chat_template_kwargs": {"preserve_reasoning": False},
        "omit_fields": ["enable_thinking", "reasoning_mode"],
        "extra_body_keys": sorted(extra_body),
        "response_format_requested": {"type": "json_object"},
        "selection_is_fixed": True,
        "generation_retry_policy": "no retry for valid or failed targeted request; preserve every attempt",
    }
    client = live.Client(args.base_url, MODEL, TIMEOUT_S, None, chat_template_kwargs={"preserve_reasoning": False}, extra_body=extra_body)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    capability: list[dict[str, Any]] = []
    completed: list[str] = []

    def save_checkpoint(status: str = "CHECKPOINT", complete: bool = False) -> None:
        bundle = checkpoint_bundle(packet, packet_sha, instructions_sha, run_id, settings, capability, rows, events, completed, status, complete, str(args.template_audit))
        write_json(args.checkpoint, live.redact(bundle))

    smoke_source = "Technical smoke marker only; this is not a frozen source case and must not be scored."
    for check_id, body in [("capability-01", 'Return exactly this JSON object: {"ok":true,"marker":"targeted-mistral"}'), ("request-format-02", 'Return exactly this JSON object: {"request_received":true,"marker":"targeted-oai-messages"}')]:
        request_id = TARGETED_PREFIX + check_id
        parsed, result, error = strict_call(client, instructions, "capability", request_id, smoke_source, body, events)
        capability.append({"check_id": check_id, "targeted_request_id": request_id, "kind": "technical_marker_only", "status": "ok" if parsed is not None else "failed", "parsed": parsed, "elapsed_s": result.get("elapsed_s"), "error": error, "no_source_case": True, "strict_json": parsed is not None, "reasoning_nonempty": events[-1].get("response_metadata", {}).get("reasoning_content_present", False)})
        completed.append(request_id)
        save_checkpoint()
        if parsed is None:
            raise SystemExit(f"capability smoke failed: {check_id}: {error}")

    for item in qualifications:
        original_id = item["qualification_id"]
        request_id = TARGETED_PREFIX + f"qualification-{original_id}"
        body = "Two anonymous target variants follow. Return the qualification JSON contract from the instructions.\n\n" + "\n\n".join(f"{variant['blind_id']}:\n{variant['target_zh_tw']}" for variant in item["variants"])
        parsed, result, error = strict_call(client, instructions, "qualification", request_id, item["source_en"], body, events)
        base = {"track": "qualification", "run_id": run_id, "targeted_request_id": request_id, "item_id": original_id, "source_id": original_id, "source_text_sha256": live.sha256_text(item["source_en"]), "judge_id": JUDGE_ID, "family": FAMILY, "timings": {"api_elapsed_s": result.get("elapsed_s"), "timeout": False}}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("judgments"), list) or len(parsed["judgments"]) != len(item["variants"]):
            rows.append({"status": "failed", **base, "reason": error or "qualification JSON contract invalid"})
        else:
            judgments_ok = True
            judgment_reason = None
            allowed_ids = {variant["blind_id"] for variant in item["variants"]}
            for judgment in parsed["judgments"]:
                if not isinstance(judgment, dict) or judgment.get("blind_id") not in allowed_ids:
                    judgments_ok = False
                    judgment_reason = "qualification blind_id contract invalid"
                    break
                valid, reason = live.score_shape(judgment)
                if not valid:
                    judgments_ok = False
                    judgment_reason = reason
                    break
            rows.append({"status": "ok" if judgments_ok else "failed", **base, "qualification": {"judgments": parsed["judgments"]}} if judgments_ok else {"status": "failed", **base, "reason": judgment_reason})
        completed.append(request_id)
        save_checkpoint()

    for item, candidate in pointwise:
        original_id = item["item_id"]
        candidate_id = candidate["blind_id"]
        request_id = TARGETED_PREFIX + f"pointwise-{original_id}-{candidate_id}"
        body = "Evaluate this one anonymous candidate and return the pointwise JSON contract.\n\n" + f"{candidate_id}:\n{candidate['target_zh_tw']}"
        parsed, result, error = strict_call(client, instructions, "pointwise", request_id, item["source_en"], body, events)
        base = {"track": "pointwise", "run_id": run_id, "targeted_request_id": request_id, "item_id": original_id, "source_id": original_id, "source_text_sha256": live.sha256_text(item["source_en"]), "judge_id": JUDGE_ID, "family": FAMILY, "candidate_id": candidate_id, "candidate": {"blind_id": candidate_id, "raw_translation_zh_tw": candidate["target_zh_tw"]}, "item": {"item_id": original_id, "domain": item.get("domain"), "input_type": "text", "source_en": item["source_en"], "target_locale": "zh-TW"}, "deterministic": live.deterministic(item["source_en"], candidate["target_zh_tw"]), "timings": {"api_elapsed_s": result.get("elapsed_s"), "timeout": False}}
        if not isinstance(parsed, dict):
            rows.append({"status": "failed", **base, "reason": error or "pointwise JSON contract invalid"})
        elif parsed.get("blind_id") not in {candidate_id, None}:
            rows.append({"status": "failed", **base, "reason": "assistant blind_id disagrees with task"})
        else:
            parsed["blind_id"] = candidate_id
            valid, reason = live.score_shape(parsed)
            if not valid:
                rows.append({"status": "failed", **base, "reason": reason})
            else:
                parsed = dict(parsed)
                if parsed.get("abstain") is not True:
                    parsed["errors"] = live.normalized_errors(parsed.get("errors", []), candidate["target_zh_tw"])
                rows.append({"status": "ok", **base, "pointwise": parsed})
        completed.append(request_id)
        save_checkpoint()

    for task in pair_tasks:
        pair_id = task["pair_id"]
        orientation = task["orientation"]
        request_id = TARGETED_PREFIX + f"pairwise-{pair_id}-{orientation}"
        body = "Compare the two anonymous candidates and return the pairwise JSON contract.\n\n" + f"A ({task['candidate_a']['blind_id']}):\n{task['candidate_a']['target_zh_tw']}\n\nB ({task['candidate_b']['blind_id']}):\n{task['candidate_b']['target_zh_tw']}"
        parsed, result, error = strict_call(client, instructions, "pairwise", request_id, task["source_en"], body, events)
        base = {"track": "pairwise", "run_id": run_id, "targeted_request_id": request_id, "item_id": task["item_id"], "source_id": task["item_id"], "source_text_sha256": live.sha256_text(task["source_en"]), "judge_id": JUDGE_ID, "family": FAMILY, "pair_id": pair_id, "orientation": orientation, "candidate_a": task["candidate_a"]["blind_id"], "candidate_b": task["candidate_b"]["blind_id"], "timings": {"api_elapsed_s": result.get("elapsed_s"), "timeout": False}}
        if not isinstance(parsed, dict) or parsed.get("winner") not in {"A", "B", "TIE"}:
            rows.append({"status": "failed", **base, "reason": error or "pairwise JSON contract invalid"})
        elif not isinstance(parsed.get("confidence"), (int, float)) or not 0 <= float(parsed["confidence"]) <= 1:
            rows.append({"status": "failed", **base, "reason": "pairwise confidence missing/outside 0..1"})
        else:
            rows.append({"status": "ok", **base, "winner": parsed["winner"], "confidence": float(parsed["confidence"]), "critical_override": bool(parsed.get("critical_override", False)), "reason_codes": parsed.get("reason_codes", [])})
        completed.append(request_id)
        save_checkpoint()

    bundle = checkpoint_bundle(packet, packet_sha, instructions_sha, run_id, settings, capability, rows, events, completed, "COMPLETE", True, str(args.template_audit))
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    write_json(args.output, live.redact(bundle))
    print(json.dumps({"output": str(args.output), "counts": bundle["counts"], "capability": capability}, ensure_ascii=False))
    return 0 if bundle["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
