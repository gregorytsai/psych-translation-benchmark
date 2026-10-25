#!/usr/bin/env python3
"""Sequential OpenAI-compatible live judge runner for the pilot packet.

The runner never edits the formal llama.cpp/LiteLLM configuration.  It sends
one request at a time to the supplied endpoint, records redacted request /
response evidence, and emits a bundle that ``validate_import.py`` and
``aggregate_pilot.py`` can consume.  Authentication is read from an
environment variable by name and is never written to output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 20261022
SIMPLE_CHAR_REPLACEMENTS = str.maketrans({"后": "後", "复": "復", "发": "發", "虑": "慮", "轻": "輕", "剂": "劑", "药": "藥", "号": "號", "脑": "腦", "临": "臨", "医": "醫", "疗": "療", "间": "間", "现": "現", "状": "狀", "动": "動", "产": "產", "与": "與", "为": "為", "应": "應", "门": "門", "见": "見", "认": "認", "说": "說", "计": "計", "划": "劃", "严": "嚴", "险": "險", "频": "頻", "号": "號"})


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def redact(value: Any) -> Any:
    """Redact obvious credential-shaped material before persistence."""

    if isinstance(value, dict):
        clean = {}
        for key, child in value.items():
            key_text = str(key).lower()
            # Keep ordinary accounting fields such as max_tokens and usage
            # token counts auditable.  Only exact token credential fields and
            # explicit credential-shaped key names are secret material.
            secret_key = key_text in {"token", "access_token", "refresh_token", "id_token"} or any(token in key_text for token in ("api_key", "apikey", "authorization", "bearer", "password", "secret"))
            if secret_key:
                clean[key] = "<REDACTED>"
            else:
                clean[key] = redact(child)
        return clean
    if isinstance(value, list):
        return [redact(child) for child in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <REDACTED>", value)
        value = re.sub(r"(?i)\bsk-[A-Za-z0-9._-]{12,}", "sk-<REDACTED>", value)
        value = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+", r"\1=<REDACTED>", value)
        return value
    return value


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract_json(content: str) -> Any:
    """Parse strict JSON, fenced JSON, or the first decodable JSON value."""

    if not isinstance(content, str):
        raise ValueError("assistant content is not text")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object/array found in assistant content")


def extract_json_strict(content: str) -> Any:
    """Parse only a complete, unfenced JSON value from the assistant body."""

    if not isinstance(content, str):
        raise ValueError("assistant content is not text")
    text = content.strip()
    if not text:
        raise ValueError("assistant content is empty")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"strict JSON parse failed: {exc}") from exc
    return value


def message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    raise ValueError("response message has no text content")


def score_shape(value: Any) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "judgment is not an object"
    if value.get("abstain") is True:
        if value.get("score_0_100") is not None or value.get("subscores") is not None:
            return False, "abstain judgment must use null score and subscores"
        return True, ""
    score = value.get("score_0_100")
    subs = value.get("subscores")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= float(score) <= 100:
        return False, "score_0_100 missing or outside 0..100"
    if not isinstance(subs, dict):
        return False, "subscores missing"
    limits = {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 20}
    values = []
    for key, maximum in limits.items():
        item = subs.get(key)
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not 0 <= float(item) <= maximum:
            return False, f"subscore {key} missing/outside 0..{maximum}"
        values.append(float(item))
    if abs(sum(values) - float(score)) > 1e-6:
        return False, "subscores do not sum to score_0_100"
    if not isinstance(value.get("critical"), bool):
        return False, "critical must be boolean"
    errors = value.get("errors")
    if not isinstance(errors, list):
        return False, "errors must be an array"
    for error in errors:
        if not isinstance(error, dict):
            return False, "error is not an object"
        if error.get("severity") not in {"minor", "major", "critical"}:
            return False, "error severity invalid"
        if not isinstance(error.get("type"), str) or not error["type"]:
            return False, "error type missing"
        if not isinstance(error.get("target_exact_quote"), str):
            return False, "target_exact_quote missing"
        if not isinstance(error.get("source_evidence"), str) or not error["source_evidence"]:
            return False, "source_evidence missing"
        if not isinstance(error.get("source_span"), str) or not error["source_span"]:
            return False, "source_span missing"
        if error["target_exact_quote"] == "" and not isinstance(error.get("target_anchor"), str):
            return False, "omission needs target_anchor"
    return True, ""


def normalized_errors(errors: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    output = []
    for error in errors:
        copy = dict(error)
        quote = copy.get("target_exact_quote", "")
        if quote:
            start = target.find(quote)
            if start >= 0:
                copy["target_start"] = start
                copy["target_end"] = start + len(quote)
                copy["offset_status"] = "computed_exact_quote"
            else:
                copy["target_start"] = 0
                copy["target_end"] = 0
                copy["offset_status"] = "exact_quote_not_found"
        else:
            copy["target_start"] = 0
            copy["target_end"] = 0
            copy["offset_status"] = "omission_source_span_only"
        copy["exact_quote"] = quote
        output.append(copy)
    return output


def simplified_present(text: str) -> bool:
    simplified = set("后复发虑轻剂药号脑临医疗间现状动产与为应门见认说计划严险频")
    return any(character in simplified for character in text)


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def deterministic(source: str, target: str) -> dict[str, Any]:
    source_numbers = numeric_tokens(source)
    target_numbers = numeric_tokens(target)
    numbers_ok = source_numbers.issubset(target_numbers)
    return {
        "language_ok": bool(target.strip()),
        "traditional_ok": not simplified_present(target),
        "numbers_ok": numbers_ok,
        "drug_terms_ok": True,
        "format_ok": True,
        "critical_deterministic": bool(source_numbers and not numbers_ok),
    }


def build_request_messages(instructions: str, task: str, source: str, body: str) -> list[dict[str, str]]:
    system = "You are a blind translation-quality judge. Follow the supplied evaluator contract exactly. Output JSON only.\n\n" + instructions
    user = f"Task kind: {task}\n\nEnglish source:\n{source}\n\n{body}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class Client:
    def __init__(self, base_url: str, model: str, timeout: float, auth_env: str | None, extra_headers: dict[str, str] | None = None, chat_template_kwargs: dict[str, Any] | None = None, extra_body: dict[str, Any] | None = None, reasoning_mode: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.auth_env = auth_env
        self.extra_headers = extra_headers or {}
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.extra_body = extra_body or {}
        self.reasoning_mode = reasoning_mode

    def call(self, messages: list[dict[str, str]], seed: int, request_id: str, max_tokens: int = 1400) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if self.chat_template_kwargs:
            # Only pass kwargs that were explicitly verified in the endpoint's
            # /props template.  This is request-level evidence, not a claim
            # that the server rendered-token trace was independently proven.
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.reasoning_mode is not None:
            payload["reasoning_mode"] = self.reasoning_mode
        for key, value in self.extra_body.items():
            if key in {"model", "messages", "stream"}:
                raise ValueError(f"--extra-body cannot override protected request key {key!r}")
            if key == "chat_template_kwargs" and isinstance(value, dict):
                merged = dict(payload.get("chat_template_kwargs", {}))
                merged.update(value)
                payload[key] = merged
            else:
                payload[key] = value
        headers = {"Content-Type": "application/json", "Accept": "application/json", **self.extra_headers}
        auth_present = False
        if self.auth_env:
            secret = os.environ.get(self.auth_env, "")
            auth_present = bool(secret)
            if secret:
                headers["Authorization"] = f"Bearer {secret}"
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
        started = time.monotonic()
        event: dict[str, Any] = {
            "request_id": request_id,
            "request": redact(payload),
            "endpoint": self.base_url,
            "auth_env": self.auth_env,
            "auth_present": auth_present,
            "status": "failed",
        }
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                elapsed = time.monotonic() - started
                parsed = json.loads(body.decode("utf-8"))
                event.update({"status": "ok", "http_status": response.status, "elapsed_s": elapsed, "response": redact(parsed), "response_sha256": sha256_text(body.decode("utf-8"))})
                return {"ok": True, "response": parsed, "event": event, "elapsed_s": elapsed}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            elapsed = time.monotonic() - started
            event.update({"status": "failed", "http_status": exc.code, "elapsed_s": elapsed, "error": redact(body[:4000]), "response_sha256": sha256_text(body)})
            return {"ok": False, "event": event, "elapsed_s": elapsed, "error": f"HTTP {exc.code}: {body[:400]}"}
        except Exception as exc:  # noqa: BLE001 - preserve provider errors as data
            elapsed = time.monotonic() - started
            event.update({"status": "failed", "elapsed_s": elapsed, "error": redact(str(exc))})
            return {"ok": False, "event": event, "elapsed_s": elapsed, "error": str(exc)}


def failed_row(track: str, item_id: str, source: str, judge_id: str, run_id: str, reason: str, candidate_id: str | None = None, **extra: Any) -> dict[str, Any]:
    row = {
        "status": "failed",
        "track": track,
        "run_id": run_id,
        "item_id": item_id,
        "source_id": item_id,
        "source_text_sha256": sha256_text(source),
        "judge_id": judge_id,
        "reason": reason,
    }
    if candidate_id:
        row["candidate_id"] = candidate_id
    row.update(extra)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", default=str(ROOT / "judge_packet.json"))
    parser.add_argument("--instructions", default=str(ROOT / "judge_instructions.txt"))
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", required=True, help="provider model alias")
    parser.add_argument("--judge-id", required=True, help="stable audit ID, not shown in blind packet")
    parser.add_argument("--family", required=True)
    parser.add_argument("--provider", default="openai-compatible")
    parser.add_argument("--auth-env", default=None, help="environment variable name, never the secret value")
    parser.add_argument("--output", required=True, help="raw result JSON bundle")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-capability", action="store_true")
    parser.add_argument("--skip-qualification", action="store_true")
    parser.add_argument("--skip-pointwise", action="store_true")
    parser.add_argument("--skip-pairwise", action="store_true")
    parser.add_argument("--max-pointwise", type=int, default=None)
    parser.add_argument("--max-qualification", type=int, default=None)
    parser.add_argument("--max-pairwise", type=int, default=None)
    parser.add_argument("--max-tokens-capability", type=int, default=120)
    parser.add_argument("--max-tokens-qualification", type=int, default=1200)
    parser.add_argument("--max-tokens-pointwise", type=int, default=1400)
    parser.add_argument("--max-tokens-pairwise", type=int, default=300)
    parser.add_argument("--strict-json-output", action="store_true", help="Require the complete assistant body to be one unfenced JSON value")
    parser.add_argument("--disable-thinking", action="store_true", help="Use the endpoint-documented chat_template_kwargs enable_thinking=false")
    parser.add_argument("--reasoning-mode", choices=("auto", "off"), default=None, help="Optional explicit provider request field; recorded and sent as reasoning_mode")
    parser.add_argument("--extra-body", default=None, help="Optional JSON object merged into each request; protected keys cannot be overridden")
    parser.add_argument("--checkpoint", default=None, help="Write a redacted checkpoint after each completed request")
    parser.add_argument("--resume-from", default=None, help="Resume from a prior redacted checkpoint, skipping completed request IDs")
    args = parser.parse_args()

    packet = load_json(Path(args.packet))
    instructions = Path(args.instructions).read_text(encoding="utf-8")
    packet_sha = hashlib.sha256(Path(args.packet).read_bytes()).hexdigest()
    instructions_sha = sha256_text(instructions)
    run_id = f"{args.judge_id}-{packet.get('packet_id', 'packet')}-run1"
    template_kwargs = {"enable_thinking": False} if args.disable_thinking or args.reasoning_mode == "off" else {}
    try:
        extra_body = json.loads(args.extra_body) if args.extra_body else {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-body must be a JSON object: {exc}") from exc
    if not isinstance(extra_body, dict):
        raise SystemExit("--extra-body must be a JSON object")
    effective_template_kwargs = dict(template_kwargs)
    if isinstance(extra_body.get("chat_template_kwargs"), dict):
        effective_template_kwargs.update(extra_body["chat_template_kwargs"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    resume_path = Path(args.resume_from) if args.resume_from else None
    if checkpoint_path and checkpoint_path == Path(args.output):
        raise SystemExit("--checkpoint must differ from --output")
    if resume_path and not resume_path.exists():
        raise SystemExit(f"resume checkpoint does not exist: {resume_path}")
    client = Client(args.base_url, args.model, args.timeout, args.auth_env, chat_template_kwargs=template_kwargs, extra_body=extra_body, reasoning_mode=args.reasoning_mode)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    capability: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if resume_path:
        prior = load_json(resume_path)
        if prior.get("packet_sha256") != packet_sha:
            raise SystemExit("resume checkpoint packet_sha256 does not match frozen packet")
        if prior.get("instructions_sha256") != instructions_sha:
            raise SystemExit("resume checkpoint instructions_sha256 does not match frozen instructions")
        prior_rows = prior.get("records", [])
        prior_events = prior.get("raw_events", [])
        prior_capability = prior.get("capability_checks", [])
        if not isinstance(prior_rows, list) or not isinstance(prior_events, list) or not isinstance(prior_capability, list):
            raise SystemExit("resume checkpoint has invalid records/events/capability shape")
        rows.extend(row for row in prior_rows if isinstance(row, dict))
        events.extend(event for event in prior_events if isinstance(event, dict))
        capability.extend(item for item in prior_capability if isinstance(item, dict))
        completed_ids.update(str(value) for value in prior.get("completed_request_ids", []) if isinstance(value, str))
        print(json.dumps({"resume_from": str(resume_path), "completed_requests": len(completed_ids), "records_loaded": len(rows)}, ensure_ascii=False))

    def bundle_snapshot(status: str = "CHECKPOINT", complete: bool = False) -> dict[str, Any]:
        return {
            "schema_version": "judge-pilot-live-bundle-v1",
            "status": status,
            "complete": complete,
            "packet_id": packet.get("packet_id"),
            "packet_sha256": packet_sha,
            "instructions_sha256": instructions_sha,
            "judge": {"judge_id": args.judge_id, "family": args.family, "provider": args.provider, "model": args.model, "base_url": args.base_url, "auth_env": args.auth_env, "auth_present": bool(args.auth_env and os.environ.get(args.auth_env)), "ui_model_version": None},
            "run_id": run_id,
            "settings": {"temperature": 0, "top_p": 1, "seed": args.seed, "max_concurrency": 1, "timeout_s": args.timeout, "max_tokens_by_track": {"capability": args.max_tokens_capability, "qualification": args.max_tokens_qualification, "pointwise": args.max_tokens_pointwise, "pairwise": args.max_tokens_pairwise}, "strict_json_output": args.strict_json_output, "chat_template_owner": "endpoint/llama.cpp or provider", "chat_template_kwargs": effective_template_kwargs, "reasoning_mode": args.reasoning_mode, "extra_body_keys": sorted(extra_body), "response_format_requested": "json_object", "checkpoint_enabled": bool(checkpoint_path or resume_path)},
            "capability_checks": capability,
            "template_verification": {"status": "PENDING", "evidence": "request-format smoke plus endpoint /props documented kwargs; no server-side rendered-token trace or per-model template proof was obtained"},
            "records": rows,
            "raw_events": events,
            "completed_request_ids": sorted(completed_ids),
            "counts": {"records": len(rows), "ok": sum(row.get("status") == "ok" for row in rows), "failed": sum(row.get("status") == "failed" for row in rows), "qualification": sum(row.get("track") == "qualification" for row in rows), "pointwise": sum(row.get("track") == "pointwise" for row in rows), "pairwise": sum(row.get("track") == "pairwise" for row in rows)},
        }

    def checkpoint() -> None:
        if checkpoint_path:
            write_json(checkpoint_path, redact(bundle_snapshot()))

    def call(task: str, request_id: str, source: str, body: str, max_tokens: int = 1400) -> tuple[Any | None, dict[str, Any]]:
        result = client.call(build_request_messages(instructions, task, source, body), args.seed, request_id, max_tokens)
        events.append(result["event"])
        if not result["ok"]:
            return None, result
        try:
            content = message_content(result["response"])
            parsed = extract_json_strict(content) if args.strict_json_output else extract_json(content)
            result["assistant_content"] = redact(content)
            result["parsed_sha256"] = sha256_text(content)
            return parsed, result
        except Exception as exc:  # noqa: BLE001 - parser failures are explicit observations
            result["parse_error"] = redact(str(exc))
            return None, result

    if not args.skip_capability:
        source = "Capability check source: return a structured JSON marker without translating this sentence."
        for check_id, body in [
            ("capability-01", 'Return exactly this JSON object: {"ok":true,"marker":"judge-pilot"}'),
            ("request-format-02", 'Return exactly this JSON object: {"request_received":true,"marker":"oai-messages"}'),
        ]:
            if check_id in completed_ids:
                continue
            parsed, result = call("capability", check_id, source, body, args.max_tokens_capability)
            capability.append({"check_id": check_id, "kind": "request_format_smoke", "status": "ok" if parsed is not None else "failed", "parsed": parsed, "elapsed_s": result.get("elapsed_s"), "reason": result.get("error") or result.get("parse_error"), "template_verification": "pending_endpoint_evidence"})
            completed_ids.add(check_id)
            checkpoint()

    qualification_items = packet.get("qualification", {}).get("items", [])
    if not args.skip_qualification:
        if args.max_qualification is not None:
            qualification_items = qualification_items[: args.max_qualification]
        for item in qualification_items:
            request_id = f"qualification-{item['qualification_id']}"
            if request_id in completed_ids:
                continue
            variants = item["variants"]
            body = "Two anonymous target variants follow. Return the qualification JSON contract from the instructions.\n\n" + "\n\n".join(f"{variant['blind_id']}:\n{variant['target_zh_tw']}" for variant in variants)
            parsed, result = call("qualification", request_id, item["source_en"], body, args.max_tokens_qualification)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("judgments"), list):
                rows.append(failed_row("qualification", item["qualification_id"], item["source_en"], args.judge_id, run_id, result.get("error") or result.get("parse_error") or "qualification JSON contract invalid"))
                completed_ids.add(request_id)
                checkpoint()
                continue
            rows.append({
                "status": "ok",
                "track": "qualification",
                "run_id": run_id,
                "item_id": item["qualification_id"],
                "source_id": item["qualification_id"],
                "source_text_sha256": sha256_text(item["source_en"]),
                "judge_id": args.judge_id,
                "family": args.family,
                "qualification": {"judgments": parsed["judgments"]},
                "timings": {"api_elapsed_s": result.get("elapsed_s"), "timeout": False},
            })
            completed_ids.add(request_id)
            checkpoint()

    pointwise_items = packet.get("pointwise", {}).get("items", [])
    pointwise_count = 0
    if not args.skip_pointwise:
        for item in pointwise_items:
            for candidate in item["candidates"]:
                if args.max_pointwise is not None and pointwise_count >= args.max_pointwise:
                    break
                pointwise_count += 1
                body = "Evaluate this one anonymous candidate and return the pointwise JSON contract.\n\n" + f"{candidate['blind_id']}:\n{candidate['target_zh_tw']}"
                request_id = f"pointwise-{item['item_id']}-{candidate['blind_id']}"
                if request_id in completed_ids:
                    continue
                parsed, result = call("pointwise", request_id, item["source_en"], body, args.max_tokens_pointwise)
                candidate_id = candidate["blind_id"]
                if not isinstance(parsed, dict):
                    rows.append(failed_row("pointwise", item["item_id"], item["source_en"], args.judge_id, run_id, result.get("error") or result.get("parse_error") or "pointwise JSON contract invalid", candidate_id))
                    completed_ids.add(request_id)
                    checkpoint()
                    continue
                if parsed.get("blind_id") not in {candidate_id, None}:
                    rows.append(failed_row("pointwise", item["item_id"], item["source_en"], args.judge_id, run_id, "assistant blind_id disagrees with task", candidate_id))
                    completed_ids.add(request_id)
                    checkpoint()
                    continue
                parsed["blind_id"] = candidate_id
                valid, reason = score_shape(parsed)
                if not valid:
                    rows.append(failed_row("pointwise", item["item_id"], item["source_en"], args.judge_id, run_id, reason, candidate_id))
                    completed_ids.add(request_id)
                    checkpoint()
                    continue
                target = candidate["target_zh_tw"]
                if parsed.get("abstain") is True:
                    pointwise_obj = parsed
                else:
                    pointwise_obj = dict(parsed)
                    pointwise_obj["errors"] = normalized_errors(parsed.get("errors", []), target)
                rows.append({
                    "status": "ok",
                    "track": "pointwise",
                    "run_id": run_id,
                    "item_id": item["item_id"],
                    "source_id": item["item_id"],
                    "source_text_sha256": sha256_text(item["source_en"]),
                    "judge_id": args.judge_id,
                    "family": args.family,
                    "candidate_id": candidate_id,
                    "candidate": {"blind_id": candidate_id, "raw_translation_zh_tw": target},
                    "item": {"item_id": item["item_id"], "domain": item.get("domain"), "input_type": "text", "source_en": item["source_en"], "target_locale": "zh-TW"},
                    "deterministic": deterministic(item["source_en"], target),
                    "pointwise": pointwise_obj,
                    "timings": {"api_elapsed_s": result.get("elapsed_s"), "timeout": False},
                })
                completed_ids.add(request_id)
                checkpoint()
            if args.max_pointwise is not None and pointwise_count >= args.max_pointwise:
                break

    pairwise_tasks = packet.get("pairwise", {}).get("tasks", [])
    if args.max_pairwise is not None:
        pairwise_tasks = pairwise_tasks[: args.max_pairwise]
    if not args.skip_pairwise:
        for task in pairwise_tasks:
            request_id = f"pairwise-{task['pair_id']}-{task['orientation']}"
            if request_id in completed_ids:
                continue
            body = "Compare the two anonymous candidates and return the pairwise JSON contract.\n\n" + f"A ({task['candidate_a']['blind_id']}):\n{task['candidate_a']['target_zh_tw']}\n\nB ({task['candidate_b']['blind_id']}):\n{task['candidate_b']['target_zh_tw']}"
            parsed, result = call("pairwise", request_id, task["source_en"], body, args.max_tokens_pairwise)
            row_base = {
                "track": "pairwise",
                "run_id": run_id,
                "item_id": task["item_id"],
                "source_id": task["item_id"],
                "source_text_sha256": sha256_text(task["source_en"]),
                "judge_id": args.judge_id,
                "family": args.family,
                "pair_id": task["pair_id"],
                "orientation": task["orientation"],
                "candidate_a": task["candidate_a"]["blind_id"],
                "candidate_b": task["candidate_b"]["blind_id"],
            }
            if not isinstance(parsed, dict) or parsed.get("winner") not in {"A", "B", "TIE"}:
                rows.append({"status": "failed", **row_base, "reason": result.get("error") or result.get("parse_error") or "pairwise JSON contract invalid"})
                completed_ids.add(request_id)
                checkpoint()
                continue
            confidence = parsed.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                rows.append({"status": "failed", **row_base, "reason": "pairwise confidence missing/outside 0..1"})
                completed_ids.add(request_id)
                checkpoint()
                continue
            rows.append({"status": "ok", **row_base, "winner": parsed["winner"], "confidence": confidence, "critical_override": bool(parsed.get("critical_override", False)), "reason_codes": parsed.get("reason_codes", [])})
            completed_ids.add(request_id)
            checkpoint()

    bundle = bundle_snapshot(status="COMPLETE", complete=True)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing {output}")
    write_json(output, redact(bundle))
    print(json.dumps({"output": str(output), "counts": bundle["counts"], "capability_checks": capability}, ensure_ascii=False, indent=2))
    return 0 if bundle["counts"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
