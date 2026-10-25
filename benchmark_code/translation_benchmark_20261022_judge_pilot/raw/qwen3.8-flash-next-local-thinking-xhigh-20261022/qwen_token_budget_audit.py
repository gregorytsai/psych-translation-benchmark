#!/usr/bin/env python3
"""Audit Qwen thinking and final-text token counts without copying CoT text."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Any


COUNT_RE = re.compile(r"Total number of tokens:\s*(\d+)")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokenize(tokenizer: str, model: str, value: str, timeout: float) -> int:
    proc = subprocess.run(
        [tokenizer, "-m", model, "--stdin", "--no-bos", "--no-parse-special", "--show-count", "--log-disable"],
        input=value.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed rc={proc.returncode}: {proc.stderr.decode(errors='replace')[-400:]}")
    match = COUNT_RE.search(proc.stdout.decode(errors="replace"))
    if not match:
        raise RuntimeError(f"llama-tokenize count missing: {proc.stdout.decode(errors='replace')[-400:]}")
    return int(match.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--tokenizer", default="/opt/homebrew/bin/llama-tokenize")
    ap.add_argument("--model", required=True, type=pathlib.Path)
    ap.add_argument("--timeout", default=180.0, type=float)
    args = ap.parse_args()

    raw_bytes = args.raw.read_bytes()
    bundle: dict[str, Any] = json.loads(raw_bytes)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event in bundle.get("raw_events", []):
        request_id = event.get("request_id")
        if request_id in {"capability-01", "request-format-02"}:
            continue
        response = event.get("response") or {}
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else {}
        message = message or {}
        reasoning = message.get("reasoning_content") or ""
        content = message.get("content") or ""
        try:
            reasoning_tokens = tokenize(args.tokenizer, str(args.model), reasoning, args.timeout)
            content_tokens = tokenize(args.tokenizer, str(args.model), content, args.timeout)
            error = None
        except Exception as exc:  # noqa: BLE001 - retain per-event audit failure
            reasoning_tokens = None
            content_tokens = None
            error = str(exc)
            failures.append({"request_id": request_id, "error": error})
        usage = response.get("usage") or {}
        request = event.get("request") or {}
        choice = choices[0] if choices else {}
        inferred_track = (
            "qualification" if str(request_id).startswith("qualification-") else
            "pointwise" if str(request_id).startswith("pointwise-") else
            "pairwise" if str(request_id).startswith("pairwise-") else None
        )
        rows.append(
            {
                "request_id": request_id,
                "track": event.get("track") or inferred_track,
                "status": event.get("status"),
                "http_status": event.get("http_status"),
                "finish_reason": choice.get("finish_reason"),
                "response_sha256": event.get("response_sha256"),
                "reasoning_sha256": sha256_text(reasoning),
                "content_sha256": sha256_text(content),
                "reasoning_chars": len(reasoning),
                "content_chars": len(content),
                "offline_reasoning_tokens": reasoning_tokens,
                "offline_completion_tokens": content_tokens,
                "api_prompt_tokens": usage.get("prompt_tokens"),
                "api_completion_tokens": usage.get("completion_tokens"),
                "api_total_tokens": usage.get("total_tokens"),
                "request_max_tokens": request.get("max_tokens"),
                "request_reasoning_budget_tokens": request.get("reasoning_budget_tokens"),
                "request_reasoning_effort": request.get("reasoning_effort"),
                "request_reasoning_format": request.get("reasoning_format"),
                "request_chat_template_kwargs": request.get("chat_template_kwargs"),
                "error": error,
            }
        )

    output: dict[str, Any] = {
        "schema_version": "qwen-thinking-token-budget-audit-v1",
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_path": str(args.raw),
        "tokenizer": str(args.tokenizer),
        "model": str(args.model),
        "model_tokenizer_source": "Qwen3.8-Flash-Next first GGUF shard",
        "scope": "39 substantive requests only; capability smoke excluded",
        "expected_requests": 39,
        "rows": rows,
        "counts": {
            "rows": len(rows),
            "ok": sum(row["status"] == "ok" for row in rows),
            "tokenized": sum(row["offline_reasoning_tokens"] is not None and row["offline_completion_tokens"] is not None for row in rows),
            "audit_failures": len(failures),
            "reasoning_nonempty": sum(row["reasoning_chars"] > 0 for row in rows),
        },
        "failures": failures,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": output["counts"]}, ensure_ascii=False))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
