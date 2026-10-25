#!/usr/bin/env python3
"""Reproducible OpenAI-compatible template smoke/quality run.

The server must already be running on an isolated port.  This deliberately
sends ordinary OpenAI messages (never pre-rendered turn markers) and records
the complete JSON response, including llama.cpp's verbose prompt evidence.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(base: str, payload: dict) -> tuple[int, dict, float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read()), time.perf_counter() - t0
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18080")
    ap.add_argument("--model", default="tg4-audit")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--reasoning-format",
        default=None,
        help="Optional llama.cpp response parser format, e.g. deepseek for Gemma4.",
    )
    ap.add_argument(
        "--user-instruction",
        default="",
        help="Optional instruction prepended as an OpenAI text part (needed by Gemma4).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    inputs = [json.loads(line) for line in (root / "inputs.jsonl").read_text().splitlines() if line.strip()]
    cases = [
        {
            "id": item["id"],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{args.user_instruction}\n\n{item['english']}"
                        if args.user_instruction
                        else item["english"]
                    ),
                }
            ],
        }
        for item in inputs
    ]
    image = json.loads((root / "image_case.json").read_text())
    cases.append(
        {
            "id": image["id"],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        ([{"type": "text", "text": args.user_instruction}] if args.user_instruction else [])
                        + [
                            {
                                "type": "image_url",
                                "image_url": {"url": "file://image_clinical_note.png"},
                            }
                        ]
                    ),
                }
            ],
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for case in cases:
            payload = {
                "model": args.model,
                "messages": case["messages"],
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 1,
                "stream": False,
                "reasoning_effort": "none",
                "verbose": True,
            }
            if args.reasoning_format:
                payload["reasoning_format"] = args.reasoning_format
            status, response, elapsed = post(args.base, payload)
            record = {
                "id": case["id"],
                "http_status": status,
                "elapsed_seconds": elapsed,
                "request": payload,
                "response": response,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            print(
                f"{case['id']} status={status} elapsed={elapsed:.3f}s "
                f"finish={choice.get('finish_reason')} content={message.get('content', '')!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
