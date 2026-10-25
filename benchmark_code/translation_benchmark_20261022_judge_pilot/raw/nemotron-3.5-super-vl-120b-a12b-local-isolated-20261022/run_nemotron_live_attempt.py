#!/usr/bin/env python3
"""Run the frozen 39-request blind pilot and preserve an attempt transcript."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/benchmark")
RUN_DIR = ROOT / "translation_benchmark_20261022_judge_pilot/raw/nemotron-3.5-super-vl-120b-a12b-local-isolated-20261022"
RUNNER = ROOT / "translation_benchmark_20261022_judge_pilot/run_live_judge.py"
PACKET = ROOT / "translation_benchmark_20261022_judge_pilot/judge_packet.json"
INSTRUCTIONS = ROOT / "translation_benchmark_20261022_judge_pilot/judge_instructions.txt"
OUTPUT = RUN_DIR / "nemotron-3.5-super-vl-120b-a12b-local-isolated.raw.json"
CHECKPOINT = RUN_DIR / "nemotron-3.5-super-vl-120b-a12b-local-isolated.checkpoint.json"
ATTEMPT_LOG = RUN_DIR / "attempt-01.log"
ATTEMPT_RESULT = RUN_DIR / "attempt-01.result.json"
ALIAS = "nemotron-3.5-super-vl-120b-a12b-local-isolated"


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER),
        "--packet", str(PACKET),
        "--instructions", str(INSTRUCTIONS),
        "--base-url", "http://127.0.0.1:18081/v1",
        "--model", ALIAS,
        "--judge-id", ALIAS,
        "--family", "nvidia-nemotron",
        "--provider", "llama.cpp-local-isolated",
        "--output", str(OUTPUT),
        "--checkpoint", str(CHECKPOINT),
        "--timeout", "240",
        "--disable-thinking",
        "--extra-body", json.dumps({"reasoning_format": "deepseek", "chat_template_kwargs": {"preserve_reasoning": False}}, separators=(",", ":")),
    ]
    started = time.time()
    metadata = {
        "schema_version": "nemotron-live-attempt-v1",
        "attempt": 1,
        "started_at_unix": started,
        "cwd": str(ROOT),
        "command": command,
        "substantive_request_plan": {"qualification_parent_requests": 12, "qualification_variant_judgments": 24, "pointwise_requests": 21, "pairwise_requests": 6, "substantive_requests": 39, "runner_capability_checks": 2},
        "request_compatibility": {"response_format": "json_object", "reasoning_format": "deepseek", "chat_template_kwargs": {"enable_thinking": False, "preserve_reasoning": False}},
        "input_packet": str(PACKET),
        "input_instructions": str(INSTRUCTIONS),
        "output": str(OUTPUT),
        "checkpoint": str(CHECKPOINT),
        "attempt_log": str(ATTEMPT_LOG),
        "status": "running",
    }
    ATTEMPT_RESULT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with ATTEMPT_LOG.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event": "start", **metadata}, ensure_ascii=False) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    finished = time.time()
    metadata.update({"finished_at_unix": finished, "elapsed_s": finished - started, "returncode": completed.returncode, "status": "complete" if completed.returncode == 0 else "complete_with_failures"})
    with ATTEMPT_LOG.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"event": "finish", "returncode": completed.returncode, "elapsed_s": finished - started}, ensure_ascii=False) + "\n")
    ATTEMPT_RESULT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"returncode": completed.returncode, "elapsed_s": finished - started, "output": str(OUTPUT), "checkpoint": str(CHECKPOINT)}, ensure_ascii=False))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
