#!/usr/bin/env python3
"""Start the exact Nemotron model in a detached, loopback-only run-local process."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/benchmark")
RUN_DIR = ROOT / "translation_benchmark_20261022_judge_pilot/raw/nemotron-3.5-super-vl-120b-a12b-local-thinking-on-20261022"
MODEL = ROOT / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/UD-Q6_K_XL/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL-00001-of-00004.gguf"
MM_PROJ = ROOT / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/mmproj-BF16.gguf"
SERVER = ROOT / "llama-v0.7.0/build-release/bin/llama-server"
ALIAS = "nemotron-3.5-super-vl-120b-a12b-local-thinking-on"


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(SERVER),
        "--model", str(MODEL),
        "--mmproj", str(MM_PROJ),
        "--alias", ALIAS,
        "--host", "127.0.0.1",
        "--port", "18081",
        "--jinja",
        "--no-webui",
        "--offline",
        "--n-gpu-layers", "all",
        "--flash-attn", "on",
        "--ctx-size", "16384",
        "--parallel", "1",
        "--no-cont-batching",
        "--fit", "on",
        "--fit-target", "16384",
        "--fit-ctx", "16384",
        "--load-mode", "mmap",
        "--no-cache-prompt",
        "--reasoning", "on",
        "--reasoning-format", "deepseek",
        "--no-reasoning-preserve",
        "--reasoning-budget", "4096",
        "--log-verbosity", "2",
    ]
    log_path = RUN_DIR / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
    record = {
        "schema_version": "nemotron-thinking-on-server-start-v1",
        "started_at_unix": time.time(),
        "pid": process.pid,
        "command": command,
        "cwd": str(ROOT),
        "log": str(log_path),
        "loopback": "127.0.0.1:18081",
        "model": str(MODEL),
        "mmproj": str(MM_PROJ),
        "settings": {
            "ctx_size": 16384,
            "reasoning": "on",
            "reasoning_format": "deepseek",
            "reasoning_preserve": False,
            "reasoning_budget_tokens": 4096,
            "native_template": True,
        },
    }
    (RUN_DIR / "server_start.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
