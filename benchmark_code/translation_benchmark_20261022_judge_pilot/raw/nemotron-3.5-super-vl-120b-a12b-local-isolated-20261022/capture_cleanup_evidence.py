#!/usr/bin/env python3
"""Capture post-run cleanup and formal-service preservation evidence."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent


def run(args: list[str]) -> dict[str, object]:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"args": args, "returncode": result.returncode, "stdout": result.stdout[:16000], "stderr": result.stderr[:4000]}


def main() -> int:
    evidence = {
        "schema_version": "nemotron-cleanup-evidence-v1",
        "captured_at_unix": time.time(),
        "isolated_pid": 62893,
        "isolated_port": 18081,
        "isolated_process_after_stop": run(["/bin/ps", "-p", "62893", "-o", "pid=,ppid=,etime=,rss=,command="]),
        "isolated_listener_after_stop": run(["/usr/sbin/lsof", "-nP", "-iTCP:18081", "-sTCP:LISTEN"]),
        "isolated_health_after_stop": run(["/usr/bin/curl", "--noproxy", "*", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:18081/health"]),
        "formal_launchd_after_stop": run(["/bin/launchctl", "print", "system/com.scc.llama-server"]),
        "formal_8080_health_after_stop": run(["/usr/bin/curl", "--noproxy", "*", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:8080/health"]),
        "formal_8080_models_after_stop": run(["/usr/bin/curl", "--noproxy", "*", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:8080/models"]),
        "existing_processes_after_stop": run(["/bin/zsh", "-lc", "/bin/ps -axo pid,etime,rss,command | /usr/bin/grep '[l]lama-server'"]),
        "memory_after_stop": run(["/usr/bin/memory_pressure", "-Q"]),
    }
    RUN_DIR.joinpath("cleanup_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"isolated_process_returncode": evidence["isolated_process_after_stop"]["returncode"], "isolated_listener_returncode": evidence["isolated_listener_after_stop"]["returncode"], "health_output": evidence["isolated_health_after_stop"]["stdout"], "formal_launchd_returncode": evidence["formal_launchd_after_stop"]["returncode"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
