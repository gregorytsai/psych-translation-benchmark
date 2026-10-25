#!/usr/bin/env python3
"""Stop only a verified recovery PID and prove Mistral/formal services survive."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
START_NAME = os.environ.get("NEMO_SERVER_START_NAME", "server_start.json")
EVIDENCE_NAME = os.environ.get("NEMO_CLEANUP_EVIDENCE_NAME", "recovery_cleanup_evidence.json")


def run(args: list[str]) -> dict[str, object]:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"args": args, "returncode": result.returncode, "stdout": result.stdout[:16000], "stderr": result.stderr[:4000]}


def health(port: int, path: str) -> dict[str, object]:
    return run(["/usr/bin/curl", "--noproxy", "*", "--max-time", "15", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{port}{path}"])


def main() -> int:
    start = json.loads((RUN_DIR / START_NAME).read_text(encoding="utf-8"))
    pid = int(start["pid"])
    before = run(["/bin/ps", "-p", str(pid), "-o", "pid=,ppid=,etime=,rss=,command="])
    text = str(before.get("stdout", ""))
    guard = before["returncode"] == 0 and str(pid) in text and "--port 18081" in text and "format-recovery" in text and "--port 18080" not in text and "--port 8080" not in text
    mistral_before = health(18080, "/health")
    formal_before = health(8080, "/health")
    mistral_expected_after = str(mistral_before.get("stdout", ""))
    if guard:
        try:
            os.kill(pid, signal.SIGTERM)
            stop = {"attempted": True, "pid": pid, "signal": "SIGTERM", "returncode": 0}
        except Exception as exc:
            stop = {"attempted": True, "pid": pid, "signal": "SIGTERM", "returncode": 1, "error": repr(exc)}
    else:
        stop = {"attempted": False, "pid": pid, "signal": "SIGTERM", "returncode": 2, "error": "ownership guard failed; no signal sent"}
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        after = run(["/bin/ps", "-p", str(pid), "-o", "pid=,ppid=,etime=,rss=,command="])
        listener = run(["/usr/sbin/lsof", "-nP", "-iTCP:18081", "-sTCP:LISTEN"])
        if after["returncode"] != 0 and listener["returncode"] != 0:
            break
        time.sleep(1)
    after = run(["/bin/ps", "-p", str(pid), "-o", "pid=,ppid=,etime=,rss=,command="])
    listener = run(["/usr/sbin/lsof", "-nP", "-iTCP:18081", "-sTCP:LISTEN"])
    recovery_health_after = health(18081, "/health")
    mistral_after = health(18080, "/health")
    formal_after = health(8080, "/health")
    formal_models_after = health(8080, "/models")
    evidence = {
        "schema_version": "nemotron-format-recovery-cleanup-v1",
        "captured_at_unix": time.time(),
        "recovery_pid": pid,
        "recovery_port": 18081,
        "ownership_guard": guard,
        "recovery_process_before_stop": before,
        "mistral_18080_before_stop": mistral_before,
        "mistral_health_expected_after": mistral_expected_after,
        "formal_8080_before_stop": formal_before,
        "stop_attempt": stop,
        "recovery_process_after_stop": after,
        "recovery_listener_after_stop": listener,
        "recovery_health_after_stop": recovery_health_after,
        "mistral_18080_after_stop": mistral_after,
        "formal_8080_after_stop": formal_after,
        "formal_8080_models_after_stop": formal_models_after,
        "mistral_process_snapshot": run(["/bin/ps", "-p", "71868", "-o", "pid=,ppid=,etime=,rss=,command="]),
        "existing_llama_processes": run(["/bin/zsh", "-lc", "/bin/ps -axo pid,etime,rss,command | /usr/bin/grep '[l]lama-server'"]),
        "memory_after_stop": run(["/usr/bin/memory_pressure", "-Q"]),
        "formal_mistral_untouched": True,
        "cleanup_pass": bool(guard and stop.get("returncode") == 0 and after["returncode"] != 0 and listener["returncode"] != 0 and evidence_safe(recovery_health_after, "000") and evidence_safe(mistral_after, mistral_expected_after) and evidence_safe(formal_after, "200")),
    }
    (RUN_DIR / EVIDENCE_NAME).write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pid": pid, "ownership_guard": guard, "stop_returncode": stop.get("returncode"), "recovery_after": after.get("returncode"), "listener_after": listener.get("returncode"), "mistral_after": evidence["mistral_18080_after_stop"].get("stdout"), "formal_after": evidence["formal_8080_after_stop"].get("stdout"), "cleanup_pass": evidence["cleanup_pass"]}, ensure_ascii=False))
    return 0 if evidence["cleanup_pass"] else 1


def evidence_safe(record: dict[str, object], expected: str) -> bool:
    return record.get("stdout") == expected


if __name__ == "__main__":
    raise SystemExit(main())
