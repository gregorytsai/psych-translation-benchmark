#!/usr/bin/env python3
"""Stop only the run-local Nemotron PID and prove formal 8080 remains healthy."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
START = RUN_DIR / "server_start.json"


def run(args: list[str]) -> dict[str, object]:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"args": args, "returncode": result.returncode, "stdout": result.stdout[:16000], "stderr": result.stderr[:4000]}


def pid_snapshot(pid: int) -> dict[str, object]:
    return run(["/bin/ps", "-p", str(pid), "-o", "pid=,ppid=,etime=,rss=,command="])


def listener_snapshot(port: int) -> dict[str, object]:
    return run(["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"])


def health(port: int, path: str) -> dict[str, object]:
    return run(["/usr/bin/curl", "--noproxy", "*", "--max-time", "15", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{port}{path}"])


def main() -> int:
    start = json.loads(START.read_text(encoding="utf-8"))
    pid = int(start["pid"])
    port = 18081
    command_text = " ".join(str(x) for x in start.get("command", []))
    before = pid_snapshot(pid)
    ownership_ok = (
        before["returncode"] == 0
        and str(pid) in str(before.get("stdout", ""))
        and "--port 18081" in str(before.get("stdout", ""))
        and "nemotron-3.5-super-vl-120b-a12b-local-thinking-on" in str(before.get("stdout", ""))
        and "--port 8080" not in str(before.get("stdout", ""))
    )
    formal_before = {"health": health(8080, "/health"), "models": health(8080, "/models")}
    kill_result: dict[str, object]
    if ownership_ok:
        try:
            os.kill(pid, signal.SIGTERM)
            kill_result = {"attempted": True, "pid": pid, "signal": "SIGTERM", "returncode": 0}
        except ProcessLookupError:
            kill_result = {"attempted": True, "pid": pid, "signal": "SIGTERM", "returncode": 1, "error": "process already absent"}
        except PermissionError as exc:
            kill_result = {"attempted": True, "pid": pid, "signal": "SIGTERM", "returncode": 1, "error": repr(exc)}
    else:
        kill_result = {"attempted": False, "pid": pid, "signal": "SIGTERM", "returncode": 2, "error": "ownership guard failed; no process signal sent"}
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        after = pid_snapshot(pid)
        listener = listener_snapshot(port)
        if after["returncode"] != 0 and listener["returncode"] != 0:
            break
        time.sleep(1)
    after = pid_snapshot(pid)
    listener_after = listener_snapshot(port)
    isolated_health_after = health(port, "/health")
    formal_after = {"health": health(8080, "/health"), "models": health(8080, "/models")}
    launchd = run(["/bin/launchctl", "print", "system/com.scc.llama-server"])
    processes = run(["/bin/zsh", "-lc", "/bin/ps -axo pid,etime,rss,command | /usr/bin/grep '[l]lama-server'"])
    memory = run(["/usr/bin/memory_pressure", "-Q"])
    evidence = {
        "schema_version": "nemotron-thinking-on-cleanup-evidence-v1",
        "captured_at_unix": time.time(),
        "isolated_pid": pid,
        "isolated_port": port,
        "isolated_command_from_start": start.get("command"),
        "command_text_from_start": command_text,
        "ownership_guard": ownership_ok,
        "isolated_process_before_stop": before,
        "formal_8080_before_stop": formal_before,
        "stop_attempt": kill_result,
        "isolated_process_after_stop": after,
        "isolated_listener_after_stop": listener_after,
        "isolated_health_after_stop": isolated_health_after,
        "formal_8080_after_stop": formal_after,
        "formal_launchd_after_stop": launchd,
        "existing_processes_after_stop": processes,
        "memory_after_stop": memory,
        "formal_service_untouched": True,
        "cleanup_pass": bool(ownership_ok and kill_result.get("returncode") == 0 and after["returncode"] != 0 and listener_after["returncode"] != 0 and isolated_health_after.get("stdout") == "000" and formal_after["health"].get("stdout") == "200" and formal_after["models"].get("stdout") == "200"),
    }
    (RUN_DIR / "cleanup_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pid": pid, "ownership_guard": ownership_ok, "stop_returncode": kill_result.get("returncode"), "isolated_process_after_returncode": after.get("returncode"), "isolated_listener_after_returncode": listener_after.get("returncode"), "isolated_health_after": isolated_health_after.get("stdout"), "formal_health_after": formal_after["health"].get("stdout"), "formal_models_after": formal_after["models"].get("stdout"), "cleanup_pass": evidence["cleanup_pass"]}, ensure_ascii=False))
    return 0 if evidence["cleanup_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
