#!/usr/bin/env python3
"""Collect a credential-safe, read-only endpoint/model inventory for the pilot.

Only aliases, provider labels, sanitized API bases, file basenames, status and
environment-variable *names/presence* are persisted.  Values of credential
variables and secret-valued config keys are never read into the output.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import socket
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ACTIVE_MODELS = WORKSPACE / "models.ini"
BENCH_CONFIG = WORKSPACE / "translation_benchmark_20260916_luna"
OUTPUT = ROOT / "endpoint_inventory.redacted.json"


def safe_api_base(value: str) -> str:
    """Keep scheme/host/port/path but drop query, fragment, userinfo."""

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return "<invalid-or-redacted>"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "<non-http-or-redacted>"
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def file_entry(value: str) -> dict[str, object]:
    path = Path(value.strip()).expanduser()
    return {"basename": path.name, "exists": path.exists(), "is_file": path.is_file()}


def parse_models_ini(path: Path) -> list[dict[str, object]]:
    # llama-server's preset file is INI-shaped but contains repeated/unknown
    # options that ConfigParser may reject.  Extract only the allowlisted
    # fields with a tiny line parser and ignore every other option.
    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for raw in lines:
        section = re.match(r"^\s*\[([^]]+)\]\s*$", raw)
        if section:
            if current:
                rows.append(current)
            current = {"alias": section.group(1), "provider": "llama.cpp-local"}
            continue
        if current is None:
            continue
        match = re.match(r"^\s*(model|model_path|mmproj)\s*=\s*(.*?)\s*$", raw, re.I)
        if match:
            current[match.group(1).lower()] = file_entry(match.group(2))
    if current:
        rows.append(current)
    return rows


def parse_litellm_audit(path: Path) -> list[dict[str, object]]:
    """Extract only model_name/model/api_base lines; never emit secret keys."""

    rows: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"^\s*-\s+model_name:\s*", raw):
            if current:
                rows.append(current)
            current = {"provider": "litellm-config-candidate", "live_status": "not_verified"}
            current["model_name"] = raw.split(":", 1)[1].strip().strip("'\"")
            continue
        if current is None:
            continue
        match = re.match(r"^\s+(model|api_base):\s*(.*?)\s*$", raw)
        if match:
            key, value = match.groups()
            if key == "api_base":
                current[key] = safe_api_base(value.strip("'\""))
            else:
                current[key] = value.strip("'\"")
    if current:
        rows.append(current)
    return rows


def get_json(url: str) -> tuple[str, object]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return "ok", payload
    except urllib.error.HTTPError as exc:
        return f"http_{exc.code}", None
    except Exception as exc:  # noqa: BLE001 - inventory records endpoint status only
        return type(exc).__name__, None


def local_inventory() -> dict[str, object]:
    health_status, health = get_json("http://127.0.0.1:8080/health")
    models_status, models = get_json("http://127.0.0.1:8080/v1/models")
    model_rows = []
    if isinstance(models, dict) and isinstance(models.get("data"), list):
        for item in models["data"]:
            if not isinstance(item, dict):
                continue
            row = {"id": item.get("id"), "object": item.get("object")}
            for key in ("status", "modalities", "context_length", "owned_by"):
                value = item.get(key)
                if isinstance(value, (str, int, float, bool, list)):
                    row[key] = value
            model_rows.append(row)
    return {
        "health": {"status": health_status, "payload_status": health.get("status") if isinstance(health, dict) else None},
        "models": {"status": models_status, "data": model_rows},
    }


def main() -> None:
    env_patterns = re.compile(r"(OPENAI|LITELLM|GEMINI|MISTRAL|QWEN|LLAMA|API|TOKEN|SECRET|KEY)", re.I)
    env_presence = [{"name": name, "present": bool(os.environ.get(name))} for name in sorted(os.environ) if env_patterns.search(name)]
    historical = []
    for path in sorted(WORKSPACE.glob("models.ini*")):
        if path.name == "models.ini":
            continue
        sections = [row["alias"] for row in parse_models_ini(path) if "mistral" in str(row["alias"]).lower()]
        if sections:
            historical.append({"file": path.name, "aliases": sections})
    litellm_path = BENCH_CONFIG / "litellm.audit.yaml"
    output = {
        "inventory_version": "redacted-v1",
        "captured_local_date": "2026-10-22",
        "credential_policy": "Only environment variable names/presence are captured; no credential values or secret-valued config fields are persisted.",
        "active_local_config": {"path": str(ACTIVE_MODELS), "provider": "llama.cpp-local", "aliases": parse_models_ini(ACTIVE_MODELS)},
        "local_openai_compatible": local_inventory(),
        "candidate_litellm_config": {"path": str(litellm_path), "present": litellm_path.exists(), "provider": "LiteLLM-proxy-candidate", "live_status": "not_found_or_not_verified", "models": parse_litellm_audit(litellm_path) if litellm_path.exists() else []},
        "historical_mistral_evidence": {"active_alias_present": any("mistral" in str(row["alias"]).lower() for row in parse_models_ini(ACTIVE_MODELS)), "historical_files": historical, "exact_local_candidate": {"alias": "mistral-medium-3.5-128b", "provider": "llama.cpp-local-or-router", "model_basename": "Mistral-Medium-3.5-128B-UD-Q4_K_XL-00001-of-00003.gguf", "directory_exists": (WORKSPACE / "Mistral-Medium-3.5-128B").is_dir(), "live_endpoint_status": "not_available_in_active_router"}},
        "docker": {"docker_binary_present": shutil.which("docker") is not None, "docker_socket_present": Path("/benchmark-user/.docker/run/docker.sock").exists(), "live_litellm_container_verified": False},
        "credential_env_presence": env_presence,
        "network_scope": {"external_litellm_probe": "not performed; no configured live URL", "local_probe": "127.0.0.1:8080 only"},
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "active_aliases": len(output["active_local_config"]["aliases"]), "local_model_rows": len(output["local_openai_compatible"]["models"]["data"]), "litellm_candidate_rows": len(output["candidate_litellm_config"]["models"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
