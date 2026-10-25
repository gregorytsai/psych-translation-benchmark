#!/usr/bin/env python3
"""Read-only, credential-safe LiteLLM model discovery.

The token is read only from the environment variable selected by
``configure_litellm_credentials.py --exec-env``.  Responses are never saved
verbatim: only allow-listed identity and capability fields are emitted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


SAFE_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
SAFE_SCALAR = (str, int, float, bool)
IDENTITY_KEYS = {
    "id",
    "model",
    "model_name",
    "model_group",
    "owned_by",
    "provider",
    "family",
    "version",
    "object",
}
CAPABILITY_KEYS = {
    "architecture",
    "context_length",
    "context_window",
    "max_input_tokens",
    "max_output_tokens",
    "max_tokens",
    "supports_chat_completions",
    "supports_function_calling",
    "supports_parallel_function_calling",
    "supports_reasoning",
    "supports_response_schema",
    "supports_system_message",
    "supports_tool_choice",
    "supports_tools",
    "supports_vision",
    "supports_native_streaming",
    "supports_prompt_caching",
    "mode",
}


def safe_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base URL must be an http(s) URL without userinfo")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _safe_value(value: Any) -> Any:
    if isinstance(value, SAFE_SCALAR):
        return value
    if isinstance(value, list) and len(value) <= 32 and all(isinstance(item, SAFE_SCALAR) for item in value):
        return value
    return None


def safe_model_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in IDENTITY_KEYS | CAPABILITY_KEYS:
        value = _safe_value(record.get(key))
        if value is not None:
            row[key] = value
    nested = record.get("model_info")
    if isinstance(nested, Mapping):
        nested_identity: dict[str, Any] = {}
        nested_capability: dict[str, Any] = {}
        for key in IDENTITY_KEYS:
            value = _safe_value(nested.get(key))
            if value is not None:
                nested_identity[key] = value
        for key in CAPABILITY_KEYS:
            value = _safe_value(nested.get(key))
            if value is not None:
                nested_capability[key] = value
        if nested_identity:
            row["model_info_identity"] = nested_identity
        if nested_capability:
            row["model_info_capability"] = nested_capability
    return row


def lineage_hint(row: Mapping[str, Any]) -> dict[str, str]:
    """Return a non-authoritative name hint, never a verified family label."""

    values = " ".join(str(row.get(key, "")) for key in ("id", "model", "model_name", "owned_by")).lower()
    for needle, label in (
        ("mistral", "mistral"),
        ("qwen", "qwen"),
        ("claude", "claude"),
        ("deepseek", "deepseek"),
        ("gemini", "gemini"),
        ("gemma", "gemma"),
        ("gpt", "gpt"),
    ):
        if needle in values:
            return {"hint": label, "source": "model-name/provider heuristic; not verified lineage"}
    return {"hint": "unknown", "source": "no safe name/provider hint"}


def request_json(url: str, token: str, timeout: float) -> tuple[int | None, str, Any | None]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            payload = json.loads(response.read().decode("utf-8")) if content_type == "application/json" else None
            return response.status, content_type, payload
    except urllib.error.HTTPError as exc:
        # Do not read the error body: it may contain deployment details.
        return exc.code, exc.headers.get("Content-Type", "").split(";", 1)[0].strip().lower(), None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None, "unavailable", None


def _records(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, Mapping)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        return [payload]
    return []


def discover(base_url: str, env_name: str, timeout: float) -> dict[str, Any]:
    if not SAFE_ENV_NAME.fullmatch(env_name):
        raise ValueError("--auth-env must be an uppercase environment variable name")
    token = os.environ.get(env_name, "")
    if not token:
        raise ValueError(f"credential environment variable {env_name} is absent")
    base = safe_base_url(base_url)
    # Keep only safe endpoint identity and response status.  The token is not
    # put into the result and is dropped before returning to the caller.
    endpoints: dict[str, Any] = {}
    for label, path in (("health", "/health"), ("models", "/v1/models"), ("model_info", "/model/info")):
        status, content_type, payload = request_json(base + path, token, timeout)
        entry: dict[str, Any] = {"http_status": status, "content_type": content_type}
        if label == "health" and isinstance(payload, Mapping):
            safe_status = _safe_value(payload.get("status"))
            if safe_status is not None:
                entry["payload_status"] = safe_status
        else:
            rows = []
            for record in _records(payload):
                safe_row = safe_model_record(record)
                if safe_row:
                    safe_row["lineage_hint"] = lineage_hint(safe_row)
                    rows.append(safe_row)
            entry["models"] = rows
        endpoints[label] = entry
    del token
    models = endpoints.get("models", {}).get("models", [])
    return {
        "schema_version": "litellm-safe-discovery-v1",
        "base_url": base,
        "auth": {"env_name": env_name, "value_persisted": False},
        "endpoints": endpoints,
        "candidate_count": len(models) if isinstance(models, list) else 0,
        "notes": [
            "Only allow-listed identity/capability fields were persisted; raw responses and credential values were not saved.",
            "lineage_hint is a name/provider heuristic, not a verified family or independence claim.",
            "Use only model IDs returned by the authenticated /v1/models response; do not infer aliases from config candidates.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://140.136.192.221:4000", help="LiteLLM gateway root URL")
    parser.add_argument("--auth-env", required=True, help="environment variable name containing the token")
    parser.add_argument("--output", required=True, help="safe JSON output path; raw responses are never written")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)
    try:
        result = discover(args.base_url, args.auth_env, args.timeout)
        output = Path(args.output).expanduser().absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", "output": str(output), "candidate_count": result["candidate_count"]}, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
