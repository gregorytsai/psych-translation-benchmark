#!/usr/bin/env python3
"""Persistent, local-only driver for the frozen 2026-10-23 translation judges.

This runner deliberately has two gates.  ``--preflight`` only inventories the
new packet interface and local model/runtime inputs.  ``--run`` (or
``--wait-for-approval``) requires the protocol worker's frozen manifest and an
explicit ``ROOT_APPROVED`` marker before it can send a judge request.  It never
uses the formal 8080 service, LiteLLM, Open WebUI, or any prior raw bundle.

The driver owns only ``judge_evaluation/local_runner`` and
``judge_evaluation/raw/local``.  Every request attempt is a new raw file; an
optional format-only repair is a second file and can never replace the first.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/benchmark")
BUNDLE = ROOT / "translation_benchmark_20261023_new_materials"
JUDGE_ROOT = BUNDLE / "judge_evaluation"
RUNNER_ROOT = JUDGE_ROOT / "local_runner"
RAW_ROOT = JUDGE_ROOT / "raw" / "local"
SOURCE_CSV = BUNDLE / "results" / "translation_results.csv"
CONFIG_PATH = RUNNER_ROOT / "judge_config.json"
PROGRESS_PATH = JUDGE_ROOT / "progress_local.md"
# The legacy single-state names are retained only for backwards-readable
# provenance.  New runs never write them: each model has its own checkpoint
# and append-only results file so a restart cannot clobber an earlier judge.
STATE_PATH = RUNNER_ROOT / "driver_state.json"
RESULTS_JSONL = RUNNER_ROOT / "local_results.jsonl"
SUMMARY_PATH = RUNNER_ROOT / "local_summary.json"
CHECKPOINT_DIR = RUNNER_ROOT / "checkpoints"
RESULTS_DIR = RUNNER_ROOT / "results"
SCHEDULER_PATH = RUNNER_ROOT / "scheduler.json"
ANALYSIS_RUNNER = JUDGE_ROOT / "analysis_runner" / "aggregate_new_panel.py"
ANALYSIS_REFRESH_EVERY = 5
HANDOFF_PATH = RUNNER_ROOT / "active_handoff.json"
SCHEMA_DIR = JUDGE_ROOT / "public" / "schemas"
POINTWISE_SCHEMA_PATH = SCHEMA_DIR / "pointwise_result.schema.json"
PAIRWISE_SCHEMA_PATH = SCHEMA_DIR / "pairwise_result.schema.json"
SERVER = ROOT / "llama-v0.7.0" / "build-release" / "bin" / "llama-server"
EXPECTED_BUILD = "b524-620d2225c"
EXPECTED_BUILD_NUMBER = "524"
EXPECTED_COMMIT = "620d2225c"
FORMAL_BASE = "http://127.0.0.1:8080"
SEED = 20261023
GIB = 1024**3
APPROVAL_TEXT = "ROOT_APPROVED"
CONTROL_TOKEN_RE = re.compile(
    r"(?:<\|im_start\|>|<\|im_end\|>|<\|im_start\|>|<\|im_end\|>|"
    r"<\|start\|>|<\|end\|>|<\|turn\|>|<\|channel\|>|<channel\|>|"
    r"<start_of_turn>|<end_of_turn>|<start_of_image>|<end_of_image>|"
    r"<\|image\|>|<\|image>|<image\|>|\[INST\]|\[/INST\]|"
    r"\[MODEL_SETTINGS\]|\[/MODEL_SETTINGS\]|\[THINK\]|\[/THINK\])"
)
ALLOWED_FINISH = {"stop", "eos", "end_turn", "completed", "complete"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(json_bytes(value))


def atomic_write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value.encode("utf-8") if isinstance(value, str) else value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def serialized_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_immutable(path: Path, value: Any) -> None:
    """Write a raw/evidence artifact once; never silently replace an attempt."""
    payload = serialized_json(value)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(f"immutable artifact already exists with different bytes: {path}")
        return
    atomic_write(path, payload)


def checkpoint_path(model_id: str) -> Path:
    return CHECKPOINT_DIR / f"{model_id}.checkpoint.json"


def results_path(model_id: str) -> Path:
    return RESULTS_DIR / f"{model_id}.results.jsonl"


def model_summary_path(model_id: str) -> Path:
    return SUMMARY_PATH.with_name(f"{model_id}.summary.json")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def redact(value: Any) -> Any:
    """Redact local image bytes while keeping request hashes and image hashes."""
    if isinstance(value, dict):
        out = {key: redact(item) for key, item in value.items()}
        image_url = out.get("image_url")
        if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
            url = image_url["url"]
            out["image_url"] = {
                "url": "<redacted-local-image-data-url>",
                "data_url_sha256": sha256_bytes(url.encode("utf-8")),
                "data_url_bytes": len(url.encode("utf-8")),
            }
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def http_json(method: str, url: str, payload: Any | None = None, timeout: float = 30.0) -> dict[str, Any]:
    started = time.monotonic()
    started_at = now()
    request_bytes = json_bytes(payload) if payload is not None else b""
    request_hash = sha256_bytes(request_bytes)
    body: Any = None
    status: int | None = None
    error: str | None = None
    response_bytes = b""
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=request_bytes or None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            response_bytes = response.read()
            try:
                body = json.loads(response_bytes.decode("utf-8"))
            except Exception:
                body = response_bytes.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        response_bytes = exc.read()
        try:
            body = json.loads(response_bytes.decode("utf-8"))
        except Exception:
            body = response_bytes.decode("utf-8", errors="replace")
        error = f"HTTPError:{exc.code}"
    except Exception as exc:  # transport evidence is retained
        error = f"{type(exc).__name__}:{exc}"
    elapsed = time.monotonic() - started
    return {
        "method": method,
        "url": url,
        "started_at": started_at,
        "finished_at": now(),
        "elapsed_s": elapsed,
        "http_status": status,
        "request_sha256": request_hash,
        "response_body_sha256": sha256_bytes(response_bytes),
        "request_redacted": redact(payload),
        "response_body": body,
        "error": error,
    }


def formal_health_evidence() -> dict[str, Any]:
    """Read formal health only; no formal state is changed."""
    return {path: http_json("GET", FORMAL_BASE + path, timeout=10) for path in ("/health", "/v1/models")}


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_port_closed(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_open(port):
            return True
        time.sleep(0.25)
    return not port_open(port)


def lsof_pids(port: int) -> list[int]:
    try:
        completed = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        if line.strip().isdigit():
            pids.append(int(line.strip()))
    return sorted(set(pids))


def vm_headroom(required_bytes: int, phase: str, load_estimate_bytes: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
        text = completed.stdout
        match = re.search(r"page size of (\d+) bytes", text)
        page_size = int(match.group(1)) if match else 4096
        pages = {
            name.lower().replace(" ", "_"): int(value)
            for name, value in re.findall(r"^Pages ([^:]+):\s+(\d+)", text, flags=re.MULTILINE)
        }
        # Purgeable may overlap inactive.  Use only the conservative disjoint
        # categories requested by the translation protocol.
        available_bytes = page_size * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative"))
        return {
            "checked_at": now(),
            "phase": phase,
            "page_size": page_size,
            "pages": pages,
            "available_bytes_conservative": available_bytes,
            "available_gib_conservative": available_bytes / GIB,
            "load_estimate_bytes": load_estimate_bytes,
            "load_estimate_gib": load_estimate_bytes / GIB,
            "required_bytes": required_bytes,
            "required_gib": required_bytes / GIB,
            "pass": available_bytes >= required_bytes,
        }
    except Exception as exc:
        return {
            "checked_at": now(),
            "phase": phase,
            "error": f"{type(exc).__name__}:{exc}",
            "available_bytes_conservative": None,
            "required_bytes": required_bytes,
            "load_estimate_bytes": load_estimate_bytes,
            "pass": False,
        }


def discover_interface() -> tuple[Path | None, dict[str, Any] | None, list[str]]:
    """Find only the new judge interface; never search old pilot packets."""
    errors: list[str] = []
    exact = [
        JUDGE_ROOT / "manifest.json",
        JUDGE_ROOT / "protocol_manifest.json",
        JUDGE_ROOT / "public_packet_manifest.json",
        JUDGE_ROOT / "packet_manifest.json",
        JUDGE_ROOT / "protocol.json",
    ]
    candidates = [p for p in exact if p.is_file()]
    if not candidates and JUDGE_ROOT.is_dir():
        candidates = sorted(JUDGE_ROOT.glob("*.json"))
    for path in candidates:
        try:
            data = read_json(path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}:{exc}")
            continue
        if isinstance(data, dict) and any(key in data for key in ("packet_path", "public_packet", "pointwise", "expected_pointwise_count", "abba_count", "job_counts", "tracks", "root_approval_marker")):
            return path, data, errors
    return None, None, errors


def approval_state(manifest_path: Path | None, manifest: dict[str, Any] | None) -> dict[str, Any]:
    marker_candidates: list[Path] = []
    if isinstance(manifest, dict):
        for key in ("approval_marker", "root_approved_marker", "marker_path"):
            value = manifest.get(key)
            if value:
                path = Path(str(value))
                marker_candidates.append(path if path.is_absolute() else (manifest_path.parent / path if manifest_path else JUDGE_ROOT / path))
    marker_candidates.extend([JUDGE_ROOT / "ROOT_APPROVED", JUDGE_ROOT / "ROOT_APPROVED.json", JUDGE_ROOT / "root_approved.json"])
    unique: list[Path] = []
    for path in marker_candidates:
        if path not in unique:
            unique.append(path)
    found: list[dict[str, Any]] = []
    for path in unique:
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            parsed: Any = None
            try:
                parsed = json.loads(raw)
            except Exception:
                pass
            approved = APPROVAL_TEXT in raw or (isinstance(parsed, dict) and parsed.get("root_approved") is True)
            manifest_match = None
            if isinstance(parsed, dict) and parsed.get("manifest_sha256") and manifest_path and manifest_path.is_file():
                manifest_match = parsed.get("manifest_sha256") == sha256_file(manifest_path)
                approved = approved and manifest_match
            found.append({"path": str(path), "sha256": sha256_file(path), "approved": approved, "manifest_sha256_match": manifest_match, "content": parsed if parsed is not None else raw.strip()})
        except Exception as exc:
            found.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}", "approved": False})
    manifest_approved = bool(isinstance(manifest, dict) and manifest.get("root_approved") is True)
    return {"manifest_path": str(manifest_path) if manifest_path else None, "manifest_root_approved": manifest_approved, "markers": found, "approved": manifest_approved or any(item.get("approved") for item in found)}


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def load_slot_packet(manifest_path: Path, manifest: dict[str, Any], slot: str) -> tuple[Path | None, Any | None, str | None]:
    """Load one public slot's pointwise/pairwise JSONL packet.

    The new packet builder intentionally keeps the five public judge slots in
    separate files. Local bindings are fixed by the root task: Qwen=judge-01,
    Nemotron=judge-02, Mistral=judge-03.
    """
    tracks = manifest.get("tracks") if isinstance(manifest, dict) else None
    if isinstance(tracks, dict):
        point_section = tracks.get("pointwise") if isinstance(tracks.get("pointwise"), dict) else {}
        pair_section = tracks.get("pairwise") if isinstance(tracks.get("pairwise"), dict) else {}
        point_dir = point_section.get("public_dir")
        pair_dir = pair_section.get("public_dir")
        if isinstance(point_dir, str) and isinstance(pair_dir, str):
            point_path = resolve_path(f"{point_dir}/{slot}.jsonl", manifest_path.parent)
            pair_path = resolve_path(f"{pair_dir}/{slot}.jsonl", manifest_path.parent)
            if point_path.is_file() and pair_path.is_file():
                try:
                    point_rows = [json.loads(line) for line in point_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    pair_rows = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    return point_path, {"packet_id": manifest.get("packet_id"), "pointwise": point_rows, "pairwise": pair_rows}, None
                except Exception as exc:
                    return point_path, None, f"slot packet load failed: {type(exc).__name__}:{exc}"
            return point_path, None, f"slot packet missing: {point_path} or {pair_path}"
    return load_packet(manifest_path, manifest)


def load_packet(manifest_path: Path, manifest: dict[str, Any]) -> tuple[Path | None, Any | None, str | None]:
    values = [manifest.get(key) for key in ("packet_path", "public_packet", "packet", "pointwise_packet")]
    packet_path: Path | None = None
    for value in values:
        if isinstance(value, str) and value:
            candidate = resolve_path(value, manifest_path.parent)
            if candidate.is_file():
                packet_path = candidate
                break
    if packet_path is None:
        # Permit the common sibling names but do not look in the old pilot.
        for candidate in (manifest_path.parent / "public_packet.json", manifest_path.parent / "judge_packet.json", manifest_path.parent / "packet.json"):
            if candidate.is_file():
                packet_path = candidate
                break
    if packet_path is None:
        return None, None, "packet path missing from frozen manifest"
    try:
        if packet_path.suffix.lower() == ".jsonl":
            data = [json.loads(line) for line in packet_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            data = read_json(packet_path)
        return packet_path, data, None
    except Exception as exc:
        return packet_path, None, f"packet load failed: {type(exc).__name__}:{exc}"


def load_instructions(manifest_path: Path, manifest: dict[str, Any]) -> tuple[Path | None, str | None, str | None]:
    values = [manifest.get(key) for key in ("instructions_path", "judge_instructions", "instructions")]
    for value in values:
        if isinstance(value, str) and value:
            path = resolve_path(value, manifest_path.parent)
            if path.is_file():
                return path, path.read_text(encoding="utf-8"), None
    inline = manifest.get("instructions_text")
    if isinstance(inline, str) and inline:
        return None, inline, None
    public_default = manifest_path.parent / "public" / "instructions.md"
    if public_default.is_file():
        return public_default, public_default.read_text(encoding="utf-8"), None
    return None, None, "instructions path/text missing from frozen manifest"


def candidate_value(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def image_path_value(item: dict[str, Any]) -> str | None:
    value = candidate_value(item, "image_path", "source_image", "image", "image_file", "image_reference", "image_alias")
    if isinstance(value, dict):
        value = value.get("path") or value.get("file")
    return str(value) if isinstance(value, str) and value else None


def source_value(item: dict[str, Any]) -> str:
    value = candidate_value(item, "source_en", "source", "source_text", "english_source", "source_text_en")
    return str(value) if value is not None else ""


def translation_value(item: dict[str, Any]) -> str:
    value = candidate_value(item, "target_zh_tw", "translation_zh_tw", "translation", "candidate_translation", "target", "text")
    return str(value) if value is not None else ""


def blind_id_value(item: dict[str, Any], default: str) -> str:
    value = candidate_value(item, "blind_id", "candidate_id", "candidate", "id", "model_id")
    return str(value) if value is not None else default


def image_hash_value(item: dict[str, Any]) -> str | None:
    value = candidate_value(item, "image_sha256", "image_hash", "source_image_sha256")
    return str(value) if value else None


def normalize_pointwise(packet: Any) -> list[dict[str, Any]]:
    if isinstance(packet, list):
        section: Any = packet
    elif isinstance(packet, dict):
        section = packet.get("pointwise")
        if isinstance(section, dict):
            section = section.get("items") or section.get("tasks") or section.get("rows")
        if section is None:
            section = packet.get("pointwise_items") or packet.get("items")
    else:
        section = None
    if not isinstance(section, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(section):
        if not isinstance(raw, dict):
            continue
        candidates = raw.get("candidates")
        if isinstance(candidates, list):
            for cindex, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    continue
                merged = dict(raw)
                merged.pop("candidates", None)
                merged.update(candidate)
                merged["task_id"] = str(raw.get("task_id") or raw.get("item_id") or f"pointwise-{index+1:03d}") + "__" + blind_id_value(candidate, f"cand-{cindex+1:02d}")
                merged["track"] = "pointwise"
                rows.append(merged)
        else:
            merged = dict(raw)
            merged["task_id"] = str(raw.get("task_id") or raw.get("job_id") or raw.get("item_id") or f"pointwise-{index+1:03d}")
            merged["track"] = "pointwise"
            rows.append(merged)
    return rows


def normalize_pairwise(packet: Any) -> list[dict[str, Any]]:
    section: Any = packet.get("pairwise") if isinstance(packet, dict) else None
    if isinstance(section, dict):
        section = section.get("tasks") or section.get("items") or section.get("pairs")
    if section is None and isinstance(packet, dict):
        section = packet.get("abba") or packet.get("pairwise_tasks")
    if not isinstance(section, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(section):
        if not isinstance(raw, dict):
            continue
        orientations = raw.get("orientations")
        if isinstance(orientations, list):
            for orientation in orientations:
                if isinstance(orientation, dict):
                    merged = dict(raw)
                    merged.pop("orientations", None)
                    merged.update(orientation)
                    merged["task_id"] = str(merged.get("task_id") or merged.get("pair_job_id") or merged.get("pair_id") or f"pairwise-{index+1:03d}")
                    merged["track"] = "pairwise"
                    rows.append(merged)
            continue
        merged = dict(raw)
        merged["task_id"] = str(raw.get("task_id") or raw.get("pair_job_id") or raw.get("pair_id") or raw.get("item_id") or f"pairwise-{index+1:03d}")
        merged["track"] = "pairwise"
        rows.append(merged)
    return rows


def normalize_interface(manifest_path: Path, manifest: dict[str, Any], packet: Any) -> dict[str, Any]:
    pointwise = normalize_pointwise(packet)
    pairwise = normalize_pairwise(packet)
    expected_point = manifest.get("expected_pointwise_count", manifest.get("pointwise_count"))
    expected_abba = manifest.get("expected_abba_count", manifest.get("pairwise_count", manifest.get("abba_count")))
    if expected_point is None and isinstance(packet, dict):
        expected_point = (packet.get("pointwise") or {}).get("count") if isinstance(packet.get("pointwise"), dict) else None
    if expected_abba is None and isinstance(packet, dict):
        expected_abba = (packet.get("pairwise") or {}).get("count") if isinstance(packet.get("pairwise"), dict) else None
    job_counts = manifest.get("job_counts") if isinstance(manifest.get("job_counts"), dict) else {}
    if expected_point is None:
        expected_point = job_counts.get("pointwise_jobs_per_judge")
    if expected_abba is None:
        expected_abba = job_counts.get("pairwise_jobs_per_judge")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_id": manifest.get("packet_id") or (packet.get("packet_id") if isinstance(packet, dict) else None),
        "packet_sha256": None,
        "expected_pointwise": int(expected_point) if str(expected_point).isdigit() else None,
        "expected_abba": int(expected_abba) if str(expected_abba).isdigit() else None,
        "pointwise": pointwise,
        "pairwise": pairwise,
    }


def load_source_csv() -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row.get("case_id", ""), []).append(row)
    return {"path": str(SOURCE_CSV), "sha256": sha256_file(SOURCE_CSV), "rows": len(rows), "by_case": by_case}


def model_total_bytes(model_path: Path) -> int:
    parent = model_path.parent
    total = 0
    for path in parent.glob("*.gguf"):
        if path.name.startswith("mmproj") or path.name.startswith("mtp"):
            continue
        # Mistral/Qwen/Nemotron split shards all live beside the first shard.
        total += path.stat().st_size
    return total or model_path.stat().st_size


def default_model_configs() -> list[dict[str, Any]]:
    return [
        {
            "id": "qwen3.8-flash-next-xhigh",
            "display_name": "Qwen3.8 Flash Next",
            "family": "qwen",
            "alias": "judge-new-qwen3.8-flash-next-xhigh",
            "port": 18201,
            "model": str(ROOT / "Qwen3.8-Flash-Next-UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf"),
            "mmproj": str(ROOT / "Qwen3.8-Flash-Next-UD-Q6_K_XL/mmproj-F16.gguf"),
            "vision_expected": True,
            "settings": {"reasoning": "on", "reasoning_format": "deepseek", "reasoning_budget": 8192, "reasoning_effort": "xhigh", "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False, "reasoning_effort": "xhigh"}},
        },
        {
            "id": "nemotron3.5-super-vl-120b-a12b-thinking",
            "display_name": "Nemotron3.5 Super VL 120B A12B",
            "family": "nemotron",
            "alias": "judge-new-nemotron3.5-super-vl-120b-a12b-thinking",
            "port": 18202,
            "model": str(ROOT / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/UD-Q6_K_XL/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL-00001-of-00004.gguf"),
            "mmproj": str(ROOT / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-Q6_K_XL/mmproj-BF16.gguf"),
            "vision_expected": True,
            "settings": {"reasoning": "on", "reasoning_format": "deepseek", "reasoning_budget": 8192, "reasoning_effort": None, "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": False}},
        },
        {
            "id": "mistral-medium-3.5-128b-high",
            "display_name": "Mistral Medium 3.5 128B",
            "family": "mistral",
            "alias": "judge-new-mistral-medium-3.5-128b-high",
            "port": 18203,
            "model": str(ROOT / "Mistral-Medium-3.5-128B/Mistral-Medium-3.5-128B-UD-Q4_K_XL-00001-of-00003.gguf"),
            "mmproj": None,
            "vision_expected": False,
            "settings": {"reasoning": "auto", "reasoning_format": "deepseek", "reasoning_budget": 8192, "reasoning_effort": "high", "chat_template_kwargs": {"preserve_reasoning": False}},
        },
    ]


def model_inventory(model: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(model["model"])
    mmproj_value = model.get("mmproj")
    mmproj_path = Path(mmproj_value) if mmproj_value else None
    errors: list[str] = []
    if not SERVER.is_file() or not os.access(SERVER, os.X_OK):
        errors.append(f"runtime missing or not executable: {SERVER}")
    if not model_path.is_file():
        errors.append(f"model missing: {model_path}")
    if mmproj_path is not None and not mmproj_path.is_file():
        errors.append(f"mmproj missing: {mmproj_path}")
    model_bytes = model_total_bytes(model_path) if model_path.is_file() else 0
    mmproj_bytes = mmproj_path.stat().st_size if mmproj_path and mmproj_path.is_file() else 0
    return {
        "model_path": str(model_path),
        "model_sha256_first_shard": sha256_file(model_path) if model_path.is_file() else None,
        "model_total_bytes": model_bytes,
        "model_total_gib": model_bytes / GIB,
        "mmproj_path": str(mmproj_path) if mmproj_path else None,
        "mmproj_sha256": sha256_file(mmproj_path) if mmproj_path and mmproj_path.is_file() else None,
        "mmproj_bytes": mmproj_bytes,
        "mmproj_gib": mmproj_bytes / GIB,
        "load_estimate_bytes": model_bytes + mmproj_bytes,
        "errors": errors,
        "vision_expected": bool(model.get("vision_expected")),
    }


def server_command(model: dict[str, Any]) -> list[str]:
    settings = model["settings"]
    command = [
        str(SERVER), "--model", model["model"], "--alias", model["alias"],
        "--host", "127.0.0.1", "--port", str(model["port"]), "--jinja",
        "--no-webui", "--offline", "--n-gpu-layers", "all", "--flash-attn", "on",
        "--ctx-size", "32768", "--parallel", "1", "--no-cont-batching",
        "--no-cache-prompt", "--spec-type", "none", "--load-mode", "mmap",
        "--reasoning", str(settings["reasoning"]), "--reasoning-format", str(settings["reasoning_format"]),
        "--reasoning-budget", str(settings["reasoning_budget"]), "--no-reasoning-preserve",
        "--log-verbosity", "2",
    ]
    if model.get("mmproj"):
        command.extend(["--mmproj", model["mmproj"]])
    # Mistral's native high selector is request/template-level.  Passing it as
    # a server default is useful provenance but does not replace the request.
    if model["family"] == "mistral":
        command.extend(["--reasoning-effort", "high"])
    return command


def runtime_version() -> str:
    try:
        completed = subprocess.run([str(SERVER), "--version"], capture_output=True, text=True, timeout=30)
        return (completed.stdout + completed.stderr).strip()
    except Exception as exc:
        return f"version_probe_error:{type(exc).__name__}:{exc}"


def runtime_version_passes(version: str) -> bool:
    """Accept the live llama.cpp spelling while asserting build+commit."""
    return bool(re.search(rf"(?:build\s*|b){re.escape(EXPECTED_BUILD_NUMBER)}\b", version, flags=re.IGNORECASE) and EXPECTED_COMMIT in version)


class LocalServer:
    def __init__(self, model: dict[str, Any], batch_id: str):
        self.model = model
        self.batch_id = batch_id
        self.port = int(model["port"])
        self.base = f"http://127.0.0.1:{self.port}"
        self.command = server_command(model)
        self.proc: subprocess.Popen[Any] | None = None
        self.stdout_handle: Any = None
        self.stderr_handle: Any = None
        self.meta: dict[str, Any] = {}
        self.cleanup_evidence: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        inventory = model_inventory(self.model)
        if inventory["errors"]:
            raise RuntimeError("model/runtime inventory failed: " + "; ".join(inventory["errors"]))
        if port_open(self.port):
            raise RuntimeError(f"port guard failed: {self.port} already listening; refusing to touch it")
        load_estimate = int(inventory["load_estimate_bytes"])
        before = vm_headroom(load_estimate + 32 * GIB, "preload", load_estimate)
        if not before.get("pass"):
            raise RuntimeError(f"preload RAM guard failed: available={before.get('available_gib_conservative')}GiB required={(load_estimate + 32*GIB)/GIB}GiB")
        raw_dir = RAW_ROOT / self.model["id"]
        raw_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = raw_dir / f"{self.batch_id}.server.stdout.log"
        stderr_path = raw_dir / f"{self.batch_id}.server.stderr.log"
        self.stdout_handle = stdout_path.open("w", encoding="utf-8")
        self.stderr_handle = stderr_path.open("w", encoding="utf-8")
        started_at = now()
        self.proc = subprocess.Popen(self.command, cwd=str(ROOT), stdout=self.stdout_handle, stderr=self.stderr_handle, start_new_session=True)
        try:
            deadline = time.monotonic() + 1800
            health: dict[str, Any] = {}
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"server exited during load code={self.proc.returncode}")
                health = http_json("GET", self.base + "/health", timeout=5)
                if health.get("http_status") == 200:
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"health timeout at {self.base}")
            listeners = lsof_pids(self.port)
            if self.proc.pid not in listeners or any(pid != self.proc.pid for pid in listeners):
                raise RuntimeError(f"port ownership mismatch: expected PID {self.proc.pid}, listeners={listeners}")
            after = vm_headroom(32 * GIB, "after_load", load_estimate)
            if not after.get("pass"):
                raise RuntimeError(f"after-load RAM guard failed: available={after.get('available_gib_conservative')}GiB")
            endpoints = {path: http_json("GET", self.base + path, timeout=30) for path in ("/health", "/props", "/v1/models", "/slots")}
            version = runtime_version()
            self.meta = {
                "batch_id": self.batch_id,
                "started_at": started_at,
                "pid": self.proc.pid,
                "port": self.port,
                "command": self.command,
                "command_sha256": sha256_json(self.command),
                "server_binary": str(SERVER),
                "server_version": version,
                "expected_build": EXPECTED_BUILD,
                "model": self.model,
                "model_inventory": inventory,
                "headroom_preload": before,
                "headroom_after_load": after,
                "endpoints_pre": endpoints,
                "formal_health_before": formal_health_evidence(),
            }
            write_json_immutable(raw_dir / f"{self.batch_id}.server_preflight.json", redact(self.meta))
            if not runtime_version_passes(version):
                raise RuntimeError(f"runtime version mismatch: expected {EXPECTED_BUILD} observed {version!r}")
            return self.meta
        except Exception:
            self.stop(reason="start_failure")
            raise

    def stop(self, reason: str = "normal") -> dict[str, Any]:
        if self.cleanup_evidence is not None:
            return self.cleanup_evidence
        pid = self.proc.pid if self.proc else None
        guard: dict[str, Any] = {"pid": pid, "port": self.port, "command_sha256": sha256_json(self.command), "reason": reason, "listeners_before": lsof_pids(self.port)}
        if self.proc is not None and self.proc.poll() is None:
            listeners = lsof_pids(self.port)
            if listeners and any(item != self.proc.pid for item in listeners):
                guard["stop_refused"] = f"unexpected listener(s) {listeners}"
                raise RuntimeError(str(guard))
            actual = subprocess.run(["ps", "-p", str(self.proc.pid), "-o", "command="], capture_output=True, text=True, timeout=10).stdout.strip()
            if str(SERVER) not in actual or f"--port {self.port}" not in actual or f"--model {self.model['model']}" not in actual:
                guard["stop_refused"] = "PID command guard mismatch"
                raise RuntimeError(str(guard))
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=60)
            except Exception as exc:
                guard["sigint_error"] = f"{type(exc).__name__}:{exc}"
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    self.proc.wait(timeout=30)
                except Exception as term_exc:
                    guard["sigterm_error"] = f"{type(term_exc).__name__}:{term_exc}"
        guard["returncode"] = self.proc.returncode if self.proc else None
        guard["port_closed"] = wait_port_closed(self.port)
        guard["listeners_after"] = lsof_pids(self.port)
        guard["formal_health_after"] = formal_health_evidence()
        guard["stopped_at"] = now()
        raw_dir = RAW_ROOT / self.model["id"]
        write_json_immutable(raw_dir / f"{self.batch_id}.server_cleanup.json", guard)
        if self.stdout_handle:
            self.stdout_handle.close()
            self.stdout_handle = None
        if self.stderr_handle:
            self.stderr_handle.close()
            self.stderr_handle = None
        self.cleanup_evidence = guard
        return guard


def image_data_url(path: Path) -> str:
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def task_image(task: dict[str, Any], manifest_path: Path) -> tuple[Path | None, str | None]:
    value = image_path_value(task)
    if not value:
        return None, image_hash_value(task)
    path = resolve_path(value, manifest_path.parent)
    if not path.is_file():
        # Packet paths commonly start at the new materials bundle.
        path = resolve_path(value, BUNDLE)
    if not path.is_file():
        # Public packet image aliases are deliberately basename-only.
        path = JUDGE_ROOT / "public" / "images" / Path(value).name
    if not path.is_file():
        return path, image_hash_value(task)
    return path, image_hash_value(task) or sha256_file(path)


def task_is_image(task: dict[str, Any]) -> bool:
    value = str(candidate_value(task, "input_type", "type", "modality") or "").lower()
    return value in {"image", "vision", "image_translation"} or bool(image_path_value(task))


def task_candidate_pair(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    a = task.get("candidate_a") or task.get("a") or task.get("left")
    b = task.get("candidate_b") or task.get("b") or task.get("right")
    if not isinstance(a, dict):
        a = {"blind_id": task.get("candidate_a_id", "cand-A"), "target_zh_tw": task.get("candidate_a_translation", "")}
    if not isinstance(b, dict):
        b = {"blind_id": task.get("candidate_b_id", "cand-B"), "target_zh_tw": task.get("candidate_b_translation", "")}
    return a, b


def build_messages(
    task: dict[str, Any],
    instructions: str,
    manifest_path: Path,
    repair: bool = False,
    repair_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    track = task.get("track", "pointwise")
    source = source_value(task)
    image_path, image_hash = task_image(task, manifest_path)
    image = task_is_image(task)
    schema = frozen_schema_for_track(track)
    schema_text = json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2)
    schema_instruction = "\n\n# Frozen output schema (follow exactly; no extra keys)\n" + schema_text
    system_text = instructions + schema_instruction
    if track == "pairwise":
        a, b = task_candidate_pair(task)
        orientation = str(task.get("orientation") or "AB").upper()
        candidate_a = translation_value(a)
        candidate_b = translation_value(b)
        candidate_a_id = blind_id_value(a, "cand-A")
        candidate_b_id = blind_id_value(b, "cand-B")
        body_text = (
            "Evaluate this single independent pairwise task. The labels are blind and must not be inferred.\n"
            + ("The attached image is the source; judge the candidates against its visible content. Do not transcribe the image or request OCR.\n" if image else f"English source:\n{source}\n")
            + f"\nCandidate A ({candidate_a_id}):\n{candidate_a}\n\nCandidate B ({candidate_b_id}):\n{candidate_b}\n"
            + f"\nThis is frozen orientation {orientation}. Return the pairwise JSON contract only."
        )
    else:
        target = translation_value(task)
        candidate_id = blind_id_value(task, "cand-unknown")
        body_text = (
            "Evaluate this single independent candidate. The label is blind and must not be inferred.\n"
            + ("The attached image is the source; judge the candidate against its visible content. Do not transcribe the image or request OCR.\n" if image else f"English source:\n{source}\n")
            + f"\nCandidate ({candidate_id}):\n{target}\n"
            + "\nReturn the pointwise JSON contract only."
        )
    if repair:
        context = repair_context or {}
        original = str(context.get("original_content") or "")
        specific_errors = context.get("format_errors") or []
        if not isinstance(specific_errors, list):
            specific_errors = [str(specific_errors)]
        protected = "score_0_100, subscores, critical, errors, winner, confidence, critical_override, reason_codes"
        body_text += (
            "\nFORMAT_REPAIR_ONLY: Return exactly one JSON object matching the frozen contract. "
            "This is a format repair, not a fresh evaluation. Preserve the original evaluation. "
            f"Do not change these protected semantic fields: {protected}.\n"
            "<original_assistant_response>\n" + original +
            "\n</original_assistant_response>\n<specific_format_errors>\n- " +
            "\n- ".join(str(item) for item in specific_errors) +
            "\n</specific_format_errors>"
        )
    if image:
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"image path missing for {task.get('task_id')}: {image_path}")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": body_text},
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        ]
        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": content}]
    else:
        messages = [{"role": "system", "content": system_text}, {"role": "user", "content": body_text}]
    metadata = {
        "track": track,
        "task_id": task.get("task_id"),
        "input_type": "image" if image else "text",
        "source_sha256": sha256_bytes(source.encode("utf-8")) if source else None,
        "image_path": str(image_path) if image_path else None,
        "image_sha256": image_hash,
        "one_fresh_source_candidate_per_call": True,
        "english_transcription_sent_for_image": False,
        "repair": repair,
        "schema_path": str(schema_path_for_track(track)),
        "schema_sha256": sha256_file(schema_path_for_track(track)),
        "schema_embedded_in_system_message": True,
        "repair_specific_errors": list((repair_context or {}).get("format_errors") or []) if repair else [],
    }
    return messages, metadata


def request_payload(model: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    settings = model["settings"]
    payload: dict[str, Any] = {
        "model": model["alias"],
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": SEED,
        "max_tokens": 16384,
        "stream": False,
        "cache_prompt": False,
        "response_format": {"type": "json_object"},
        "verbose": True,
        "reasoning_format": settings["reasoning_format"],
        "reasoning_budget_tokens": settings["reasoning_budget"],
        "chat_template_kwargs": settings["chat_template_kwargs"],
    }
    if settings.get("reasoning_effort") is not None:
        payload["reasoning_effort"] = settings["reasoning_effort"]
    return payload


def extract_message(response_body: Any) -> tuple[dict[str, Any], str, str]:
    if not isinstance(response_body, dict):
        raise ValueError("response body is not JSON object")
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response choices missing")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response message missing")
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    if not isinstance(reasoning, str):
        reasoning = "" if reasoning is None else str(reasoning)
    return choice, content, reasoning


def parse_json_content(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def schema_path_for_track(track: str) -> Path:
    if track == "pairwise":
        return PAIRWISE_SCHEMA_PATH
    return POINTWISE_SCHEMA_PATH


def frozen_schema_for_track(track: str) -> dict[str, Any]:
    path = schema_path_for_track(track)
    if not path.is_file():
        raise FileNotFoundError(f"frozen result schema missing: {path}")
    schema = read_json(path)
    if not isinstance(schema, dict):
        raise ValueError(f"frozen result schema is not an object: {path}")
    return schema


def schema_validation_method() -> str:
    try:
        import jsonschema  # type: ignore
        _ = jsonschema
        return "jsonschema.Draft202012Validator"
    except Exception:
        return "manual_frozen_schema_fallback_jsonschema_unavailable"


def _jsonschema_diagnostics(schema: dict[str, Any], parsed: Any) -> list[str]:
    """Use the installed validator when available, while keeping stable codes."""
    try:
        import jsonschema  # type: ignore
        validator = jsonschema.Draft202012Validator(schema)
        diagnostics: list[str] = []
        for error in sorted(validator.iter_errors(parsed), key=lambda item: list(item.path)):
            location = ".".join(str(part) for part in error.path) or "$"
            diagnostics.append(f"jsonschema:{error.validator}:{location}")
        return diagnostics
    except Exception:
        return []


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_contract_details(track: str, parsed: Any, expected_ids: Iterable[str]) -> dict[str, Any]:
    """Validate the exact frozen public schema plus the protocol arithmetic.

    Evidence disagreements (for example ``critical=true`` without a
    critical-severity error) are deliberately diagnostics only.  They must be
    visible to the importer, but must not rewrite or discard a structurally
    valid scalar judgment and must never trigger a retry.
    """
    schema = frozen_schema_for_track(track)
    method = schema_validation_method()
    schema_diagnostics = _jsonschema_diagnostics(schema, parsed)
    errors: list[str] = []
    evidence_diagnostics: list[dict[str, Any]] = []
    # Draft-2020-12 violations are structural.  Evidence-quality diagnostics
    # are produced separately below and never enter this list.
    errors.extend("jsonschema_invalid:" + item for item in schema_diagnostics)

    if not isinstance(parsed, dict):
        errors.append("top_level_not_object")
        return {"valid": False, "error": errors[0], "validation_errors": errors,
                "evidence_diagnostics": evidence_diagnostics,
                "schema_validation_method": method, "jsonschema_diagnostics": schema_diagnostics}

    if track == "pairwise":
        allowed_keys = {"winner", "confidence", "critical_override", "reason_codes"}
        required_keys = allowed_keys
        missing = sorted(required_keys - set(parsed))
        extra = sorted(set(parsed) - allowed_keys)
        if missing:
            errors.append("pairwise_required_missing:" + ",".join(missing))
        if extra:
            errors.append("additional_properties:" + ",".join(extra))
        if parsed.get("winner") not in {"A", "B", "TIE"}:
            errors.append("pairwise_winner_invalid")
        confidence = parsed.get("confidence")
        if not _is_number(confidence):
            errors.append("pairwise_confidence_invalid")
        elif not 0 <= float(confidence) <= 1:
            errors.append("confidence_out_of_range")
        if not isinstance(parsed.get("critical_override"), bool):
            errors.append("pairwise_critical_override_invalid")
        reason_codes = parsed.get("reason_codes")
        if not isinstance(reason_codes, list) or not reason_codes or any(not isinstance(item, str) or not item.strip() for item in reason_codes):
            errors.append("reason_codes_invalid")
    else:
        # The new local packet has no qualification track.  Keeping this
        # explicit prevents accidentally accepting an old pilot envelope.
        if track != "pointwise":
            errors.append("unsupported_track")
        allowed_keys = {"blind_id", "status", "abstain", "score_0_100", "subscores", "critical", "errors", "block_reason"}
        required_keys = {"blind_id", "status", "abstain", "score_0_100", "subscores", "critical", "errors"}
        missing = sorted(required_keys - set(parsed))
        extra = sorted(set(parsed) - allowed_keys)
        if missing:
            errors.append("required_field_missing:" + ",".join(missing))
        if extra:
            errors.append("additional_properties:" + ",".join(extra))
        blind_id = parsed.get("blind_id")
        allowed_ids = set(expected_ids)
        if not isinstance(blind_id, str):
            errors.append("blind_id_invalid")
        elif not re.fullmatch(r"b-[0-9a-f]{12}", blind_id):
            errors.append("blind_id_pattern_invalid")
        elif allowed_ids and blind_id not in allowed_ids:
            errors.append("blind_id_not_in_task")
        status = parsed.get("status")
        if status not in {"ok", "blocked", "missing"}:
            errors.append("status_invalid")
        if not isinstance(parsed.get("abstain"), bool):
            errors.append("abstain_missing_or_invalid")
        if not isinstance(parsed.get("critical"), bool):
            errors.append("critical_missing_or_invalid")
        if not isinstance(parsed.get("errors"), list):
            errors.append("errors_missing")
        else:
            error_keys = {"severity", "type", "target_exact_quote", "source_evidence", "source_span", "target_anchor", "confidence"}
            for index, item in enumerate(parsed["errors"]):
                prefix = f"error_schema:{index}"
                if not isinstance(item, dict):
                    errors.append(prefix + ":not_object")
                    continue
                missing_error = sorted(error_keys - set(item))
                extra_error = sorted(set(item) - error_keys)
                if missing_error:
                    errors.append(prefix + ":required:" + ",".join(missing_error))
                if extra_error:
                    errors.append(prefix + ":additional:" + ",".join(extra_error))
                if item.get("severity") not in {"minor", "major", "critical"}:
                    errors.append(prefix + ":severity_invalid")
                if not isinstance(item.get("type"), str) or not item["type"].strip():
                    errors.append(prefix + ":type_invalid")
                for field in ("target_exact_quote", "source_evidence", "source_span", "target_anchor"):
                    if not isinstance(item.get(field), str):
                        errors.append(prefix + f":{field}_invalid")
                    elif field == "source_evidence" and not item[field].strip():
                        errors.append(prefix + ":source_evidence_empty")
                confidence = item.get("confidence")
                if not _is_number(confidence):
                    errors.append(prefix + ":confidence_invalid")
                elif not 0 <= float(confidence) <= 1:
                    errors.append(prefix + ":confidence_out_of_range")
        status = parsed.get("status")
        if "block_reason" in parsed and not isinstance(parsed.get("block_reason"), str):
            errors.append("block_reason_invalid")
        if status == "ok":
            if parsed.get("abstain") is not False:
                errors.append("ok_abstain_must_be_false")
            score = parsed.get("score_0_100")
            if not isinstance(score, int) or isinstance(score, bool):
                errors.append("score_invalid")
            elif not 0 <= score <= 100:
                errors.append("score_out_of_range")
            subscores = parsed.get("subscores")
            required = ("terminology_locale", "semantic_safety", "fluency", "completeness_format")
            limits = {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 20}
            if not isinstance(subscores, dict):
                errors.append("subscores_missing")
            else:
                missing_subscore = sorted(set(required) - set(subscores))
                extra_subscore = sorted(set(subscores) - set(required))
                if missing_subscore:
                    errors.append("subscores_required_missing:" + ",".join(missing_subscore))
                if extra_subscore:
                    errors.append("additional_properties:subscores." + ",".join(extra_subscore))
                for key in required:
                    value = subscores.get(key)
                    if not isinstance(value, int) or isinstance(value, bool):
                        errors.append(f"subscore_type_invalid:{key}")
                    elif not 0 <= value <= limits[key]:
                        errors.append(f"subscore_out_of_range:{key}")
                if not errors or not any(item.startswith(("subscore_", "subscores_", "additional_properties:subscores.")) for item in errors):
                    if isinstance(score, int) and isinstance(subscores, dict) and all(isinstance(subscores.get(key), int) and not isinstance(subscores.get(key), bool) for key in required):
                        if sum(subscores[key] for key in required) != score:
                            errors.append("subscores_sum_mismatch")
            if isinstance(parsed.get("errors"), list):
                has_critical = any(isinstance(item, dict) and item.get("severity") == "critical" for item in parsed["errors"])
                if isinstance(parsed.get("critical"), bool) and parsed["critical"] != has_critical:
                    evidence_diagnostics.append({"code": "critical_flag_vs_error_severity", "critical": parsed["critical"], "has_critical_severity": has_critical})
        elif status in {"blocked", "missing"}:
            if parsed.get("abstain") is not True:
                errors.append("blocked_abstain_must_be_true")
            if parsed.get("score_0_100") is not None or parsed.get("subscores") is not None:
                errors.append("blocked_score_subscores_must_be_null")
            if not isinstance(parsed.get("block_reason"), str) or not parsed.get("block_reason", "").strip():
                errors.append("block_reason_missing")
            if parsed.get("critical") is not False or parsed.get("errors") != []:
                errors.append("blocked_evidence_contract_invalid")

    return {"valid": not errors, "error": errors[0] if errors else None,
            "validation_errors": errors, "evidence_diagnostics": evidence_diagnostics,
            "schema_validation_method": method, "jsonschema_diagnostics": schema_diagnostics}


def validate_contract(track: str, parsed: Any, expected_ids: Iterable[str]) -> tuple[bool, str | None]:
    details = validate_contract_details(track, parsed, expected_ids)
    return bool(details["valid"]), details["error"]


def evidence_quality_diagnostics(track: str, parsed: Any, task: dict[str, Any]) -> list[dict[str, Any]]:
    """Surface evidence quality without changing a valid scalar judgment."""
    if track != "pointwise" or not isinstance(parsed, dict) or parsed.get("status") != "ok":
        return []
    target = translation_value(task)
    source = source_value(task)
    diagnostics: list[dict[str, Any]] = []
    for index, item in enumerate(parsed.get("errors", [])):
        if not isinstance(item, dict):
            continue
        quote = item.get("target_exact_quote", "")
        anchor = item.get("target_anchor", "")
        source_span = item.get("source_span", "")
        if isinstance(quote, str) and quote and isinstance(target, str) and quote not in target:
            diagnostics.append({"code": "target_exact_quote_not_in_target", "error_index": index})
        if isinstance(anchor, str) and anchor and isinstance(target, str) and anchor not in target:
            diagnostics.append({"code": "target_anchor_not_in_target", "error_index": index})
        if isinstance(quote, str) and quote == "":
            if not isinstance(source_span, str) or not source_span.strip():
                diagnostics.append({"code": "omission_source_span_missing", "error_index": index})
            if not isinstance(anchor, str) or not anchor.strip():
                diagnostics.append({"code": "omission_target_anchor_missing", "error_index": index})
        if source and isinstance(item.get("source_evidence"), str) and item["source_evidence"].strip() and item["source_evidence"].strip() not in source:
            diagnostics.append({"code": "source_evidence_not_exact_source_substring", "error_index": index})
    return diagnostics


def response_metadata(
    response: dict[str, Any],
    model: dict[str, Any],
    request: dict[str, Any],
    tokenize_base: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    body = response.get("response_body")
    error: str | None = None
    metadata: dict[str, Any] = {
        "http_status": response.get("http_status"),
        "finish_reason": None,
        "content": "",
        "reasoning_content": "",
        "content_sha256": None,
        "reasoning_sha256": None,
        "usage": None,
        "timings": None,
        "actual_reasoning_tokens": None,
        "actual_reasoning_tokens_source": None,
        "reasoning_tokenization": None,
        "reasoning_content_present": False,
        "off_mode_violation": False,
    }
    if response.get("http_status") != 200:
        return metadata, response.get("error") or "http_not_200"
    try:
        choice, content, reasoning = extract_message(body)
        metadata["finish_reason"] = choice.get("finish_reason")
        metadata["content"] = content
        metadata["reasoning_content"] = reasoning
        metadata["content_sha256"] = sha256_bytes(content.encode("utf-8"))
        metadata["reasoning_sha256"] = sha256_bytes(reasoning.encode("utf-8"))
        metadata["reasoning_content_present"] = bool(reasoning.strip())
        if choice.get("finish_reason") not in ALLOWED_FINISH:
            error = f"invalid_finish_reason:{choice.get('finish_reason')}"
        usage = body.get("usage") if isinstance(body, dict) else None
        metadata["usage"] = usage
        metadata["timings"] = body.get("timings") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details") or usage.get("completion_tokens_detail") or {}
            for key in ("reasoning_tokens", "thinking_tokens", "reasoning_token_count"):
                if isinstance(details, dict) and isinstance(details.get(key), int):
                    metadata["actual_reasoning_tokens"] = details[key]
                    metadata["actual_reasoning_tokens_source"] = f"usage.completion_tokens_details.{key}"
                    break
            if metadata["actual_reasoning_tokens"] is None:
                for key in ("reasoning_tokens", "thinking_tokens"):
                    if isinstance(usage.get(key), int):
                        metadata["actual_reasoning_tokens"] = usage[key]
                        metadata["actual_reasoning_tokens_source"] = f"usage.{key}"
                        break
        # Usage telemetry in this llama.cpp build may omit a reasoning-token
        # field.  Count the retained reasoning string through the native
        # non-generating endpoint instead of inferring from characters or
        # treating presence as a token count.  The audit is intentionally
        # content-only with parse_special=true; template tag offsets are not
        # silently added to the count.
        if tokenize_base and metadata["actual_reasoning_tokens"] is None:
            token_request = {"content": reasoning, "add_special": False, "parse_special": True}
            token_result = http_json("POST", tokenize_base + "/tokenize", token_request, timeout=120)
            token_body = token_result.get("response_body") if isinstance(token_result.get("response_body"), dict) else {}
            token_list = token_body.get("tokens") if isinstance(token_body, dict) else None
            token_count = len(token_list) if isinstance(token_list, list) else None
            metadata["actual_reasoning_tokens"] = token_count
            metadata["actual_reasoning_tokens_source"] = "non_generating_tokenize_content_only" if token_count is not None else "non_generating_tokenize_failed"
            metadata["reasoning_tokenization"] = {
                "method": "native_endpoint_content_only_parse_special_true",
                "interpretation": "retained_reasoning_content_only; native template tag offsets are not included",
                "request": redact(token_request),
                "request_sha256": sha256_json(token_request),
                "http_envelope": {key: value for key, value in token_result.items() if key not in {"request_redacted", "response_body"}},
                "response_body": token_body,
                "response_body_sha256": token_result.get("response_body_sha256"),
                "token_count": token_count,
                "budget": model["settings"].get("reasoning_budget"),
            }
        # All three local judge routes are configured as thinking-on.  Mistral
        # expresses that mode through its native high selector (and therefore
        # deliberately has no enable_thinking request field), but its native
        # response still must expose reasoning_content.  Do not classify that
        # expected output as an off-mode violation.
        thinking_expected = model["family"] in {"qwen", "nemotron", "mistral"}
        if not thinking_expected and reasoning.strip():
            metadata["off_mode_violation"] = True
            error = error or "off_mode_violation_nonempty_reasoning_content"
        if CONTROL_TOKEN_RE.search(content):
            error = error or "control_token_or_template_marker_in_content"
    except Exception as exc:
        error = f"response_parse:{type(exc).__name__}:{exc}"
    return metadata, error


def template_probe(server: LocalServer, model: dict[str, Any]) -> dict[str, Any]:
    marker = f"JUDGE_TEMPLATE_PROBE_{model['id'].replace('-', '_')}"
    text_payload = request_payload(model, [{"role": "system", "content": marker + "_SYSTEM"}, {"role": "user", "content": marker + "_USER"}])
    text_payload["max_tokens"] = 1
    text_result = http_json("POST", server.base + "/apply-template", {**text_payload, "add_generation_prompt": True}, timeout=60)
    probes = [{"kind": "text", "result": text_result}]
    if model.get("mmproj"):
        image = BUNDLE / "technical_smoke" / "technical_marker.png"
        if image.is_file():
            image_payload = request_payload(model, [{"role": "user", "content": [{"type": "text", "text": marker + "_IMAGE"}, {"type": "image_url", "image_url": {"url": image_data_url(image)}}]}])
            image_payload["max_tokens"] = 1
            probes.append({"kind": "image", "result": http_json("POST", server.base + "/apply-template", {**image_payload, "add_generation_prompt": True}, timeout=120)})
    records: list[dict[str, Any]] = []
    for probe in probes:
        result = probe["result"]
        body = result.get("response_body") if isinstance(result.get("response_body"), dict) else {}
        prompt = body.get("prompt") if isinstance(body, dict) else None
        prompt = prompt if isinstance(prompt, str) else ""
        records.append({
            "kind": probe["kind"],
            "endpoint": result.get("url"),
            "http_envelope": {key: value for key, value in result.items() if key not in {"request_redacted", "response_body"}},
            "request_redacted": result.get("request_redacted"),
            "response_body": body,
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "native_marker_counts": {"probe_marker": prompt.count(marker), "reasoning_effort": prompt.count(str(model["settings"].get("reasoning_effort"))) if model["settings"].get("reasoning_effort") else 0, "image_marker": sum(prompt.count(x) for x in ("<|image>", "<image|>", "<start_of_image>", "[IMG]", "<|image|>"))},
            "prompt_only": True,
        })
    props_result = http_json("GET", server.base + "/props", timeout=30)
    props_body = props_result.get("response_body") if isinstance(props_result.get("response_body"), dict) else {}
    modalities = props_body.get("modalities") if isinstance(props_body, dict) else {}
    architecture = props_body.get("architecture") if isinstance(props_body, dict) else {}
    input_modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else []
    if isinstance(modalities, dict) and "vision" in modalities:
        vision_supported: bool | None = bool(modalities.get("vision"))
    elif isinstance(input_modalities, list) and input_modalities:
        vision_supported = "image" in input_modalities
    else:
        vision_supported = None
    # Some multimodal native templates deliberately use a runtime-generated
    # media sentinel instead of a public-looking ``<image>`` token.  Bind the
    # marker to the same /props envelope used for the vision capability gate;
    # this avoids a false N/A while still requiring the actual apply-template
    # prompt to contain the observed marker.
    media_marker = props_body.get("media_marker") if isinstance(props_body, dict) else None
    if not isinstance(media_marker, str) or not media_marker:
        media_marker = None
    image_marker_tokens = [
        "<|image>", "<image|>", "<|image|>", "<start_of_image>",
        "<|vision_start|>", "<|image_pad|>", "[IMG]", "<image>",
    ]
    if media_marker:
        image_marker_tokens.append(media_marker)
    image_record = next((record for record in records if record.get("kind") == "image"), None)
    image_prompt = str(image_record.get("prompt") or "") if image_record else ""
    image_marker_count = sum(image_prompt.count(token) for token in image_marker_tokens)
    if image_record is not None:
        image_record["native_marker_counts"]["image_marker"] = image_marker_count
        image_record["native_marker_counts"]["image_marker_tokens"] = [
            token for token in image_marker_tokens if image_prompt.count(token) > 0
        ]
        image_record["native_media_marker"] = media_marker
    text_record = next((record for record in records if record.get("kind") == "text"), {})
    text_prompt = str(text_record.get("prompt") or "")
    text_http = text_record.get("http_envelope", {}).get("http_status") == 200
    text_role_pass = text_prompt.count(marker + "_SYSTEM") == 1 and text_prompt.count(marker + "_USER") == 1
    if model["family"] in {"qwen", "nemotron"}:
        thinking_pass = text_prompt.endswith("<think>\n") or text_prompt.endswith("<think>")
    else:
        thinking_pass = text_prompt.count('[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]') == 1
    image_http = image_record.get("http_envelope", {}).get("http_status") == 200 if image_record else False
    image_pass = bool(image_record and image_http and image_prompt and image_marker_count > 0 and (vision_supported is not False))
    checks = {
        "text_http_200": text_http,
        "text_prompt_nonempty": bool(text_prompt),
        "text_exact_role_markers_once": text_role_pass,
        "thinking_mode_native_observed": thinking_pass,
        "props_vision_supported": vision_supported is True if model.get("vision_expected") else vision_supported is False or vision_supported is None,
        "image_probe_native_marker": image_pass if model.get("vision_expected") else True,
    }
    # If an expected vision route is not proven, the driver may continue only
    # with explicit image N/A records; it never grades those image cells.
    evidence = {"captured_at": now(), "model_id": model["id"], "props": {"http_envelope": {key: value for key, value in props_result.items() if key not in {"request_redacted", "response_body"}}, "body": props_body}, "vision_supported": vision_supported, "native_media_marker": media_marker, "probes": records, "checks": checks, "overall_text_pass": all(checks[key] for key in ("text_http_200", "text_prompt_nonempty", "text_exact_role_markers_once", "thinking_mode_native_observed")), "overall_vision_pass": image_pass if model.get("vision_expected") else True}
    write_json_immutable(RAW_ROOT / model["id"] / f"{server.batch_id}.template_probe.json", redact(evidence))
    if not evidence["overall_text_pass"]:
        raise RuntimeError(f"template text gate failed for {model['id']}: {evidence}")
    return evidence


def technical_smoke(server: LocalServer, model: dict[str, Any], vision_supported: bool | None = None) -> dict[str, Any]:
    """Run only synthetic marker requests after approval, before frozen jobs."""
    marker = f"JUDGE_TECHNICAL_SMOKE_{model['id'].replace('-', '_')}"
    records: list[dict[str, Any]] = []
    requests: list[tuple[str, list[dict[str, Any]], str | None]] = [
        ("text", [{"role": "system", "content": "Technical marker only. Do not evaluate a translation."}, {"role": "user", "content": f"Return any short JSON object containing marker {marker}. This is not a frozen source case."}], None),
    ]
    image_path = BUNDLE / "technical_smoke" / "technical_marker.png"
    if (vision_supported is True) and model.get("mmproj") and image_path.is_file():
        requests.append(("image", [{"role": "system", "content": "Technical marker only. Do not evaluate a translation."}, {"role": "user", "content": [{"type": "text", "text": f"Inspect this technical marker image and return any short JSON object containing marker {marker}. This is not a frozen source case."}, {"type": "image_url", "image_url": {"url": image_data_url(image_path)}}]}], sha256_file(image_path)))
    else:
        records.append({"kind": "image", "status": "N/A_not_supported", "reason": "runtime_props_or_template_did_not_prove_native_vision", "vision_supported": vision_supported, "no_frozen_material": True})
    for kind, messages, image_hash in requests:
        payload = request_payload(model, messages)
        result = http_json("POST", server.base + "/v1/chat/completions", payload, timeout=900)
        metadata, error = response_metadata(result, model, payload, tokenize_base=server.base)
        status = "ok"
        if result.get("http_status") != 200:
            status = "invalid"
            error = error or "http_not_200"
        elif metadata.get("finish_reason") not in ALLOWED_FINISH:
            status = "invalid"
            error = error or f"invalid_finish_reason:{metadata.get('finish_reason')}"
        elif not str(metadata.get("content", "")).strip():
            status = "invalid"
            error = error or "empty_final_content"
        elif model["family"] in {"qwen", "nemotron", "mistral"} and (
            not metadata.get("reasoning_content_present")
            or not isinstance(metadata.get("actual_reasoning_tokens"), int)
            or metadata.get("actual_reasoning_tokens") <= 0
        ):
            # Native thinking-on is not proven by a request flag alone.  The
            # first synthetic canary must expose retained reasoning and a
            # non-generating token count before any frozen job is graded.
            status = "invalid"
            error = error or "thinking_native_canary_unproven"
        record = {
            "schema_version": "judge-technical-smoke-v1",
            "recorded_at": now(),
            "model_id": model["id"],
            "kind": kind,
            "status": status,
            "error": error,
            "no_frozen_material": True,
            "image_sha256": image_hash,
            "request": redact(payload),
            "request_sha256": sha256_json(payload),
            "http_envelope": {key: value for key, value in result.items() if key not in {"request_redacted", "response_body"}},
            "response": {"body": result.get("response_body"), "body_sha256": result.get("response_body_sha256")},
            "response_metadata": {key: value for key, value in metadata.items() if key not in {"content", "reasoning_content"}},
            "content": metadata.get("content", ""),
            "reasoning_content": metadata.get("reasoning_content", ""),
            "marker_observed_in_content": marker in str(metadata.get("content", "")),
        }
        records.append(record)
        # A restart must never overwrite the prior smoke evidence.  Scope the
        # artifact to this isolated server batch just like template probes and
        # server envelopes; the stable summary remains batch-scoped below.
        write_json_immutable(RAW_ROOT / model["id"] / "technical_smoke" / f"{server.batch_id}.{kind}.json", record)
    evidence = {"schema_version": "judge-technical-smoke-summary-v1", "checked_at": now(), "model_id": model["id"], "records": [{key: value for key, value in record.items() if key not in {"request", "response", "content", "reasoning_content"}} for record in records], "all_supported_routes_pass": all(record.get("status") in {"ok", "N/A_not_supported"} for record in records)}
    write_json_immutable(RAW_ROOT / model["id"] / f"{server.batch_id}.technical_smoke.summary.json", redact(evidence))
    if not evidence["all_supported_routes_pass"]:
        raise RuntimeError(f"technical smoke failed for {model['id']}: {evidence}")
    return evidence


def repair_allowed(error: str | None) -> bool:
    if not error:
        return False
    # A repair is permitted only for a malformed response envelope/shape.  A
    # coherent score conflict, arithmetic/range failure, evidence disagreement,
    # transport failure, or template/finish violation is retained as-is and is
    # never retried as a quality choice.
    forbidden = (
        "subscores_sum_mismatch", "subscore_out_of_range", "subscore_type_invalid",
        "score_invalid", "score_out_of_range", "coherent", "range",
        "confidence_out_of_range", "reason_codes_invalid", "error_schema",
        "critical_flag_vs_error_severity", "critical_missing_or_invalid",
        "off_mode", "control_token", "invalid_finish_reason", "http_", "HTTP",
        "transport", "unsupported_track",
    )
    if any(item in error for item in forbidden):
        return False
    allowed = (
        "json_parse", "response_parse", "top_level_not_object", "judgment_not_object",
        "required_field_missing", "pairwise_required_missing", "additional_properties",
        "blind_id_missing", "blind_id_invalid", "blind_id_pattern_invalid",
        "blind_id_not_in_task", "status_invalid", "abstain_missing_or_invalid",
        "ok_abstain_must_be_false", "blocked_abstain_must_be_true", "block_reason_missing",
        "blocked_score_subscores_must_be_null", "blocked_contract_invalid",
        "blocked_evidence_contract_invalid", "subscores_missing", "subscores_required_missing",
        "errors_missing", "pairwise_winner_invalid", "pairwise_confidence_invalid",
        "pairwise_critical_override_invalid",
    )
    return error.startswith(allowed)


def infrastructure_retry_allowed(attempt: dict[str, Any]) -> bool:
    """Allow at most one exact replay when no candidate judgment was parsed."""
    if attempt.get("status") != "invalid" or attempt.get("parsed") is not None:
        return False
    reason = str(attempt.get("invalid_reason") or "")
    transport_markers = (
        "http_not_200", "HTTPError", "URLError", "TimeoutError", "TimeoutExpired",
        "ConnectionRefusedError", "ConnectionResetError", "RemoteDisconnected",
        "ConnectionError", "BrokenPipeError", "transport", "socket",
    )
    if any(marker in reason for marker in transport_markers):
        return True
    if reason == "actual_template_render_gate_failed":
        envelope = attempt.get("actual_template_render") or {}
        gate = envelope.get("gate") if isinstance(envelope, dict) else {}
        http_envelope = envelope.get("http_envelope") if isinstance(envelope, dict) else {}
        # Retry only a render transport/non-200 failure.  A stable 200 prompt
        # with a wrong marker is a template gate failure, never a regrade.
        return not bool(gate.get("http_200")) or http_envelope.get("http_status") != 200
    return False


def expected_ids(task: dict[str, Any]) -> list[str]:
    if task.get("track") == "pairwise":
        return []
    if isinstance(task.get("variants"), list):
        return [blind_id_value(item, f"qual-{index+1}") for index, item in enumerate(task["variants"]) if isinstance(item, dict)]
    return [blind_id_value(task, "cand-unknown")]


def task_payload(
    server: LocalServer,
    model: dict[str, Any],
    task: dict[str, Any],
    instructions: str,
    manifest_path: Path,
    repair: bool = False,
    original_attempt: dict[str, Any] | None = None,
    format_errors: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repair_context = None
    if repair:
        repair_context = {
            "original_content": (original_attempt or {}).get("content", ""),
            "format_errors": list(format_errors or [(original_attempt or {}).get("invalid_reason", "format_invalid")]),
        }
    messages, task_meta = build_messages(task, instructions, manifest_path, repair=repair, repair_context=repair_context)
    payload = request_payload(model, messages)
    # Bind the actual scored request to a non-generating native render.  The
    # earlier synthetic probe is only a route gate; this envelope proves that
    # this exact source/candidate (and, for image jobs, this exact image) was
    # rendered by the same isolated llama.cpp template before generation.
    render_payload = {**payload, "add_generation_prompt": True}
    render_result = http_json("POST", server.base + "/apply-template", render_payload, timeout=120)
    render_body = render_result.get("response_body") if isinstance(render_result.get("response_body"), dict) else {}
    render_prompt = render_body.get("prompt") if isinstance(render_body, dict) else ""
    render_prompt = render_prompt if isinstance(render_prompt, str) else ""
    render_marker_tokens_list = ["<|image>", "<image|>", "<|image|>", "<start_of_image>", "<|vision_start|>", "<|image_pad|>", "[IMG]", "<image>"]
    props_pre = (server.meta.get("endpoints_pre") or {}).get("/props") if isinstance(server.meta, dict) else None
    props_pre_body = props_pre.get("response_body") if isinstance(props_pre, dict) and isinstance(props_pre.get("response_body"), dict) else {}
    native_media_marker = props_pre_body.get("media_marker") if isinstance(props_pre_body, dict) else None
    if isinstance(native_media_marker, str) and native_media_marker:
        render_marker_tokens_list.append(native_media_marker)
    render_marker_tokens = tuple(render_marker_tokens_list)
    render_marker_count = sum(render_prompt.count(token) for token in render_marker_tokens)
    render_gate = {
        "http_200": render_result.get("http_status") == 200,
        "prompt_nonempty": bool(render_prompt),
        "source_occurrences": render_prompt.count(source_value(task)) if source_value(task) else 0,
        "candidate_occurrences": [render_prompt.count(translation_value(item)) for item in (task_candidate_pair(task) if task.get("track") == "pairwise" else (task,))],
        "image_marker_count": render_marker_count,
        "actual_image": task_is_image(task),
    }
    render_gate["source_once_if_text"] = (render_gate["source_occurrences"] == 1) if not task_is_image(task) else True
    candidate_values = [translation_value(item) for item in (task_candidate_pair(task) if task.get("track") == "pairwise" else (task,))]
    expected_occurrences = {value: candidate_values.count(value) for value in candidate_values if value}
    render_gate["candidate_once"] = all(
        bool(value) and count == expected_occurrences.get(value, 1)
        for value, count in zip(candidate_values, render_gate["candidate_occurrences"])
    )
    render_gate["native_image_marker_if_image"] = (render_marker_count > 0) if task_is_image(task) else True
    render_gate["pass"] = bool(render_gate["http_200"] and render_gate["prompt_nonempty"] and render_gate["source_once_if_text"] and render_gate["candidate_once"] and render_gate["native_image_marker_if_image"])
    render_evidence = {
        "http_envelope": {key: value for key, value in render_result.items() if key not in {"request_redacted", "response_body"}},
        "request": redact(render_payload),
        "request_sha256": sha256_json(render_payload),
        "response_body": render_body,
        "response_body_sha256": render_result.get("response_body_sha256"),
        "prompt": render_prompt,
        "prompt_sha256": sha256_bytes(render_prompt.encode("utf-8")),
        "native_marker_counts": {token: render_prompt.count(token) for token in render_marker_tokens},
        "native_media_marker": native_media_marker,
        "gate": render_gate,
        "model_id": model["id"],
        "model_sha256_first_shard": sha256_file(Path(model["model"])),
        "mmproj_sha256": sha256_file(Path(model["mmproj"])) if model.get("mmproj") and Path(model["mmproj"]).is_file() else None,
    }
    if not render_gate["pass"]:
        # Preserve the non-generating evidence as an invalid technical
        # attempt; do not proceed to grading with an unproven render and do
        # not spend a format-repair request on a route/template failure.
        failure = "actual_template_render_gate_failed"
        return {
            "schema_version": "judge-local-attempt-v1",
            "recorded_at": now(), "model_id": model["id"],
            "model_identity": {"family": model["family"], "alias": model["alias"], "model": model["model"], "mmproj": model.get("mmproj")},
            "track": task.get("track"), "task_id": task.get("task_id"), "repair": repair,
            "task_metadata": task_meta, "request": redact(payload), "request_sha256": sha256_json(payload),
            "http_envelope": {"render_only": True, "render_http_status": render_result.get("http_status"), "render_response_body_sha256": render_result.get("response_body_sha256")},
            "response": {"body": None, "body_sha256": None}, "response_metadata": {},
            "content": "", "reasoning_content": "", "parsed": None, "status": "invalid",
            "invalid_reason": failure, "validation_errors": [failure], "evidence_diagnostics": [],
            "schema_validation_method": schema_validation_method(), "jsonschema_diagnostics": [],
            "repair_context": repair_context, "actual_template_render": render_evidence,
        }, {"status": "invalid", "invalid_reason": failure, "response_metadata": {}, "parsed": None, "validation": {"valid": False, "error": failure, "validation_errors": [failure], "evidence_diagnostics": []}}
    result = http_json("POST", server.base + "/v1/chat/completions", payload, timeout=1800)
    response_meta, response_error = response_metadata(result, model, payload, tokenize_base=server.base)
    parsed: Any = None
    parse_error: str | None = None
    if response_error is None:
        try:
            parsed = parse_json_content(response_meta["content"])
            validation = validate_contract_details(str(task.get("track", "pointwise")), parsed, expected_ids(task))
            validation["evidence_diagnostics"] = list(validation.get("evidence_diagnostics", [])) + evidence_quality_diagnostics(str(task.get("track", "pointwise")), parsed, task)
            if not validation["valid"]:
                parse_error = validation["error"]
        except Exception as exc:
            parse_error = f"json_parse:{type(exc).__name__}:{exc}"
            validation = {
                "valid": False, "error": parse_error, "validation_errors": [parse_error],
                "evidence_diagnostics": [], "schema_validation_method": schema_validation_method(),
                "jsonschema_diagnostics": [],
            }
    else:
        parse_error = response_error
        validation = {
            "valid": False, "error": parse_error, "validation_errors": [parse_error] if parse_error else [],
            "evidence_diagnostics": [], "schema_validation_method": schema_validation_method(),
            "jsonschema_diagnostics": [],
        }
    status = "ok" if parse_error is None else "invalid"
    return {
        "schema_version": "judge-local-attempt-v1",
        "recorded_at": now(),
        "model_id": model["id"],
        "model_identity": {"family": model["family"], "alias": model["alias"], "model": model["model"], "mmproj": model.get("mmproj")},
        "track": task.get("track"),
        "task_id": task.get("task_id"),
        "repair": repair,
        "task_metadata": task_meta,
        "request": redact(payload),
        "request_sha256": sha256_json(payload),
        "http_envelope": {key: value for key, value in result.items() if key not in {"request_redacted", "response_body"}},
        "response": {"body": result.get("response_body"), "body_sha256": result.get("response_body_sha256")},
        "response_metadata": {key: value for key, value in response_meta.items() if key not in {"content", "reasoning_content"}},
        "content": response_meta.get("content", ""),
        "reasoning_content": response_meta.get("reasoning_content", ""),
        "parsed": parsed,
        "status": status,
        "invalid_reason": parse_error,
        "validation_errors": validation.get("validation_errors", []),
        "evidence_diagnostics": validation.get("evidence_diagnostics", []),
        "schema_validation_method": validation.get("schema_validation_method"),
        "jsonschema_diagnostics": validation.get("jsonschema_diagnostics", []),
        "repair_context": repair_context,
        "actual_template_render": render_evidence,
    }, {"status": status, "invalid_reason": parse_error, "response_metadata": response_meta, "parsed": parsed, "validation": validation}


def raw_path(model_id: str, task_id: str, attempt: int, kind: str = "format") -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    if attempt == 1:
        suffix = ""
    elif kind == "infrastructure":
        suffix = f"__infrastructure_retry{attempt - 1}"
    else:
        suffix = f"__attempt{attempt}"
    return RAW_ROOT / model_id / "requests" / f"{safe}{suffix}.json"


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_progress(status: str, message: str, state: dict[str, Any] | None = None) -> None:
    lines = ["# New local judge evaluation progress", "", f"- Updated: `{now()}`", f"- Status: **{status}**", "", message]
    handoff: dict[str, Any] = {}
    if HANDOFF_PATH.is_file():
        try:
            loaded = read_json(HANDOFF_PATH)
            if isinstance(loaded, dict):
                handoff = loaded
        except Exception:
            handoff = {}
    scheduler: dict[str, Any] = {}
    if SCHEDULER_PATH.is_file():
        try:
            loaded = read_json(SCHEDULER_PATH)
            if isinstance(loaded, dict):
                scheduler = loaded
        except Exception:
            scheduler = {}
    requested = [str(item) for item in scheduler.get("requested_order", [])]
    actual = {str(item) for item in scheduler.get("actual_order", [])}
    pending = [item for item in requested if item not in actual]
    server_preflight = state.get("server_preflight") if isinstance(state, dict) else None
    if not isinstance(server_preflight, dict):
        server_preflight = {}
    lines.extend([
        "",
        "## Active handoff",
        "",
        f"- controller PID: `{os.getpid()}`",
        f"- controller terminal session: `{handoff.get('terminal_session_id', 'not recorded by driver; see active_handoff.json')}`",
        f"- active model: `{state.get('model_id') if isinstance(state, dict) else scheduler.get('status', 'none')}`",
        f"- isolated server PID/port: `{server_preflight.get('pid', 'not loaded')}` / `{server_preflight.get('port', 'not loaded')}`",
        f"- scheduler status: `{scheduler.get('status', 'not initialized')}`",
        f"- automatic pending model candidates: `{', '.join(pending) if pending else 'none recorded'}`",
        f"- scheduler file: `{SCHEDULER_PATH}`",
        f"- immutable raw root: `{RAW_ROOT}`",
    ])
    if state:
        counts = state.get("counts") or {}
        lines.extend(["", "## Current counts", "", f"- model: `{state.get('model_id')}`", f"- pointwise planned/completed: `{counts.get('pointwise_planned', 0)}/{counts.get('pointwise_completed', 0)}`", f"- ABBA planned/completed: `{counts.get('abba_planned', 0)}/{counts.get('abba_completed', 0)}`", f"- valid: `{counts.get('valid', 0)}`; invalid/missing: `{counts.get('invalid', 0)}`; format-only repairs: `{counts.get('format_repairs', 0)}`; infrastructure retries: `{counts.get('infrastructure_retries', 0)}`", f"- throughput calls/hour: `{state.get('throughput_calls_per_hour')}`"])
    atomic_write(PROGRESS_PATH, "\n".join(lines) + "\n")


def job_list_sha256(interface: dict[str, Any]) -> str:
    jobs = [
        {"track": row.get("track"), "task_id": str(row.get("task_id"))}
        for row in (interface.get("pointwise", []) + interface.get("pairwise", []))
    ]
    return sha256_json(jobs)


def mode_sha256(model: dict[str, Any]) -> str:
    return sha256_json({
        "model": model,
        "fixed_generation": {
            "context": 32768, "max_tokens": 16384, "reasoning_budget_tokens": 8192,
            "temperature": 0, "top_p": 1, "seed": SEED, "stream": False,
            "parallel": 1, "cache_prompt": False, "speculative": "off",
        },
    })


def state_identity(interface: dict[str, Any], source: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": model["id"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "manifest_sha256": interface.get("manifest_sha256"),
        "packet_sha256": interface.get("packet_sha256"),
        "instructions_sha256": interface.get("instructions_sha256"),
        "source_csv_sha256": source.get("sha256"),
        "job_list_sha256": job_list_sha256(interface),
        "mode_sha256": mode_sha256(model),
    }


def state_template(interface: dict[str, Any], source: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    state = {
        "schema_version": "judge-local-driver-state-v1",
        "run_started_at": now(),
        "phase": "GENERATING",
        "model_id": model["id"],
        "model": model,
        "interface": {key: value for key, value in interface.items() if key not in {"pointwise", "pairwise"}},
        "source_csv": source,
        "records": [],
        "completed_task_ids": [],
        "counts": {"pointwise_planned": len(interface["pointwise"]), "pointwise_completed": 0, "abba_planned": len(interface["pairwise"]), "abba_completed": 0, "valid": 0, "invalid": 0, "format_repairs": 0, "infrastructure_retries": 0, "technical_n_a_image": 0},
        "throughput_calls_per_hour": 0.0,
    }
    state.update(state_identity(interface, source, model))
    return state


def assert_state_identity(state: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = []
    for key, value in expected.items():
        if state.get(key) != value:
            mismatches.append(f"{key}: expected={value!r} observed={state.get(key)!r}")
    if mismatches:
        raise RuntimeError("checkpoint identity mismatch; refusing resume: " + "; ".join(mismatches))


def record_from_attempt(
    task: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any] | None = None,
    second_kind: str = "format",
) -> dict[str, Any]:
    final = second or first
    attempts = [str(raw_path(str(final.get("model_id")), str(task["task_id"]), 1))]
    if second is not None:
        attempts.append(str(raw_path(str(final.get("model_id")), str(task["task_id"]), 2, second_kind)))
    return {
        "schema_version": "judge-local-record-v1",
        "model_id": final.get("model_id"),
        "track": task.get("track"),
        "task_id": str(task["task_id"]),
        "task": task,
        "status": final.get("status"),
        "invalid_reason": final.get("invalid_reason"),
        "attempts": attempts,
        "format_repair_attempted": second is not None and second_kind == "format",
        "infrastructure_retry_attempted": second is not None and second_kind == "infrastructure",
        "recovery_kind": second_kind if second is not None else None,
        "parsed": final.get("parsed"),
        "response_metadata": final.get("response_metadata"),
        "request_sha256": final.get("request_sha256"),
        "response_body_sha256": (final.get("response") or {}).get("body_sha256"),
        "english_transcription_sent_for_image": bool((final.get("task_metadata") or {}).get("english_transcription_sent_for_image")),
        "elapsed_s": (final.get("http_envelope") or {}).get("elapsed_s"),
        "actual_template_render": final.get("actual_template_render"),
        "evidence_diagnostics": final.get("evidence_diagnostics", []),
        "validation_errors": final.get("validation_errors", []),
        "schema_validation_method": final.get("schema_validation_method"),
        "repair_context": final.get("repair_context"),
        "completed_at": now(),
    }


def recover_existing_raw_attempts(
    state: dict[str, Any],
    tasks: list[dict[str, Any]],
    started_monotonic: float,
) -> set[str]:
    """Reload prior immutable attempts before any new server is started."""
    completed = set(state.get("completed_task_ids", []))
    result_ids: set[str] = set()
    existing_results = results_path(str(state["model_id"]))
    if existing_results.is_file():
        try:
            for line in existing_results.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("task_id"):
                        result_ids.add(str(row["task_id"]))
        except Exception as exc:
            raise RuntimeError(f"cannot reload append-only result file {existing_results}: {type(exc).__name__}:{exc}") from exc
    recovered = 0
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id in completed:
            continue
        first_path = raw_path(str(state["model_id"]), task_id, 1)
        second_path = raw_path(str(state["model_id"]), task_id, 2, "format")
        infrastructure_path = raw_path(str(state["model_id"]), task_id, 2, "infrastructure")
        if not first_path.exists() and (second_path.exists() or infrastructure_path.exists()):
            orphan = second_path if second_path.exists() else infrastructure_path
            raise RuntimeError(f"retry raw exists without original; refusing overwrite: {orphan}")
        if not first_path.exists():
            continue
        if second_path.exists() and infrastructure_path.exists():
            raise RuntimeError(f"both format and infrastructure retries exist; refusing ambiguous resume: {task_id}")
        try:
            first = read_json(first_path)
            second = read_json(second_path) if second_path.exists() else (read_json(infrastructure_path) if infrastructure_path.exists() else None)
        except Exception as exc:
            raise RuntimeError(f"cannot reload immutable raw attempt {first_path}: {type(exc).__name__}:{exc}") from exc
        second_kind = "infrastructure" if infrastructure_path.exists() else "format"
        record = record_from_attempt(task, first, second, second_kind)
        update_state(state, record, 0, started_monotonic, persist=False)
        if task_id not in result_ids:
            append_jsonl(existing_results, record)
            result_ids.add(task_id)
        completed.add(task_id)
        recovered += 1
    if recovered:
        write_json(checkpoint_path(str(state["model_id"])), state)
    return completed


def update_state(
    state: dict[str, Any],
    record: dict[str, Any],
    attempt_count: int,
    started_monotonic: float,
    persist: bool = True,
) -> None:
    state["records"].append(record)
    task_id = record.get("task_id")
    if task_id and task_id not in state["completed_task_ids"]:
        state["completed_task_ids"].append(task_id)
        if record.get("track") == "pointwise":
            state["counts"]["pointwise_completed"] += 1
        elif record.get("track") == "pairwise":
            state["counts"]["abba_completed"] += 1
    if record.get("status") == "ok":
        state["counts"]["valid"] += 1
    elif record.get("status") == "N/A_not_supported":
        state["counts"]["technical_n_a_image"] += 1
    else:
        state["counts"]["invalid"] += 1
    if record.get("format_repair_attempted"):
        state["counts"]["format_repairs"] += 1
    if record.get("infrastructure_retry_attempted"):
        state["counts"]["infrastructure_retries"] = state["counts"].get("infrastructure_retries", 0) + 1
    elapsed = max(0.001, time.monotonic() - started_monotonic)
    state["throughput_calls_per_hour"] = round(attempt_count / elapsed * 3600.0, 3)
    state["last_updated_at"] = now()
    if persist:
        write_json(checkpoint_path(str(state["model_id"])), state)
        append_jsonl(results_path(str(state["model_id"])), record)
        completed_total = state["counts"]["pointwise_completed"] + state["counts"]["abba_completed"]
        if completed_total and completed_total % ANALYSIS_REFRESH_EVERY == 0:
            refresh_analysis_reports(state, reason=f"batch_{completed_total}")
    write_progress("GENERATING", "Requests are executed one fresh source/candidate at a time; raw attempts are immutable. Invalid/missing outputs are retained and excluded from any future summary.", state)


def refresh_analysis_reports(state: dict[str, Any], reason: str) -> dict[str, Any]:
    """Refresh read-only aggregate reports in a separate no-LLM subprocess.

    Aggregation is advisory to generation: a report refresh can never turn a
    completed judge request into a retry or interrupt the immutable local raw
    record.  The result is retained in the per-model checkpoint for audit.
    """
    started = time.monotonic()
    evidence: dict[str, Any] = {
        "started_at": now(),
        "reason": reason,
        "command": [sys.executable, str(ANALYSIS_RUNNER), "--refresh"],
        "analysis_runner": str(ANALYSIS_RUNNER),
    }
    if not ANALYSIS_RUNNER.is_file():
        evidence.update({"status": "skipped_missing", "error": "analysis_runner_missing"})
    else:
        try:
            completed = subprocess.run(
                [sys.executable, str(ANALYSIS_RUNNER), "--refresh"],
                cwd=str(JUDGE_ROOT), capture_output=True, text=True,
                timeout=60, check=False,
            )
            evidence.update({
                "status": "ok" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            })
        except subprocess.TimeoutExpired as exc:
            evidence.update({"status": "timeout", "error": f"TimeoutExpired:{exc}"})
        except Exception as exc:
            evidence.update({"status": "error", "error": f"{type(exc).__name__}:{exc}"})
    evidence["finished_at"] = now()
    evidence["elapsed_s"] = time.monotonic() - started
    state.setdefault("analysis_refreshes", []).append(evidence)
    state["last_analysis_refresh"] = evidence
    write_json(checkpoint_path(str(state["model_id"])), state)
    return evidence


def derive_abba(state: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for record in state.get("records", []):
        if record.get("track") != "pairwise" or record.get("status") != "ok":
            continue
        task = record.get("task") or {}
        pair_id = str(task.get("pair_key") or task.get("pair_id") or task.get("item_id") or record.get("task_id", "").split("__", 1)[0])
        orientation = str(task.get("orientation") or "AB").upper()
        groups.setdefault(pair_id, {})[orientation] = record
    rows: list[dict[str, Any]] = []
    for pair_id, orientations in sorted(groups.items()):
        ab = orientations.get("AB")
        ba = orientations.get("BA")
        row: dict[str, Any] = {"pair_id": pair_id, "has_AB": bool(ab), "has_BA": bool(ba), "status": "missing"}
        if ab and ba:
            aw = ((ab.get("parsed") or {}).get("winner"))
            bw = ((ba.get("parsed") or {}).get("winner"))
            normalized_ab = {"A": "candidate_a", "B": "candidate_b", "TIE": "TIE"}.get(aw, "invalid")
            normalized_ba = {"A": "candidate_b", "B": "candidate_a", "TIE": "TIE"}.get(bw, "invalid")
            row.update({"AB_winner": aw, "BA_winner": bw, "normalized_AB": normalized_ab, "normalized_BA": normalized_ba})
            if normalized_ab == "invalid" or normalized_ba == "invalid":
                row["status"] = "abstain_invalid"
            elif normalized_ab == normalized_ba:
                row["status"] = "consistent" if normalized_ab != "TIE" else "consistent_tie"
                row["canonical_winner"] = normalized_ab
            else:
                # A swapped inconsistent pair is not silently relabeled TIE.
                row["status"] = "abstain_inconsistent"
                row["canonical_winner"] = None
                row["position_bias_signal"] = normalized_ab == "candidate_a" and normalized_ba == "candidate_b"
    return {"derived_at": now(), "judge_id": state.get("model_id"), "pairs": rows, "counts": {"groups": len(rows), "consistent": sum(row.get("status") in {"consistent", "consistent_tie"} for row in rows), "abstain_inconsistent": sum(row.get("status") == "abstain_inconsistent" for row in rows), "missing": sum(row.get("status") == "missing" for row in rows), "position_following": sum(bool(row.get("position_bias_signal")) for row in rows)}, "mistral_old_position_bias_note": "Old pilot reported p-01/p-03 position sensitivity; this run reports each frozen pair's AB/BA result and never converts an inconsistent swap to TIE." if state.get("model", {}).get("family") == "mistral" else None}


def preflight() -> int:
    manifest_path, manifest, errors = discover_interface()
    source = load_source_csv() if SOURCE_CSV.is_file() else {"error": f"missing {SOURCE_CSV}"}
    models = default_model_configs()
    inventories = [{"id": model["id"], **model_inventory(model), "command": server_command(model)} for model in models]
    approval = approval_state(manifest_path, manifest)
    packet_path = None
    packet = None
    packet_error = "manifest missing"
    instructions_path = None
    instructions_error = "manifest missing"
    interface = None
    if manifest_path and manifest:
        packet_path, packet, packet_error = load_slot_packet(manifest_path, manifest, "judge-01")
        instructions_path, instructions, instructions_error = load_instructions(manifest_path, manifest)
        if packet is not None:
            interface = normalize_interface(manifest_path, manifest, packet)
            interface["packet_sha256"] = sha256_json({"pointwise": sha256_file(packet_path) if packet_path else None, "pairwise": sha256_file(packet_path.parent.parent / "pairwise" / "judge-01.jsonl") if packet_path else None})
            if instructions is not None:
                interface["instructions_sha256"] = sha256_bytes(instructions.encode("utf-8"))
    formal = formal_health_evidence()
    evidence = {
        "schema_version": "judge-local-preflight-v1",
        "checked_at": now(),
        "formal_read_only": formal,
        "runtime": {"binary": str(SERVER), "version": runtime_version() if SERVER.is_file() else None, "expected": EXPECTED_BUILD},
        "source_csv": source,
        "manifest": {"path": str(manifest_path) if manifest_path else None, "sha256": sha256_file(manifest_path) if manifest_path else None, "errors": errors},
        "approval": approval,
        "packet": {"path": str(packet_path) if packet_path else None, "error": packet_error},
        "instructions": {"path": str(instructions_path) if instructions_path else None, "error": instructions_error},
        "interface_counts": {"pointwise": len(interface["pointwise"]) if interface else None, "abba": len(interface["pairwise"]) if interface else None, "expected_pointwise": interface.get("expected_pointwise") if interface else None, "expected_abba": interface.get("expected_abba") if interface else None},
        "models": inventories,
        "formal_untouched": True,
        "can_start_generation": bool(interface and approval.get("approved") and not packet_error and not instructions_error and not any(item["errors"] for item in inventories)),
    }
    write_json(RUNNER_ROOT / "preflight.json", redact(evidence))
    if not manifest_path:
        write_progress("WAITING_FOR_PROTOCOL", "The protocol worker has not published a new frozen manifest yet. No model was started and no judge request was sent.")
    elif not approval.get("approved"):
        write_progress("WAITING_FOR_ROOT_APPROVAL", f"Frozen manifest discovered at `{manifest_path}`, but no explicit ROOT_APPROVED marker is present. No scoring request was sent.")
    elif evidence["can_start_generation"]:
        write_progress("READY_FOR_GENERATION", f"Frozen interface is ready: pointwise={len(interface['pointwise'])}, ABBA={len(interface['pairwise'])}; exact local model/mmproj inventories recorded. Run phase remains gated until driver --run.")
    else:
        write_progress("PREFLIGHT_BLOCKED", "The frozen manifest/interface is incomplete or a model/runtime inventory gate failed. No model was started.")
    print(json.dumps({"preflight": str(RUNNER_ROOT / 'preflight.json'), "can_start_generation": evidence["can_start_generation"], "pointwise": evidence["interface_counts"]["pointwise"], "abba": evidence["interface_counts"]["abba"], "approval": approval.get("approved")}, ensure_ascii=False))
    return 0 if evidence["can_start_generation"] else 2


def run_one_model(model: dict[str, Any], interface: dict[str, Any], instructions: str, manifest_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    expected_identity = state_identity(interface, source, model)
    checkpoint = checkpoint_path(model["id"])
    state = state_template(interface, source, model)
    if checkpoint.is_file():
        state = read_json(checkpoint)
        assert_state_identity(state, expected_identity)
    tasks = interface["pointwise"] + interface["pairwise"]
    started_monotonic = time.monotonic()
    completed = recover_existing_raw_attempts(state, tasks, started_monotonic)
    # A completed model is immutable at the judge-call level.  Resume skips
    # starting the server again and simply returns its saved diagnostics.
    if state.get("phase") == "MODEL_COMPLETE":
        write_progress("MODEL_COMPLETE", f"Resuming completed `{model['id']}` from its immutable per-model checkpoint; no new judge requests.", state)
        return state
    state["phase"] = "GENERATING"
    write_json(checkpoint, state)
    batch_id = f"judge-{model['id']}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    server = LocalServer(model, batch_id)
    attempt_count = 0
    write_progress("STARTING_MODEL", f"Starting isolated `{model['id']}`; formal 8080 is read-only and untouched.", state)
    try:
        server_meta = server.start()
        state["server_preflight"] = server_meta
        state["template_probe"] = template_probe(server, model)
        vision_supported = bool(state["template_probe"].get("vision_supported") is True and state["template_probe"].get("overall_vision_pass"))
        state["vision_runtime_supported"] = vision_supported
        state["vision_gate_reason"] = "native_props_and_apply_template_pass" if vision_supported else "native_image_route_not_proven; image cells recorded N/A_not_supported"
        state["technical_smoke"] = technical_smoke(server, model, vision_supported)
        write_json(checkpoint, state)
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id in completed:
                continue
            if task_is_image(task) and not vision_supported:
                record = {"schema_version": "judge-local-record-v1", "model_id": model["id"], "track": task.get("track"), "task_id": task_id, "task": task, "status": "N/A_not_supported", "invalid_reason": "runtime_props_vision_false_or_mmproj_missing", "image_hash": image_hash_value(task), "english_transcription_sent_for_image": False, "quality_scored": False}
                n_a_path = raw_path(model["id"], task_id, 1)
                write_json_immutable(n_a_path, {**record, "vision_gate": {"supported": vision_supported, "reason": state.get("vision_gate_reason")}, "recorded_at": now()})
                record["attempts"] = [str(n_a_path)]
                update_state(state, record, attempt_count, started_monotonic)
                completed.add(task_id)
                continue
            attempt_count += 1
            try:
                first, info = task_payload(server, model, task, instructions, manifest_path, repair=False)
            except Exception as exc:
                first = {"schema_version": "judge-local-attempt-v1", "model_id": model["id"], "track": task.get("track"), "task_id": task_id, "repair": False, "status": "invalid", "invalid_reason": f"request_build:{type(exc).__name__}:{exc}", "task": task, "validation_errors": [f"request_build:{type(exc).__name__}:{exc}"]}
                info = {"status": "invalid", "invalid_reason": first["invalid_reason"], "response_metadata": {}}
            first_path = raw_path(model["id"], task_id, 1)
            write_json_immutable(first_path, first)
            record = record_from_attempt(task, first)
            if infrastructure_retry_allowed(first):
                # One exact replay is allowed only for an ungraded transport
                # failure.  It is not a format repair and cannot mutate the
                # first raw attempt or semantic fields.
                attempt_count += 1
                try:
                    second, info2 = task_payload(server, model, task, instructions, manifest_path, repair=False)
                except Exception as exc:
                    second = {"schema_version": "judge-local-attempt-v1", "model_id": model["id"], "track": task.get("track"), "task_id": task_id, "repair": False, "recovery_kind": "infrastructure_retry", "status": "invalid", "invalid_reason": f"request_build:{type(exc).__name__}:{exc}", "task": task, "validation_errors": [f"request_build:{type(exc).__name__}:{exc}"]}
                    info2 = {"status": "invalid", "invalid_reason": second["invalid_reason"], "response_metadata": {}}
                second["recovery_kind"] = "infrastructure_retry"
                second["infrastructure_retry"] = True
                second_path = raw_path(model["id"], task_id, 2, "infrastructure")
                write_json_immutable(second_path, second)
                record = record_from_attempt(task, first, second, "infrastructure")
            elif first.get("status") != "ok" and repair_allowed(first.get("invalid_reason")):
                attempt_count += 1
                record["format_repair_attempted"] = True
                try:
                    format_errors = list(first.get("validation_errors") or [first.get("invalid_reason")])
                    second, info2 = task_payload(server, model, task, instructions, manifest_path, repair=True, original_attempt=first, format_errors=format_errors)
                except Exception as exc:
                    second = {"schema_version": "judge-local-attempt-v1", "model_id": model["id"], "track": task.get("track"), "task_id": task_id, "repair": True, "status": "invalid", "invalid_reason": f"request_build:{type(exc).__name__}:{exc}", "task": task, "validation_errors": [f"request_build:{type(exc).__name__}:{exc}"], "repair_context": {"original_content": first.get("content", ""), "format_errors": format_errors}}
                    info2 = {"status": "invalid", "invalid_reason": second["invalid_reason"], "response_metadata": {}}
                second_path = raw_path(model["id"], task_id, 2, "format")
                write_json_immutable(second_path, second)
                record = record_from_attempt(task, first, second, "format")
            update_state(state, record, attempt_count, started_monotonic)
            completed.add(task_id)
        state["abba_diagnostics"] = derive_abba(state)
        state["phase"] = "MODEL_COMPLETE"
        state["completed_at"] = now()
        # Flush a final partial batch so the shared read-only reports do not
        # wait for the next model's fifth completed task.
        refresh_analysis_reports(state, reason="model_complete")
        write_json(checkpoint, state)
        write_json(model_summary_path(model["id"]), state)
        write_progress("MODEL_COMPLETE", f"Completed `{model['id']}`; raw attempts, invalid/missing states, and ABBA diagnostics are retained. Proceeding to guarded cleanup.", state)
        return state
    finally:
        try:
            cleanup = server.stop(reason="model_complete" if state.get("phase") == "MODEL_COMPLETE" else "model_failure")
            state["server_cleanup"] = cleanup
            state["formal_health_after_cleanup"] = cleanup.get("formal_health_after")
            write_json(checkpoint, state)
        finally:
            try:
                server.stop(reason="idempotent_cleanup_check")
            except Exception:
                pass


def run_all(wait_for_approval: bool, poll_seconds: int) -> int:
    while True:
        manifest_path, manifest, _ = discover_interface()
        approval = approval_state(manifest_path, manifest)
        if manifest_path and manifest and approval.get("approved"):
            packet_path, packet, packet_error = load_slot_packet(manifest_path, manifest, "judge-01")
            instructions_path, instructions, instructions_error = load_instructions(manifest_path, manifest)
            if packet is not None and instructions is not None and not packet_error and not instructions_error:
                interface = normalize_interface(manifest_path, manifest, packet)
                interface["packet_sha256"] = sha256_json({"pointwise": sha256_file(packet_path) if packet_path else None, "pairwise": sha256_file(packet_path.parent.parent / "pairwise" / "judge-01.jsonl") if packet_path else None})
                interface["instructions_sha256"] = sha256_bytes(instructions.encode("utf-8"))
                if interface["expected_pointwise"] is not None and interface["expected_pointwise"] != len(interface["pointwise"]):
                    raise RuntimeError(f"frozen pointwise count mismatch: expected {interface['expected_pointwise']} actual {len(interface['pointwise'])}")
                if interface["expected_abba"] is not None and interface["expected_abba"] != len(interface["pairwise"]):
                    raise RuntimeError(f"frozen ABBA count mismatch: expected {interface['expected_abba']} actual {len(interface['pairwise'])}")
                if not SOURCE_CSV.is_file():
                    raise RuntimeError(f"source CSV missing: {SOURCE_CSV}")
                source = load_source_csv()
                if source["rows"] != 238:
                    raise RuntimeError(f"translation source CSV row count is not frozen 238: {source['rows']}")
                requested_models = list(zip(default_model_configs(), ("judge-01", "judge-02", "judge-03")))
                scheduler: dict[str, Any] = {
                    "schema_version": "judge-local-scheduler-v1",
                    "manifest_sha256": interface["manifest_sha256"],
                    "config_sha256": sha256_file(CONFIG_PATH),
                    "requested_order": [model["id"] for model, _ in requested_models],
                    "actual_order": [],
                    "deferred_ram": [],
                    "events": [],
                    "status": "READY",
                }
                if SCHEDULER_PATH.is_file():
                    prior_scheduler = read_json(SCHEDULER_PATH)
                    if prior_scheduler.get("manifest_sha256") != scheduler["manifest_sha256"] or prior_scheduler.get("config_sha256") != scheduler["config_sha256"]:
                        raise RuntimeError("scheduler identity mismatch; refusing resume")
                    scheduler = prior_scheduler
                    scheduler["actual_order"] = list(dict.fromkeys(scheduler.get("actual_order", [])))
                write_json(SCHEDULER_PATH, scheduler)
                slot_interfaces: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
                for model, slot in requested_models:
                    slot_point_path, slot_packet, slot_error = load_slot_packet(manifest_path, manifest, slot)
                    if slot_packet is None or slot_error:
                        raise RuntimeError(f"{slot}: {slot_error}")
                    slot_interface = normalize_interface(manifest_path, manifest, slot_packet)
                    slot_interface["packet_sha256"] = sha256_json({"pointwise": sha256_file(slot_point_path) if slot_point_path else None, "pairwise": sha256_file(slot_point_path.parent.parent / "pairwise" / f"{slot}.jsonl") if slot_point_path else None})
                    slot_interface["instructions_sha256"] = interface["instructions_sha256"]
                    if len(slot_interface["pointwise"]) != 238 or len(slot_interface["pairwise"]) != 72:
                        raise RuntimeError(f"{slot}: frozen packet count mismatch pointwise={len(slot_interface['pointwise'])} pairwise={len(slot_interface['pairwise'])}")
                    slot_interfaces[model["id"]] = (model, slot_interface)
                summaries: list[dict[str, Any]] = []
                pending: list[tuple[dict[str, Any], str]] = []
                for model, slot in requested_models:
                    saved_checkpoint = checkpoint_path(model["id"])
                    if saved_checkpoint.is_file():
                        saved_state = read_json(saved_checkpoint)
                        assert_state_identity(saved_state, state_identity(slot_interfaces[model["id"]][1], source, model))
                        if saved_state.get("phase") == "MODEL_COMPLETE":
                            summaries.append(saved_state)
                            if model["id"] not in scheduler.get("actual_order", []):
                                scheduler["actual_order"].append(model["id"])
                            scheduler.setdefault("events", []).append({"at": now(), "model_id": model["id"], "slot": slot, "event": "resume_skip_model_complete"})
                            continue
                    pending.append((model, slot))
                scheduler["status"] = "RUNNING"
                write_progress("READY_FOR_GENERATION", f"ROOT_APPROVED observed; frozen packet counts pointwise={len(interface['pointwise'])}, ABBA={len(interface['pairwise'])}. Scheduler requested Qwen→Nemotron→Mistral, with approved RAM-safe deferral recorded in `{SCHEDULER_PATH}`.")
                while pending:
                    eligible: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
                    for model, slot in pending:
                        inventory = model_inventory(model)
                        if inventory["errors"]:
                            raise RuntimeError("model/runtime inventory failed: " + "; ".join(inventory["errors"]))
                        headroom = vm_headroom(inventory["load_estimate_bytes"] + 32 * GIB, "scheduler_preload", inventory["load_estimate_bytes"])
                        if headroom.get("pass"):
                            eligible.append((model, slot, {"inventory": inventory, "headroom": headroom}))
                        else:
                            event = {"at": now(), "model_id": model["id"], "slot": slot, "event": "deferred_ram", "inventory": inventory, "headroom": headroom}
                            scheduler.setdefault("deferred_ram", []).append(event)
                            scheduler.setdefault("events", []).append(event)
                    if not eligible:
                        scheduler["status"] = "WAITING_FOR_RAM"
                        write_json(SCHEDULER_PATH, scheduler)
                        write_progress("WAITING_FOR_RAM", f"No pending local judge satisfies model+mmproj+32 GiB; scheduler retains deferred evidence and will recheck within 60s.")
                        if not wait_for_approval:
                            raise RuntimeError("no pending judge meets RAM gate")
                        time.sleep(min(60, max(1, poll_seconds)))
                        continue
                    model, slot, selection = eligible[0]
                    pending = [(candidate, candidate_slot) for candidate, candidate_slot in pending if candidate["id"] != model["id"]]
                    if model["id"] not in scheduler["actual_order"]:
                        scheduler["actual_order"].append(model["id"])
                    scheduler["events"].append({"at": now(), "model_id": model["id"], "slot": slot, "event": "selected_ram_safe", "selection": selection})
                    scheduler["status"] = "RUNNING_" + model["id"]
                    write_json(SCHEDULER_PATH, scheduler)
                    slot_interface = slot_interfaces[model["id"]][1]
                    while True:
                        try:
                            summaries.append(run_one_model(model, slot_interface, instructions, manifest_path, source))
                            break
                        except RuntimeError as exc:
                            message = str(exc)
                            if "RAM guard failed" not in message or not wait_for_approval:
                                raise
                            # The precheck can race with other allocations.
                            # Return this model to the pending queue and let
                            # the next eligible model make progress; no judge
                            # request is sent when LocalServer.start() fails.
                            event = {"at": now(), "model_id": model["id"], "slot": slot, "event": "deferred_ram_start_race", "error": message}
                            scheduler.setdefault("deferred_ram", []).append(event)
                            scheduler.setdefault("events", []).append(event)
                            pending.append((model, slot))
                            scheduler["status"] = "WAITING_FOR_RAM"
                            write_json(SCHEDULER_PATH, scheduler)
                            write_progress("WAITING_FOR_RAM", f"`{model['id']}` was deferred after an isolated RAM-gate race; scheduler will try the next eligible judge and revisit it. {message}")
                            break
                    # If the model was deferred in the inner loop, continue
                    # with the next pending model immediately.
                scheduler["status"] = "ALL_MODELS_COMPLETE"
                scheduler["completed_at"] = now()
                write_json(SCHEDULER_PATH, scheduler)
                write_json(SUMMARY_PATH, {"schema_version": "judge-local-summary-v1", "completed_at": now(), "scheduler": scheduler, "models": [{"model_id": item.get("model_id"), "phase": item.get("phase"), "counts": item.get("counts"), "abba_diagnostics": item.get("abba_diagnostics")} for item in summaries], "quality_scoring": "not performed by driver; raw judge JSON is retained for root review"})
                write_progress("ALL_MODELS_COMPLETE", f"All three local judges completed serially. Pointwise/ABBA raw records remain under `{RAW_ROOT}`; no formal endpoint or prior bundle was modified.")
                return 0
            if not wait_for_approval:
                write_progress("WAITING_FOR_PACKET", f"ROOT_APPROVED is present but packet/instructions are incomplete: packet={packet_error!r}, instructions={instructions_error!r}")
                return 2
        if not wait_for_approval:
            write_progress("WAITING_FOR_ROOT_APPROVAL", "The new frozen protocol manifest and explicit ROOT_APPROVED marker are not both present. No model was started.")
            return 2
        write_progress("WAITING_FOR_ROOT_APPROVAL", "Waiting for the protocol worker's new frozen manifest and explicit ROOT_APPROVED marker; no model is running and no judge request is sent.")
        time.sleep(max(1, min(60, poll_seconds)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="inventory the new interface/models without starting a model")
    parser.add_argument("--run", action="store_true", help="run all three judges if ROOT_APPROVED is present")
    parser.add_argument("--wait-for-approval", action="store_true", help="persistently wait up to 60s between checks, then run all judges")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.preflight or not (args.run or args.wait_for_approval):
        return preflight()
    return run_all(args.wait_for_approval, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
