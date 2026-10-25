#!/usr/bin/env python3
"""Run the frozen 2026-10-23 seven-model translation execution.

This runner deliberately keeps material gating, native template rendering,
vision capability checks, batch lifecycle evidence, and translation outputs
separate.  It never sends image ground truth to a model and never falls back
from a failed native vision route to OCR text.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "translation_config.json"
RAW_ROOT = ROOT / "raw"
RESULT_ROOT = ROOT / "results"
SMOKE_ROOT = ROOT / "technical_smoke"
GIB = 1024 ** 3
# The current llama.cpp runtime emits both the older ``<|token|>`` spelling
# and Gemma 4's ``<|token>``/``<token|>`` pair spelling.  Keep this broad for
# output-leak checks, while render_checks below counts the exact native pairs.
CONTROL_TOKEN_RE = re.compile(
    r"<\|[^>\n]+(?:\|)?>|"
    r"<[^>\n]+\|>|"
    r"<(?:start|end)_of_[^>\n]+>"
)
VALID_FINISH_REASONS = {"stop", "eos", "end_turn"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="milliseconds")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False) + "\n")
        fh.flush()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_materials(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = Path(config["materials_manifest"])
    selection_path = Path(config["materials_selection"])
    frozen_path = Path(config["frozen_text"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    text_rows = [json.loads(line) for line in frozen_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return manifest, text_rows, selection["selected_images"]


def verify_material_gate(config: dict[str, Any]) -> dict[str, Any]:
    manifest, text_rows, selected_images = load_materials(config)
    if manifest.get("freeze_ready") is not True:
        raise RuntimeError("materials gate failed: freeze_ready is not true")
    planned = manifest.get("planned_cases", {})
    expected_counts = {"text": 30, "images": 4, "cases_total": 34, "translator_slots": 7, "planned_translator_case_cells": 238}
    for key, value in expected_counts.items():
        if planned.get(key) != value:
            raise RuntimeError(f"materials gate failed: planned_cases.{key}={planned.get(key)!r}, expected {value}")

    frozen_path = Path(config["frozen_text"])
    authoritative = Path(manifest["authoritative_source_bundle"]["path"]) / "text_materials.jsonl"
    frozen_sha = sha256_file(frozen_path)
    expected_sha = manifest["frozen_artifacts"]["frozen_text_sha256"]
    source_sha = manifest["authoritative_source_bundle"]["source_text_sha256"]
    if frozen_sha != expected_sha or frozen_sha != source_sha:
        raise RuntimeError(f"materials gate failed: frozen text SHA {frozen_sha} != expected {expected_sha}/{source_sha}")
    if frozen_path.read_bytes() != authoritative.read_bytes():
        raise RuntimeError("materials gate failed: frozen text is not byte-identical to authoritative text_materials.jsonl")
    if len(text_rows) != 30 or len({x.get("item_id") for x in text_rows}) != 30:
        raise RuntimeError("materials gate failed: text records are not exactly 30 unique item_ids")

    expected_item_hashes = manifest["text_inventory"]["source_text_hashes_sha256_by_item"]
    for row in text_rows:
        item_id = row.get("item_id")
        source = row.get("source_en")
        if not isinstance(source, str) or sha256_bytes(source.encode("utf-8")) != expected_item_hashes.get(item_id):
            raise RuntimeError(f"materials gate failed: source hash mismatch for {item_id}")

    selection_count = len(selected_images)
    if selection_count != 4:
        raise RuntimeError(f"materials gate failed: selected image count is not 4 ({selection_count})")
    selected_ids = [x.get("image_id") for x in selected_images]
    if selected_ids != ["img-01", "img-04", "img-09", "img-10"]:
        raise RuntimeError(f"materials gate failed: selected image ids {selected_ids}")
    expected_image_hashes = manifest["image_freeze"]["selected_png_sha256"]
    image_evidence: list[dict[str, Any]] = []
    for image in selected_images:
        path = Path(image["source_absolute_path"])
        if not path.is_file():
            raise RuntimeError(f"materials gate failed: missing image {path}")
        actual = sha256_file(path)
        if actual != image["sha256"] or actual != expected_image_hashes[image["image_id"]]:
            raise RuntimeError(f"materials gate failed: image hash mismatch for {image['image_id']}")
        image_evidence.append({"image_id": image["image_id"], "path": str(path), "sha256": actual, "ground_truth_status": image.get("ground_truth_status")})

    gate = {
        "verified_at": now(),
        "manifest": str(Path(config["materials_manifest"])),
        "freeze_ready": True,
        "frozen_text_sha256": frozen_sha,
        "authoritative_text_path": str(authoritative),
        "frozen_text_byte_identical": True,
        "text_count": len(text_rows),
        "image_count": len(selected_images),
        "case_count": len(text_rows) + len(selected_images),
        "planned_model_case_cells": planned["planned_translator_case_cells"],
        "selected_images": image_evidence,
        "source_exactness_classification": manifest.get("text_exactness_qa", {}).get("status"),
        "known_source_quirks_retained": manifest.get("text_exactness_qa", {}).get("known_source_quirks_retained", []),
    }
    write_json(ROOT / "materials_gate_verified.json", gate)
    return gate


def formal_endpoint_preflight(base: str) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "checked_at": now(),
        "base": base,
        "read_only": True,
        "interpretation": "A sandbox transport failure is read-context inconclusive; this probe never changes the formal service.",
        "endpoints": {},
    }
    for suffix in ("/health", "/props", "/v1/models"):
        result = http_get(base.rstrip("/") + suffix, timeout=5)
        evidence["endpoints"][suffix] = result
    write_json(RAW_ROOT / "formal_endpoint_readonly_preflight.json", evidence)
    return evidence


def http_get(url: str, timeout: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            body: Any
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw.decode("utf-8", errors="replace")
            return {"http_status": response.status, "elapsed_seconds": time.perf_counter() - t0, "headers": dict(response.headers), "body": body, "body_sha256": sha256_bytes(raw)}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw.decode("utf-8", errors="replace")
        return {"http_status": exc.code, "elapsed_seconds": time.perf_counter() - t0, "headers": dict(exc.headers), "body": body, "body_sha256": sha256_bytes(raw)}
    except Exception as exc:  # transport evidence is part of the protocol
        return {"http_status": None, "elapsed_seconds": time.perf_counter() - t0, "error": f"{type(exc).__name__}: {exc}"}


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 1800) -> dict[str, Any]:
    body_bytes = json_bytes(payload)
    t0 = time.perf_counter()
    request = urllib.request.Request(url, data=body_bytes, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            try:
                parsed: Any = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw_body": raw.decode("utf-8", errors="replace")}
            return {"http_status": response.status, "elapsed_seconds": time.perf_counter() - t0, "headers": dict(response.headers), "body": parsed, "body_sha256": sha256_bytes(raw), "request_sha256": sha256_bytes(body_bytes)}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw_body": raw.decode("utf-8", errors="replace")}
        return {"http_status": exc.code, "elapsed_seconds": time.perf_counter() - t0, "headers": dict(exc.headers), "body": parsed, "body_sha256": sha256_bytes(raw), "request_sha256": sha256_bytes(body_bytes)}
    except Exception as exc:
        return {"http_status": None, "elapsed_seconds": time.perf_counter() - t0, "error": f"{type(exc).__name__}: {exc}", "request_sha256": sha256_bytes(body_bytes)}


def run_command_version(binary: str) -> str:
    result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    return (result.stdout + result.stderr).strip()


def runtime_version_evidence(config: dict[str, Any], observed: str) -> dict[str, Any]:
    expected = str(config.get("llama_version_expected", ""))
    expected_build = re.search(r"build\s+(\d+)", expected)
    expected_commit = re.search(r"commit\s+([0-9a-f]+)", expected, flags=re.IGNORECASE)
    observed_build = re.search(r"build\s+(\d+)", observed)
    observed_commit = re.search(r"commit\s+([0-9a-f]+)", observed, flags=re.IGNORECASE)
    build_expected = expected_build.group(1) if expected_build else None
    commit_expected = expected_commit.group(1) if expected_commit else None
    build_observed = observed_build.group(1) if observed_build else None
    commit_observed = observed_commit.group(1) if observed_commit else None
    return {
        "checked_at": now(),
        "expected": expected,
        "observed": observed,
        "expected_build": build_expected,
        "observed_build": build_observed,
        "expected_commit": commit_expected,
        "observed_commit": commit_observed,
        "pass": bool(build_expected and commit_expected and build_expected == build_observed and commit_expected == commit_observed),
    }


def props_version_evidence(config: dict[str, Any], props_body: Any) -> dict[str, Any]:
    expected = runtime_version_evidence(config, str(config.get("llama_version_expected", "")))
    body = props_body if isinstance(props_body, dict) else {}
    build_info = body.get("build_info")
    expected_build = expected.get("expected_build")
    expected_commit = expected.get("expected_commit")
    observed_text = str(build_info or "")
    return {
        "checked_at": now(),
        "build_info": build_info,
        "expected_build": expected_build,
        "expected_commit": expected_commit,
        "pass": bool(expected_build and expected_commit and observed_text == f"b{expected_build}-{expected_commit}"),
    }


def vm_headroom(required_bytes: int = 32 * GIB, load_estimate_bytes: int = 0, phase: str = "preload") -> dict[str, Any]:
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
    text = result.stdout
    page_match = re.search(r"page size of (\d+) bytes", text)
    page_size = int(page_match.group(1)) if page_match else 4096
    pages: dict[str, int] = {}
    for name, value in re.findall(r"^Pages ([^:]+):\s+(\d+)", text, flags=re.MULTILINE):
        pages[name.lower().replace(" ", "_")] = int(value)
    # This is deliberately conservative: purgeable pages may overlap with
    # inactive accounting, so use only free/inactive/speculative pages.
    available_pages = sum(pages.get(key, 0) for key in ("free", "inactive", "speculative"))
    available_bytes = available_pages * page_size
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


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_port_closed(port: int, timeout: float = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_is_open(port):
            return True
        time.sleep(0.25)
    return not port_is_open(port)


def redacted(value: Any) -> Any:
    if isinstance(value, dict):
        out = {key: redacted(item) for key, item in value.items()}
        if "image_url" in out and isinstance(out["image_url"], dict) and "url" in out["image_url"]:
            url = out["image_url"]["url"]
            out["image_url"] = {"url": "<redacted-local-image-data-url>", "data_url_sha256": sha256_bytes(str(url).encode("utf-8"))}
        return out
    if isinstance(value, list):
        return [redacted(item) for item in value]
    return value


def import_gguf_reader() -> Any:
    path = Path("/benchmark/translation_benchmark_20260916_luna/extract_gguf_metadata.py")
    spec = importlib.util.spec_from_file_location("gguf_metadata_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GGUF metadata reader {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gguf_metadata_without_hash(path: Path) -> dict[str, Any]:
    module = import_gguf_reader()
    original = Path.read_bytes
    try:
        Path.read_bytes = lambda self: b""  # type: ignore[assignment]
        return module.read_gguf(str(path))
    finally:
        Path.read_bytes = original  # type: ignore[assignment]


def build_inventory(config: dict[str, Any], models: list[dict[str, Any]], server_version: str) -> dict[str, Any]:
    inventory: dict[str, Any] = {"created_at": now(), "server_binary": config["llama_server"], "server_version": server_version, "models": {}}
    for model in models:
        model_path = Path(model["model"])
        mmproj_path = Path(model["mmproj"])
        if not model_path.is_file() or not mmproj_path.is_file():
            raise RuntimeError(f"missing configured model/mmproj for {model['id']}")
        model_sha = sha256_file(model_path)
        mmproj_sha = sha256_file(mmproj_path)
        model_meta = gguf_metadata_without_hash(model_path)
        mmproj_meta = gguf_metadata_without_hash(mmproj_path)
        model_metadata = model_meta.get("metadata", {})
        mmproj_metadata = mmproj_meta.get("metadata", {})
        embedded = model_metadata.get("tokenizer.chat_template") or ""
        template_sha = sha256_bytes(embedded.encode("utf-8")) if embedded else None
        adapter_sha = sha256_file(Path(model["template_file"])) if model.get("template_file") else None
        inventory["models"][model["id"]] = {
            "model_path": str(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "model_sha256": model_sha,
            "mmproj_path": str(mmproj_path),
            "mmproj_size_bytes": mmproj_path.stat().st_size,
            "mmproj_sha256": mmproj_sha,
            "model_metadata": {key: model_metadata.get(key) for key in ("general.architecture", "general.name", "general.context_length", "general.file_type", "general.quantization_version", "general.size_label")},
            "mmproj_metadata": {key: mmproj_metadata.get(key) for key in ("general.architecture", "general.name", "general.type", "general.finetune", "general.size_label", "clip.projector_type")},
            "embedded_template_present": bool(embedded),
            "embedded_template_bytes": len(embedded.encode("utf-8")) if embedded else 0,
            "embedded_template_sha256": template_sha,
            "adapter_template_path": str(model["template_file"]) if model.get("template_file") else None,
            "adapter_template_sha256": adapter_sha,
        }
        print(f"inventory {model['id']} model_sha={model_sha[:12]} mmproj_sha={mmproj_sha[:12]} embedded_template={template_sha}", flush=True)
    write_json(RAW_ROOT / "model_inventory.json", inventory)
    return inventory


def build_cases(text_rows: list[dict[str, Any]], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in text_rows:
        cases.append({
            "case_id": row["item_id"],
            "input_type": "text",
            "source_id": row["source_id"],
            "source_en": row["source_en"],
            "source_sha256": sha256_bytes(row["source_en"].encode("utf-8")),
            "source_metadata": {key: row.get(key) for key in ("domain", "source_url", "source_title", "source_section", "source_locator", "text_kind", "topic", "expected_features")},
            "image_path": None,
            "image_sha256": None,
            "image_ground_truth_en": None,
            "image_ground_truth_status": None,
        })
    for image in images:
        gt = image.get("ground_truth_text")
        if isinstance(gt, list):
            gt_text = "\n".join(gt)
        else:
            gt_text = image.get("provisional_reference_text") or ""
        cases.append({
            "case_id": image["image_id"],
            "input_type": "image",
            "source_id": image["image_id"],
            "source_en": None,
            "source_sha256": image["sha256"],
            "source_metadata": {key: image.get(key) for key in ("role", "layout_stratum", "image_kind", "source_cues", "source_pdf_id", "source_pdf_page", "source_pdf_sha256")},
            "image_path": image["source_absolute_path"],
            "image_sha256": image["sha256"],
            "image_ground_truth_en": gt_text,
            "image_ground_truth_status": image.get("ground_truth_status"),
        })
    return cases


def image_data_url(path: str) -> str:
    raw = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def make_payload(config: dict[str, Any], model: dict[str, Any], case: dict[str, Any], technical: bool = False) -> dict[str, Any]:
    fixed = config["fixed_generation"]
    instruction = config["shared_instruction"]
    if technical:
        source = "TECHNICAL_MARKER_20261023"
        image_instruction = instruction + " Technical marker smoke only."
    elif case["input_type"] == "text":
        source = case["source_en"]
        image_instruction = instruction
    else:
        source = None
        image_instruction = instruction

    if case["input_type"] == "text":
        if model["family"] == "translategemma":
            # The validated TG adapter supplies the native professional
            # translation instruction and fixed en->zh-TW fields.
            messages = [{"role": "user", "content": source}]
        else:
            messages = [{"role": "user", "content": image_instruction + "\n\n" + source}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": image_instruction},
            {"type": "image_url", "image_url": {"url": image_data_url(case["image_path"])}}
        ]}]
    payload: dict[str, Any] = {
        "model": model["alias"],
        "messages": messages,
        "max_tokens": fixed["max_tokens"],
        "temperature": fixed["temperature"],
        "top_p": fixed["top_p"],
        "seed": fixed["seed"],
        "stream": fixed["stream"],
        "reasoning_effort": fixed["reasoning_effort"],
        "reasoning_format": model["reasoning_format"],
        "cache_prompt": fixed["cache_prompt"],
        "verbose": True,
    }
    return payload


def server_command(config: dict[str, Any], model: dict[str, Any]) -> list[str]:
    fixed = config["fixed_generation"]
    command = [
        config["llama_server"],
        "--model", model["model"],
        "--mmproj", model["mmproj"],
        "--host", "127.0.0.1",
        "--port", str(model["port"]),
        "--alias", model["alias"],
        "--jinja",
        "--no-warmup",
        "--reasoning", "off",
        "--reasoning-format", model["reasoning_format"],
        "--ctx-size", str(fixed["ctx_size"]),
        "--parallel", str(fixed["parallel"]),
        "--batch-size", str(fixed["batch_size"]),
        "--ubatch-size", str(fixed["ubatch_size"]),
        "--no-cont-batching",
        "--no-cache-prompt",
        "--kv-unified",
        "--spec-type", "none",
        "--seed", str(fixed["seed"]),
        "--n-gpu-layers", str(fixed["n_gpu_layers"]),
        "--flash-attn", str(fixed["flash_attn"]),
        "--metrics",
    ]
    if model.get("template_file"):
        command.extend(["--chat-template-file", model["template_file"]])
    if model.get("chat_template_kwargs") is not None:
        command.extend(["--chat-template-kwargs", json.dumps(model["chat_template_kwargs"], separators=(",", ":"))])
    return command


def wait_for_health(base: str, proc: subprocess.Popen[Any], timeout: float = 1800) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited during load with code {proc.returncode}")
        last = http_get(base + "/health", timeout=5)
        if last.get("http_status") == 200:
            return last
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {base}/health; last={last}")


class RunningServer:
    def __init__(self, config: dict[str, Any], model: dict[str, Any], batch_id: str, inventory: dict[str, Any]):
        self.config = config
        self.model = model
        self.batch_id = batch_id
        self.inventory = inventory
        self.port = int(model["port"])
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen[Any] | None = None
        self.stdout_handle: Any = None
        self.stderr_handle: Any = None
        self.headroom: dict[str, Any] | None = None
        self.headroom_after_load: dict[str, Any] | None = None
        model_identity = inventory["models"][model["id"]]
        self.load_estimate_bytes = int(model_identity["model_size_bytes"]) + int(model_identity["mmproj_size_bytes"])
        self.preload_required_bytes = self.load_estimate_bytes + 32 * GIB
        self.start_time: str | None = None
        self.command = server_command(config, model)

    def start(self) -> dict[str, Any]:
        if port_is_open(self.port):
            raise RuntimeError(f"port guard failed: {self.port} is already open")
        self.headroom = vm_headroom(required_bytes=self.preload_required_bytes, load_estimate_bytes=self.load_estimate_bytes, phase="preload")
        if not self.headroom["pass"]:
            raise RuntimeError(
                f"RAM preload headroom guard failed: {self.headroom['available_gib_conservative']:.2f} GiB < "
                f"load_estimate+32={self.headroom['required_gib']:.2f} GiB"
            )
        raw_dir = RAW_ROOT / self.model["id"]
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_handle = (raw_dir / f"{self.batch_id}.server.stdout.log").open("w", encoding="utf-8")
        self.stderr_handle = (raw_dir / f"{self.batch_id}.server.stderr.log").open("w", encoding="utf-8")
        self.start_time = now()
        self.proc = subprocess.Popen(
            self.command,
            cwd=str(ROOT.parent),
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            text=True,
            start_new_session=True,
        )
        try:
            health = wait_for_health(self.base, self.proc)
        except Exception:
            self.stop()
            raise
        self.headroom_after_load = vm_headroom(required_bytes=32 * GIB, load_estimate_bytes=self.load_estimate_bytes, phase="after_load")
        if not self.headroom_after_load["pass"]:
            self.stop()
            raise RuntimeError(
                f"RAM after-load headroom guard failed: {self.headroom_after_load['available_gib_conservative']:.2f} GiB < 32 GiB"
            )
        endpoints = {path: http_get(self.base + path, timeout=20) for path in ("/health", "/props", "/v1/models", "/slots")}
        server_version = run_command_version(self.config["llama_server"])
        version_evidence = runtime_version_evidence(self.config, server_version)
        props_body = endpoints.get("/props", {}).get("body") if isinstance(endpoints.get("/props"), dict) else None
        props_evidence = props_version_evidence(self.config, props_body)
        meta = {
            "batch_id": self.batch_id,
            "model_id": self.model["id"],
            "display_name": self.model["display_name"],
            "port": self.port,
            "pid": self.proc.pid,
            "started_at": self.start_time,
            "server_binary": self.config["llama_server"],
            "server_version": server_version,
            "runtime_version_evidence": version_evidence,
            "props_version_evidence": props_evidence,
            "reasoning_configuration_evidence": {
                "server_reasoning": "off",
                "model_reasoning_format": self.model["reasoning_format"],
                "chat_template_kwargs": self.model.get("chat_template_kwargs"),
                "deprecated_enable_thinking_warning_expected": bool(self.model.get("chat_template_kwargs", {}).get("enable_thinking") is False) if isinstance(self.model.get("chat_template_kwargs"), dict) else False,
                "deprecated_enable_thinking_warning_note": "Gemma4 native template kwargs retain enable_thinking=false for compatibility; runtime may warn that reasoning off is the preferred control. Gate uses observed reasoning_in_content/reasoning_content, not this warning.",
            },
            "command": self.command,
            "headroom": self.headroom,
            "headroom_after_load": self.headroom_after_load,
            "headroom_gate": {
                "preload_required_bytes": self.preload_required_bytes,
                "preload_required_gib": self.preload_required_bytes / GIB,
                "after_load_required_bytes": 32 * GIB,
                "preload_pass": bool(self.headroom.get("pass")),
                "after_load_pass": bool(self.headroom_after_load.get("pass")),
            },
            "health_ready": health,
            "endpoints_pre": endpoints,
            "model_identity": self.inventory["models"][self.model["id"]],
            "config_sha256": sha256_bytes(json_bytes(self.config)),
        }
        write_json(RAW_ROOT / self.model["id"] / f"{self.batch_id}.server_preflight.json", meta)
        if not version_evidence["pass"] or not props_evidence["pass"]:
            self.stop()
            raise RuntimeError(
                f"runtime version gate failed: command_pass={version_evidence['pass']} props_pass={props_evidence['pass']}"
            )
        return meta

    def stop(self) -> dict[str, Any]:
        stopped_at = now()
        returncode = self.proc.returncode if self.proc else None
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
        if self.proc is not None:
            returncode = self.proc.returncode
        closed = wait_port_closed(self.port, timeout=30)
        if self.stdout_handle:
            self.stdout_handle.close()
        if self.stderr_handle:
            self.stderr_handle.close()
        evidence = {"batch_id": self.batch_id, "model_id": self.model["id"], "pid": self.proc.pid if self.proc else None, "port": self.port, "stopped_at": stopped_at, "returncode": returncode, "port_closed": closed}
        write_json(RAW_ROOT / self.model["id"] / f"{self.batch_id}.server_cleanup.json", evidence)
        return evidence


def apply_template_evidence(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Capture the non-generation template rendering envelope for a smoke case."""
    template_payload = dict(payload)
    # Chat completions uses this default, but making it explicit makes the
    # documented /apply-template evidence unambiguous and reproducible.
    template_payload["add_generation_prompt"] = True
    response = http_post_json(base + "/apply-template", template_payload)
    body = response.get("body") if isinstance(response.get("body"), dict) else {}
    prompt = body.get("prompt") if isinstance(body.get("prompt"), str) else ""
    return {
        "endpoint": base + "/apply-template",
        "request_redacted": redacted(template_payload),
        "http_envelope": {key: value for key, value in response.items() if key != "body"},
        "response_body": body,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")) if prompt else None,
        "prompt_chars": len(prompt),
    }


def _role_positions(prompt: str) -> dict[str, list[int]]:
    patterns = {
        "user": r"(?:<start_of_turn>|<\|turn>)user\b",
        "model": r"(?:<start_of_turn>|<\|turn>)model\b",
    }
    return {role: [match.start() for match in re.finditer(pattern, prompt)] for role, pattern in patterns.items()}


def _image_marker_evidence(prompt: str) -> dict[str, Any]:
    # These are the actual native pairs observed in this runtime.  Gemma 4
    # uses <|image><image|>; TranslateGemma's adapter uses start/end_of_image.
    pair_specs = (
        ("translategemma_start_end", "<start_of_image>", "<end_of_image>"),
        ("gemma4_pipe_pair", "<|image>", "<image|>"),
        ("legacy_pipe_pair", "<|image|>", "<|image|>"),
        ("angle_pair", "<image>", "</image>"),
    )
    pairs: dict[str, Any] = {}
    valid_pairs: list[str] = []
    for name, start, end in pair_specs:
        start_count = prompt.count(start)
        end_count = prompt.count(end)
        ordered = start_count == 1 and end_count == 1 and prompt.find(start) < prompt.find(end)
        pairs[name] = {"start": start, "end": end, "start_count": start_count, "end_count": end_count, "ordered": ordered}
        if ordered:
            valid_pairs.append(name)
    media_markers = re.findall(r"<__media(?:_[^>\n]+)?__>", prompt)
    return {
        "pairs": pairs,
        "valid_pair_kinds": valid_pairs,
        "media_markers": media_markers,
        "exact_single_media_marker": len(media_markers) == 1,
        "exact_native_image_marker": bool(valid_pairs) or len(media_markers) == 1,
    }


def render_checks(
    model: dict[str, Any],
    case: dict[str, Any],
    request_payload: dict[str, Any],
    response_body: Any,
    technical: bool,
    template_apply: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_text = json.dumps(request_payload, ensure_ascii=False)
    response_body = response_body if isinstance(response_body, dict) else {}
    choice = (response_body.get("choices") or [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    verbose = response_body.get("_verbose") or response_body.get("__verbose") or {}
    verbose = verbose if isinstance(verbose, dict) else {}
    verbose_prompt = verbose.get("prompt") if isinstance(verbose.get("prompt"), str) else ""
    apply_prompt = template_apply.get("prompt", "") if isinstance(template_apply, dict) else ""
    if not isinstance(apply_prompt, str):
        apply_prompt = ""
    prompt = verbose_prompt or apply_prompt
    if verbose_prompt and apply_prompt:
        prompt_evidence_source = "verbose_and_apply_match" if verbose_prompt == apply_prompt else "verbose_and_apply_mismatch"
    elif verbose_prompt:
        prompt_evidence_source = "verbose_only"
    elif apply_prompt:
        prompt_evidence_source = "apply_only"
    else:
        prompt_evidence_source = "none"

    role_positions = _role_positions(prompt)
    role_counts = {role: len(positions) for role, positions in role_positions.items()}
    exactly_one_user_model = role_counts == {"user": 1, "model": 1}
    generation_marker = exactly_one_user_model and role_positions["model"][0] > role_positions["user"][0]
    generation_hint = verbose.get("generation_prompt")
    if not isinstance(generation_hint, str):
        settings = verbose.get("generation_settings")
        generation_hint = settings.get("generation_prompt") if isinstance(settings, dict) else None
    generation_hint_present = bool(isinstance(generation_hint, str) and generation_hint and generation_hint in prompt)
    image_evidence = _image_marker_evidence(prompt)
    image_marker = image_evidence["exact_native_image_marker"]
    source = "TECHNICAL_MARKER_20261023" if technical else (case.get("source_en") or "")
    source_occurrences = prompt.count(source) if case["input_type"] == "text" and source else 0

    messages = request_payload.get("messages")
    request_message = messages[0] if isinstance(messages, list) and messages and isinstance(messages[0], dict) else {}
    request_shape: dict[str, Any] = {
        "messages_is_list": isinstance(messages, list),
        "message_count": len(messages) if isinstance(messages, list) else None,
        "first_role": request_message.get("role"),
    }
    if case["input_type"] == "text":
        request_content = request_message.get("content")
        request_shape.update({
            "text_content_is_string": isinstance(request_content, str),
            "text_content_source_occurrences": request_content.count(source) if isinstance(request_content, str) and source else 0,
            "text_request_one_user_string": isinstance(messages, list) and len(messages) == 1 and request_message.get("role") == "user" and isinstance(request_content, str),
            "tg_source_only_request": model["family"] != "translategemma" or (
                isinstance(messages, list) and len(messages) == 1 and request_message.get("role") == "user" and request_content == source
            ),
        })
    else:
        content_parts = request_message.get("content")
        part_types = [part.get("type") if isinstance(part, dict) else None for part in content_parts] if isinstance(content_parts, list) else []
        request_shape.update({
            "image_content_is_list": isinstance(content_parts, list),
            "image_content_part_types": part_types,
            "image_request_one_text_one_image": part_types.count("text") == 1 and part_types.count("image_url") == 1 and len(part_types) == 2,
            "image_request_messages_shape_valid": isinstance(messages, list) and len(messages) == 1 and request_message.get("role") == "user",
        })

    if model["family"] == "translategemma":
        adapter_required = (
            "You are a professional English (en) to Chinese (zh-TW) translator.",
            "Chinese (zh-TW)",
            "Produce only the Chinese translation",
            "without any additional explanations",
        )
        if case["input_type"] == "image":
            adapter_required = adapter_required + ("provided image",)
        adapter_complete = all(marker in prompt for marker in adapter_required)
    else:
        adapter_required = ()
        adapter_complete = True

    content = message.get("content") if isinstance(message, dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    content_text = content if isinstance(content, str) else ""
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    reasoning_off_violation = bool(reasoning_text.strip())
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    template_apply_status = template_apply.get("http_envelope", {}).get("http_status") if isinstance(template_apply, dict) else None
    template_apply_success = template_apply_status == 200 and bool(apply_prompt)
    render_failures: list[str] = []
    if not prompt:
        render_failures.append("rendered_prompt_missing")
    if not exactly_one_user_model:
        render_failures.append("expected_exactly_one_user_and_one_model_generation")
    if not generation_marker:
        render_failures.append("model_generation_marker_missing_or_misordered")
    if not generation_hint_present and verbose_prompt:
        render_failures.append("verbose_generation_hint_missing_from_prompt")
    if case["input_type"] == "text":
        if source_occurrences != 1:
            render_failures.append(f"source_occurrences={source_occurrences}")
        if not request_shape.get("text_request_one_user_string"):
            render_failures.append("text_request_shape_invalid")
        if not request_shape.get("tg_source_only_request"):
            render_failures.append("translategemma_request_not_source_only")
    else:
        if not request_shape.get("image_request_one_text_one_image") or not request_shape.get("image_request_messages_shape_valid"):
            render_failures.append("image_request_not_exactly_one_text_plus_one_image")
        if not image_marker:
            render_failures.append("exact_native_image_marker_pair_missing")
    if not adapter_complete:
        render_failures.append("translategemma_adapter_instruction_incomplete")
    if reasoning_off_violation:
        render_failures.append("reasoning_content_nonempty_while_reasoning_off")

    return {
        "verbose_evidence_present": bool(verbose),
        "template_apply_evidence_present": isinstance(template_apply, dict),
        "template_apply_success": template_apply_success,
        "template_apply_http_status": template_apply_status,
        "template_apply_request_sha256": (template_apply.get("http_envelope", {}).get("request_sha256") if isinstance(template_apply, dict) else None),
        "template_apply_response_body_sha256": (template_apply.get("http_envelope", {}).get("body_sha256") if isinstance(template_apply, dict) else None),
        "template_apply_prompt_sha256": sha256_bytes(apply_prompt.encode("utf-8")) if apply_prompt else None,
        "template_apply_prompt_chars": len(apply_prompt),
        "prompt_evidence_source": prompt_evidence_source,
        "rendered_prompt_crosscheck_match": prompt_evidence_source in {"verbose_and_apply_match", "verbose_only", "apply_only"},
        "rendered_prompt_present": bool(prompt),
        "rendered_prompt_sha256": sha256_bytes(prompt.encode("utf-8")) if prompt else None,
        "rendered_prompt_chars": len(prompt),
        "rendered_prompt_role_counts": role_counts,
        "rendered_prompt_exactly_one_user_one_model": exactly_one_user_model,
        "rendered_prompt_source_occurrences": source_occurrences,
        "rendered_prompt_source_or_marker_present": source_occurrences == 1 if case["input_type"] == "text" else image_marker,
        "rendered_prompt_image_marker_present": image_marker,
        "rendered_prompt_image_marker_evidence": image_evidence,
        "rendered_prompt_generation_marker_present": generation_marker,
        "generation_hint": generation_hint,
        "generation_hint_present_in_prompt": generation_hint_present,
        "request_shape": request_shape,
        "request_control_token_leak": bool(CONTROL_TOKEN_RE.search(request_text)),
        "content_control_token_leak": bool(CONTROL_TOKEN_RE.search(content_text)),
        "reasoning_control_token_leak": bool(CONTROL_TOKEN_RE.search(reasoning_text)),
        "reasoning_off_violation": reasoning_off_violation,
        "finish_reason": finish_reason,
        "finish_reason_valid": finish_reason in VALID_FINISH_REASONS,
        "translategemma_adapter_required_markers": list(adapter_required),
        "translategemma_adapter_complete": adapter_complete,
        "native_template_owner": model["template_mode"],
        "technical_marker": technical,
        "image_native_capability_evidence": bool(image_marker and prompt and exactly_one_user_model),
        "render_gate_failures": render_failures,
        "render_gate_pass": not render_failures,
        "raw_generation_prompt": generation_hint,
    }


def one_attempt(base: str, config: dict[str, Any], model: dict[str, Any], case: dict[str, Any], inventory: dict[str, Any], technical: bool, attempt: int, retry_of: str | None) -> dict[str, Any]:
    payload = make_payload(config, model, case, technical=technical)
    request_redacted = redacted(payload)
    template_apply = apply_template_evidence(base, payload) if technical else None
    response = http_post_json(base + "/v1/chat/completions", payload)
    body = response.get("body")
    status_code = response.get("http_status")
    response_body = body if isinstance(body, dict) else {}
    choices = response_body.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    verbose = response_body.get("_verbose") or response_body.get("__verbose") or {}
    usage = response_body.get("usage") if isinstance(response_body.get("usage"), dict) else {}
    timings = response_body.get("timings") or (verbose.get("timings") if isinstance(verbose, dict) else {}) or {}
    content_text = content if isinstance(content, str) else ""
    if status_code != 200:
        status = "transport_error" if status_code is None else "http_error"
    elif not isinstance(content, str):
        status = "format_fail"
    elif not content.strip():
        status = "generation_empty"
    elif not (isinstance(verbose, dict) and verbose.get("prompt")) and not (isinstance(template_apply, dict) and template_apply.get("prompt")):
        status = "render_evidence_missing"
    elif choice.get("finish_reason") == "length" or verbose.get("truncated") is True:
        status = "truncated"
    elif choice.get("finish_reason") not in VALID_FINISH_REASONS:
        status = "finish_reason_invalid"
    elif isinstance(reasoning, str) and reasoning.strip():
        status = "reasoning_off_violation"
    elif CONTROL_TOKEN_RE.search(content_text) or CONTROL_TOKEN_RE.search(reasoning if isinstance(reasoning, str) else ""):
        status = "template_leak"
    else:
        status = "technical_ok" if technical else "ok"
    render = render_checks(model, case, payload, body, technical, template_apply=template_apply)
    if status in {"technical_ok", "ok"} and not render.get("render_gate_pass"):
        status = "technical_render_gate_fail" if technical else "render_gate_fail"
    return {
        "attempt": attempt,
        "retry_of": retry_of,
        "observed_at": now(),
        "case_id": case["case_id"],
        "input_type": case["input_type"],
        "model_id": model["id"],
        "model_alias": model["alias"],
        "model_identity": inventory["models"][model["id"]],
        "config_sha256": sha256_bytes(json_bytes(config)),
        "request_redacted": request_redacted,
        "template_apply": template_apply,
        "request_sha256": response.get("request_sha256"),
        "http_envelope": {key: value for key, value in response.items() if key != "body"},
        "response": body,
        "response_body_sha256": response.get("body_sha256"),
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "timings": timings,
        "verbose": verbose,
        "elapsed_seconds": response.get("elapsed_seconds"),
        "status": status,
        "render_checks": render,
        "source_id": case["source_id"],
        "source_sha256": case["source_sha256"],
        "image_sha256": case["image_sha256"],
        "image_ground_truth_en": case["image_ground_truth_en"],
        "image_ground_truth_status": case["image_ground_truth_status"],
    }


def request_with_technical_retries(base: str, config: dict[str, Any], model: dict[str, Any], case: dict[str, Any], inventory: dict[str, Any], technical: bool, max_retries: int) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    retry_of = f"{model['id']}::{case['case_id']}"
    for attempt in range(max_retries + 1):
        record = one_attempt(base, config, model, case, inventory, technical, attempt, retry_of if attempt else None)
        attempts.append(record)
        if record["status"] not in {"transport_error", "http_error"} or attempt >= max_retries:
            break
    selected = attempts[-1]
    return {"selected": selected, "attempts": attempts, "retry_count": max(0, len(attempts) - 1), "technical_retry_rule": "only transport/HTTP errors; original attempt retained"}


def smoke_model(config: dict[str, Any], model: dict[str, Any], inventory: dict[str, Any], technical_retries: int) -> dict[str, Any]:
    batch_id = f"smoke_{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}_{model['id']}"
    raw_dir = RAW_ROOT / model["id"]
    smoke_path = SMOKE_ROOT / f"{batch_id}.jsonl"
    if smoke_path.exists():
        raise RuntimeError(f"refusing to overwrite existing smoke raw evidence: {smoke_path}")
    technical_text = {"case_id": "technical-text-marker", "input_type": "text", "source_id": "technical-marker", "source_en": "TECHNICAL_MARKER_20261023", "source_sha256": sha256_bytes(b"TECHNICAL_MARKER_20261023"), "image_path": None, "image_sha256": None, "image_ground_truth_en": None, "image_ground_truth_status": "not_applicable"}
    technical_image_path = SMOKE_ROOT / "technical_marker.png"
    technical_image = {"case_id": "technical-image-marker", "input_type": "image", "source_id": "technical-marker-image", "source_en": None, "source_sha256": sha256_file(technical_image_path), "image_path": str(technical_image_path), "image_sha256": sha256_file(technical_image_path), "image_ground_truth_en": None, "image_ground_truth_status": "not_applicable"}
    records: list[dict[str, Any]] = []
    server: RunningServer | None = None
    preflight: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    try:
        server = RunningServer(config, model, batch_id, inventory)
        preflight = server.start()
        for case in (technical_text, technical_image):
            result = request_with_technical_retries(server.base, config, model, case, inventory, technical=True, max_retries=technical_retries)
            record = {"batch_id": batch_id, "server_pid": server.proc.pid if server.proc else None, "server_port": server.port, "preflight": preflight, **result}
            records.append(record)
            append_jsonl(smoke_path, record)
    except Exception as exc:
        failure = {"batch_id": batch_id, "model_id": model["id"], "status": "batch_start_fail", "error": f"{type(exc).__name__}: {exc}", "preflight": preflight}
        records.append({"selected": failure, "attempts": [failure], "retry_count": 0})
        append_jsonl(smoke_path, records[-1])
    finally:
        if server is not None:
            cleanup = server.stop()
    selected = [x.get("selected", {}) for x in records]
    text_rec = next((x for x in selected if x.get("input_type") == "text" or x.get("case_id") == "technical-text-marker"), {})
    image_rec = next((x for x in selected if x.get("input_type") == "image" or x.get("case_id") == "technical-image-marker"), {})
    text_render = text_rec.get("render_checks", {})
    image_render = image_rec.get("render_checks", {})
    text_ok = (
        text_rec.get("status") == "technical_ok"
        and text_render.get("render_gate_pass") is True
        and text_render.get("finish_reason_valid") is True
        and text_render.get("reasoning_off_violation") is not True
    )
    image_ok = (
        image_rec.get("status") == "technical_ok"
        and image_render.get("render_gate_pass") is True
        and image_render.get("finish_reason_valid") is True
        and image_render.get("reasoning_off_violation") is not True
    )
    probe_path = SMOKE_ROOT / f"{model['id']}.template_probe.summary.json"
    template_probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else {"text_template_ok": False, "image_template_ok": False, "status": "template_probe_missing"}
    summary = {
        "model_id": model["id"],
        "display_name": model["display_name"],
        "smoke_batch_id": batch_id,
        "checked_at": now(),
        "text_native_supported": text_ok,
        "image_native_supported": image_ok,
        "text_status": text_rec.get("status", "missing"),
        "image_status": image_rec.get("status", "missing"),
        "text_render_checks": text_rec.get("render_checks", {}),
        "image_render_checks": image_rec.get("render_checks", {}),
        "template_probe": template_probe,
        "preflight": preflight,
        "cleanup": cleanup,
        "raw_jsonl_path": str(smoke_path),
        "raw_jsonl_sha256": sha256_file(smoke_path) if smoke_path.is_file() else None,
        "model_identity": inventory["models"][model["id"]],
        "notes": "Smoke only; no frozen material was sent. image_native_supported is required before real image requests. Previous raw attempts are never overwritten.",
    }
    write_json(SMOKE_ROOT / f"{model['id']}.summary.v2.json", summary)
    return summary


def failure_case_row(config: dict[str, Any], model: dict[str, Any], case: dict[str, Any], inventory: dict[str, Any], status: str, error: str, smoke: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": model["id"], "model_name": model["display_name"], "model_alias": model["alias"], "case_id": case["case_id"], "input_type": case["input_type"],
        "source_id": case["source_id"], "source_en": case["source_en"] or "", "source_sha256": case["source_sha256"], "image_path": case["image_path"] or "", "image_sha256": case["image_sha256"] or "", "image_ground_truth_en": case["image_ground_truth_en"] or "", "image_ground_truth_status": case["image_ground_truth_status"] or "",
        "translation_zh_tw": "", "reasoning_content": "", "http_status": "", "finish_reason": "", "status": status, "elapsed_seconds": "", "prompt_tokens": "", "completion_tokens": "", "total_tokens": "", "prompt_tok_s": "", "generation_tok_s": "", "cache_tokens": "", "request_sha256": "", "response_body_sha256": "", "config_sha256": sha256_bytes(json_bytes(config)), "model_sha256": inventory["models"][model["id"]]["model_sha256"], "mmproj_sha256": inventory["models"][model["id"]]["mmproj_sha256"], "template_mode": model["template_mode"], "embedded_template_sha256": inventory["models"][model["id"]]["embedded_template_sha256"] or "", "adapter_template_sha256": inventory["models"][model["id"]]["adapter_template_sha256"] or "", "rendered_prompt_sha256": "", "retry_count": 0, "error": error, "smoke_text_supported": smoke.get("text_native_supported"), "smoke_image_supported": smoke.get("image_native_supported"), "case_record_path": "",
    }


def result_row(config: dict[str, Any], model: dict[str, Any], case: dict[str, Any], inventory: dict[str, Any], record: dict[str, Any], retry_count: int, smoke: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
    render = record.get("render_checks") if isinstance(record.get("render_checks"), dict) else {}
    return {
        "model_id": model["id"], "model_name": model["display_name"], "model_alias": model["alias"], "case_id": case["case_id"], "input_type": case["input_type"],
        "source_id": case["source_id"], "source_en": case["source_en"] or "", "source_sha256": case["source_sha256"], "image_path": case["image_path"] or "", "image_sha256": case["image_sha256"] or "", "image_ground_truth_en": case["image_ground_truth_en"] or "", "image_ground_truth_status": case["image_ground_truth_status"] or "",
        "translation_zh_tw": record.get("content") if isinstance(record.get("content"), str) else "", "reasoning_content": record.get("reasoning_content") if isinstance(record.get("reasoning_content"), str) else "", "http_status": record.get("http_envelope", {}).get("http_status", ""), "finish_reason": record.get("finish_reason", ""), "status": record.get("status", ""), "elapsed_seconds": record.get("elapsed_seconds", ""), "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n", "")), "completion_tokens": usage.get("completion_tokens", timings.get("predicted_n", "")), "total_tokens": usage.get("total_tokens", ""), "prompt_tok_s": timings.get("prompt_per_second", ""), "generation_tok_s": timings.get("predicted_per_second", ""), "cache_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", timings.get("cache_n", "")), "request_sha256": record.get("request_sha256", ""), "response_body_sha256": record.get("response_body_sha256", ""), "config_sha256": record.get("config_sha256", ""), "model_sha256": inventory["models"][model["id"]]["model_sha256"], "mmproj_sha256": inventory["models"][model["id"]]["mmproj_sha256"], "template_mode": model["template_mode"], "embedded_template_sha256": inventory["models"][model["id"]]["embedded_template_sha256"] or "", "adapter_template_sha256": inventory["models"][model["id"]]["adapter_template_sha256"] or "", "rendered_prompt_sha256": render.get("rendered_prompt_sha256", ""), "retry_count": retry_count, "error": "", "smoke_text_supported": smoke.get("text_native_supported"), "smoke_image_supported": smoke.get("image_native_supported"), "case_record_path": str(raw_path),
    }


CSV_FIELDS = [
    "model_id", "model_name", "model_alias", "case_id", "input_type", "source_id", "source_en", "source_sha256", "image_path", "image_sha256", "image_ground_truth_en", "image_ground_truth_status", "translation_zh_tw", "reasoning_content", "http_status", "finish_reason", "status", "elapsed_seconds", "prompt_tokens", "completion_tokens", "total_tokens", "prompt_tok_s", "generation_tok_s", "cache_tokens", "request_sha256", "response_body_sha256", "config_sha256", "model_sha256", "mmproj_sha256", "template_mode", "embedded_template_sha256", "adapter_template_sha256", "rendered_prompt_sha256", "retry_count", "error", "smoke_text_supported", "smoke_image_supported", "case_record_path"
]


def write_results_csv(rows: list[dict[str, Any]]) -> Path:
    path = RESULT_ROOT / "translation_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_results_md(config: dict[str, Any], gate: dict[str, Any], rows: list[dict[str, Any]], smoke_summaries: list[dict[str, Any]]) -> Path:
    lines = [
        "# 2026-10-23 七模型新素材翻譯執行報告",
        "",
        f"執行時間（最後更新）：{now()}",
        "",
        "本文件只報告 translation execution 與 transport/template evidence，不做品質打分、不呼叫 judges、不使用 LiteLLM/雲端，不修改 formal 8080 preset。圖片是端到端 image translation 輸入，不宣稱 OCR CER 或 OCR exactness。",
        "",
        f"Materials gate：`freeze_ready=true`；{gate['text_count']} text + {gate['image_count']} image = {gate['case_count']} cases；7 translators planned = {gate['planned_model_case_cells']} case cells。frozen text SHA-256 `{gate['frozen_text_sha256']}`，4 selected PNG hashes verified。",
        "",
        f"新 runtime：`{config['llama_server']}`；固定版本 evidence 保存在各 batch 的 preflight JSON。固定條件為 ctx 8192、temperature 0、top_p 1、seed 20261023、stream false、parallel 1、max_tokens 4096、reasoning off、cache false、MTP/speculative off。",
        "",
        "## Technical smoke",
        "",
        "Smoke 使用獨立 technical marker text/image，沒有送入 frozen translation material，也不納入 CSV。",
        "",
        "| model | text native | image native | text status | image status |",
        "|---|---:|---:|---|---|",
    ]
    for smoke in smoke_summaries:
        lines.append(f"| {smoke.get('display_name', smoke.get('model_id'))} | {smoke.get('text_native_supported')} | {smoke.get('image_native_supported')} | {smoke.get('text_status')} | {smoke.get('image_status')} |")
    lines.extend(["", "## Translation case counts", "", "| model | rows | ok | technical/unsupported/fail |", "|---|---:|---:|---:|"])
    for model in config["models"]:
        model_rows = [row for row in rows if row["model_id"] == model["id"]]
        ok = sum(row["status"] == "ok" for row in model_rows)
        other = len(model_rows) - ok
        lines.append(f"| {model['display_name']} | {len(model_rows)} | {ok} | {other} |")
    lines.extend([
        "",
        "`results/translation_results.csv` 是逐 model×case 的完整 CSV，含 source/translation、image ground-truth metadata（不送入 prompt）、timings/usage/cache、HTTP finish reason、template/config/model/mmproj hashes 與 raw record 路徑。raw JSONL、server endpoint snapshots、verbose non-generation prompts、stdout/stderr、PID/port cleanup evidence 位於 `raw/`。",
        "",
        "文字 source classification 依 freeze manifest 的最新 effective classification：text-16/text-28 的 bullet/list flattening 只作 formatting/source-derived note，不冒稱所有 30 筆均為 verbatim。既有 source quirks（例如 `..`、`.:`）沒有修正。",
        "",
        "品質評分與 judge pipeline 明確留待後續授權步驟。",
    ])
    path = RESULT_ROOT / "translation_results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_progress(status: str, detail: str) -> None:
    path = ROOT / "progress.md"
    path.write_text(
        "# 2026-10-23 新素材七模型翻譯進度\n\n"
        f"- Last update: `{now()}`\n"
        "- Materials gate: `READY`；30 text + 4 image frozen cases，text SHA and selected image hashes verified。\n"
        "- Formal endpoint: read-only preflight only; sandbox transport errors are inconclusive and do not imply formal service failure; no formal config/service change。\n"
        f"- Status: **{status}**。{detail}\n"
        "- Raw server/request/response/verbose/PID-port evidence is under `raw/`; CSV/MD are under `results/`。\n",
        encoding="utf-8",
    )


def run_generation(config: dict[str, Any], gate: dict[str, Any], cases: list[dict[str, Any]], models: list[dict[str, Any]], inventory: dict[str, Any], technical_retries: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, model in enumerate(models, start=1):
        smoke_path = SMOKE_ROOT / f"{model['id']}.summary.v2.json"
        if not smoke_path.is_file():
            smoke_path = SMOKE_ROOT / f"{model['id']}.summary.json"
        smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {"text_native_supported": False, "image_native_supported": False, "text_status": "smoke_missing", "image_status": "smoke_missing"}
        batch_id = f"generation_{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}_{model['id']}"
        raw_path = RAW_ROOT / model["id"] / f"{batch_id}.cases.jsonl"
        if raw_path.exists():
            raise RuntimeError(f"refusing to overwrite existing generation raw evidence: {raw_path}")
        print(f"generation batch {index}/{len(models)} model={model['id']} text_supported={smoke.get('text_native_supported')} image_supported={smoke.get('image_native_supported')}", flush=True)
        server: RunningServer | None = None
        try:
            server = RunningServer(config, model, batch_id, inventory)
            preflight = server.start()
            for case in cases:
                if case["input_type"] == "text" and not smoke.get("text_native_supported"):
                    row = failure_case_row(config, model, case, inventory, "N/A_not_supported", f"native text smoke failed: {smoke.get('text_status')}", smoke)
                    rows.append(row)
                    append_jsonl(raw_path, {"batch_id": batch_id, "case_id": case["case_id"], "status": row["status"], "error": row["error"], "attempted": False, "smoke": smoke})
                    continue
                if case["input_type"] == "image" and not smoke.get("image_native_supported"):
                    row = failure_case_row(config, model, case, inventory, "N/A_not_supported", f"native image smoke failed: {smoke.get('image_status')}; no OCR fallback", smoke)
                    rows.append(row)
                    append_jsonl(raw_path, {"batch_id": batch_id, "case_id": case["case_id"], "status": row["status"], "error": row["error"], "attempted": False, "smoke": smoke})
                    continue
                result = request_with_technical_retries(server.base, config, model, case, inventory, technical=False, max_retries=technical_retries)
                wrapper = {"batch_id": batch_id, "server_pid": server.proc.pid if server.proc else None, "server_port": server.port, "preflight": preflight, **result}
                append_jsonl(raw_path, wrapper)
                row = result_row(config, model, case, inventory, result["selected"], result["retry_count"], smoke, raw_path)
                rows.append(row)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"generation batch failure model={model['id']}: {error}", flush=True)
            for case in cases:
                row = failure_case_row(config, model, case, inventory, "batch_start_fail", error, smoke)
                rows.append(row)
                append_jsonl(raw_path, {"batch_id": batch_id, "case_id": case["case_id"], "status": row["status"], "error": error, "attempted": False, "smoke": smoke})
        finally:
            if server is not None:
                server.stop()
        update_progress("GENERATION_IN_PROGRESS", f"completed model batch {index}/{len(models)}: {model['id']}。Rows collected so far: {len(rows)}。")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "smoke", "generation", "full"), default="preflight")
    parser.add_argument("--only", default="", help="comma-separated model ids")
    parser.add_argument("--technical-retries", type=int, default=1)
    args = parser.parse_args()
    if args.technical_retries < 0 or args.technical_retries > 1:
        raise SystemExit("--technical-retries must be 0 or 1")
    config = load_config()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    gate = verify_material_gate(config)
    formal_endpoint_preflight(config["formal_endpoint"])
    models = config["models"]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        models = [model for model in models if model["id"] in wanted]
        if len(models) != len(wanted):
            unknown = wanted - {model["id"] for model in models}
            raise SystemExit(f"unknown model ids: {sorted(unknown)}")
    server_version = run_command_version(config["llama_server"])
    version_evidence = runtime_version_evidence(config, server_version)
    write_json(RAW_ROOT / "runtime_version_check.json", version_evidence)
    if not version_evidence["pass"]:
        raise SystemExit(f"runtime version gate failed: expected={version_evidence['expected']!r} observed={version_evidence['observed']!r}")
    print(f"materials gate READY cases={gate['case_count']} planned_cells={gate['planned_model_case_cells']}", flush=True)
    print(f"llama-server={server_version.replace(chr(10), ' | ')}", flush=True)
    inventory_path = RAW_ROOT / "model_inventory.json"
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        inventory = build_inventory(config, models, server_version)
    if args.phase == "preflight":
        update_progress("PREFLIGHT_COMPLETE", f"materials gate and model identity inventory verified for {len(models)} configured model(s). Technical marker smoke is pending.")
        return 0

    if not (SMOKE_ROOT / "technical_marker.png").is_file():
        raise SystemExit("missing technical_smoke/technical_marker.png; compile technical_smoke/marker.swift first")
    if args.phase in {"smoke", "full"}:
        smoke_summaries: list[dict[str, Any]] = []
        for index, model in enumerate(models, start=1):
            print(f"technical smoke {index}/{len(models)} model={model['id']}", flush=True)
            smoke_summaries.append(smoke_model(config, model, inventory, args.technical_retries))
        write_json(SMOKE_ROOT / "summary.json", {"checked_at": now(), "models": smoke_summaries})
        all_text = all(item.get("text_native_supported") for item in smoke_summaries)
        all_image = all(item.get("image_native_supported") for item in smoke_summaries)
        update_progress("TECHNICAL_SMOKE_COMPLETE", f"{sum(bool(item.get('text_native_supported')) for item in smoke_summaries)}/{len(smoke_summaries)} text and {sum(bool(item.get('image_native_supported')) for item in smoke_summaries)}/{len(smoke_summaries)} image native smoke routes passed. all_text={all_text}, all_image={all_image}.")
        print(f"technical smoke complete text={sum(bool(item.get('text_native_supported')) for item in smoke_summaries)}/{len(smoke_summaries)} image={sum(bool(item.get('image_native_supported')) for item in smoke_summaries)}/{len(smoke_summaries)}", flush=True)
        if args.phase == "smoke":
            return 0
    else:
        summary_path = SMOKE_ROOT / "summary.v2.json"
        if not summary_path.is_file():
            summary_path = SMOKE_ROOT / "summary.json"
        if not summary_path.is_file():
            raise SystemExit("generation requires technical_smoke/summary.v2.json (or summary.json); run smoke/derivation first")
        smoke_summaries = json.loads(summary_path.read_text(encoding="utf-8"))["models"]

    smoke_by_id = {item["model_id"]: item for item in smoke_summaries}
    if args.phase in {"generation", "full"}:
        missing_template_probes = [
            model["id"] for model in models
            if not (
                smoke_by_id.get(model["id"], {}).get("template_probe", {}).get("text_template_ok") is True
                and smoke_by_id.get(model["id"], {}).get("template_probe", {}).get("image_template_ok") is True
            )
        ]
        if missing_template_probes:
            raise SystemExit(f"generation requires passing non-generation /apply-template probes: {missing_template_probes}")
    cases = build_cases(*load_materials(config)[1:])
    rows = run_generation(config, gate, cases, models, inventory, args.technical_retries)
    csv_path = write_results_csv(rows)
    write_results_md(config, gate, rows, [smoke_by_id.get(model["id"], {}) for model in models])
    write_json(RESULT_ROOT / "translation_run_summary.json", {"completed_at": now(), "materials_gate": gate, "planned_rows": len(models) * len(cases), "actual_rows": len(rows), "models": [model["id"] for model in models], "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}, "csv": str(csv_path), "quality_scoring": "not_run_by_authorization"})
    update_progress("GENERATION_COMPLETE", f"wrote {len(rows)} model-case rows to results/translation_results.csv and translation_results.md. No quality score/judge was run; all raw evidence and unsupported cases are retained.")
    print(f"generation complete rows={len(rows)} csv={csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
