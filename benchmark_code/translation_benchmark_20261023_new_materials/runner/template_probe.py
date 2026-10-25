#!/usr/bin/env python3
"""Capture documented /apply-template evidence without model generation.

This complements the immutable v1 completion smoke.  It does not call
/v1/chat/completions, so it cannot alter translation outputs or quality
evidence.  Each probe has its own server batch/raw JSONL and guarded cleanup.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "runner" / "run_translation.py"
CONFIG_PATH = ROOT / "configs" / "translation_config.json"
INVENTORY_PATH = ROOT / "raw" / "model_inventory.json"
RAW_ROOT = ROOT / "raw"
SMOKE_ROOT = ROOT / "technical_smoke"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("translation_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False) + "\n")
        fh.flush()


def main() -> int:
    runner = load_runner()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    marker = SMOKE_ROOT / "technical_marker.png"
    if not marker.is_file():
        raise RuntimeError(f"missing technical marker {marker}")
    summaries: list[dict[str, Any]] = []
    run_stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    for model in config["models"]:
        batch_id = f"template_probe_{run_stamp}_{model['id']}"
        raw_path = RAW_ROOT / model["id"] / f"{batch_id}.jsonl"
        if raw_path.exists():
            raise RuntimeError(f"refusing to overwrite existing probe raw {raw_path}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cases = [
            {"case_id": "technical-text-marker", "input_type": "text", "source_id": "technical-marker", "source_en": "TECHNICAL_MARKER_20261023", "source_sha256": runner.sha256_bytes(b"TECHNICAL_MARKER_20261023"), "image_path": None, "image_sha256": None, "image_ground_truth_en": None, "image_ground_truth_status": "not_applicable"},
            {"case_id": "technical-image-marker", "input_type": "image", "source_id": "technical-marker-image", "source_en": None, "source_sha256": runner.sha256_file(marker), "image_path": str(marker), "image_sha256": runner.sha256_file(marker), "image_ground_truth_en": None, "image_ground_truth_status": "not_applicable"},
        ]
        server = None
        preflight = None
        cleanup = None
        selected: list[dict[str, Any]] = []
        try:
            server = runner.RunningServer(config, model, batch_id, inventory)
            preflight = server.start()
            for case in cases:
                payload = runner.make_payload(config, model, case, technical=True)
                evidence = runner.apply_template_evidence(server.base, payload)
                body = {"_verbose": {}}
                checks = runner.render_checks(model, case, payload, body, True, template_apply=evidence)
                record = {
                    "batch_id": batch_id,
                    "server_pid": server.proc.pid if server.proc else None,
                    "server_port": server.port,
                    "preflight": preflight,
                    "case_id": case["case_id"],
                    "input_type": case["input_type"],
                    "template_apply": evidence,
                    "render_checks": checks,
                    "status": "template_ok" if evidence.get("http_envelope", {}).get("http_status") == 200 and checks.get("render_gate_pass") else "template_gate_fail",
                }
                append_jsonl(raw_path, record)
                selected.append(record)
        except Exception as exc:
            record = {"batch_id": batch_id, "model_id": model["id"], "status": "template_probe_batch_fail", "error": f"{type(exc).__name__}: {exc}", "preflight": preflight}
            append_jsonl(raw_path, record)
            selected.append(record)
        finally:
            if server is not None:
                cleanup = server.stop()
        text = next((r for r in selected if r.get("case_id") == "technical-text-marker"), {})
        image = next((r for r in selected if r.get("case_id") == "technical-image-marker"), {})
        summary = {
            "model_id": model["id"],
            "display_name": model["display_name"],
            "batch_id": batch_id,
            "request_replayed": False,
            "generation_performed": False,
            "raw_jsonl_path": str(raw_path),
            "raw_jsonl_sha256": sha256_file(raw_path),
            "text_template_ok": text.get("status") == "template_ok",
            "image_template_ok": image.get("status") == "template_ok",
            "text_http_status": text.get("template_apply", {}).get("http_envelope", {}).get("http_status"),
            "image_http_status": image.get("template_apply", {}).get("http_envelope", {}).get("http_status"),
            "text_render_checks": text.get("render_checks", {}),
            "image_render_checks": image.get("render_checks", {}),
            "preflight": preflight,
            "cleanup": cleanup,
            "notes": "Non-generation /apply-template evidence only; no frozen material and no completion request.",
        }
        write_json(SMOKE_ROOT / f"{model['id']}.template_probe.summary.json", summary)
        summaries.append(summary)
    write_json(SMOKE_ROOT / "template_probe.summary.json", {"checked_at": runner.now(), "generation_performed": False, "models": summaries})
    print(json.dumps({"text": sum(bool(x["text_template_ok"]) for x in summaries), "image": sum(bool(x["image_template_ok"]) for x in summaries), "models": len(summaries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
