#!/usr/bin/env python3
"""Re-derive native technical-smoke gates without re-querying the models.

The first smoke run returned HTTP 200/stop/non-empty responses for all seven
routes but the checker only recognised ``<|image|>``.  This validator reads
those immutable JSONL records, applies the corrected native marker/turn
validator, and writes versioned summaries.  It intentionally does not send a
request and does not overwrite the original JSONL or v1 summaries.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "runner" / "run_translation.py"
SMOKE = ROOT / "technical_smoke"
CONFIG = ROOT / "configs" / "translation_config.json"
INVENTORY = ROOT / "raw" / "model_inventory.json"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("translation_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    runner = load_runner()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    summaries: list[dict[str, Any]] = []
    validator_sha = sha256_file(RUNNER)
    for model in config["models"]:
        raw_path = SMOKE / f"{model['id']}.jsonl"
        if not raw_path.is_file():
            raise RuntimeError(f"missing immutable smoke raw: {raw_path}")
        records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = [record.get("selected", {}) for record in records]
        text_record = next((record for record in selected if record.get("case_id") == "technical-text-marker"), {})
        image_record = next((record for record in selected if record.get("case_id") == "technical-image-marker"), {})
        text_case = {"case_id": "technical-text-marker", "input_type": "text", "source_en": "TECHNICAL_MARKER_20261023"}
        image_case = {"case_id": "technical-image-marker", "input_type": "image", "source_en": None}
        text_render = runner.render_checks(model, text_case, text_record.get("request_redacted", {}), text_record.get("response"), True, template_apply=text_record.get("template_apply"))
        image_render = runner.render_checks(model, image_case, image_record.get("request_redacted", {}), image_record.get("response"), True, template_apply=image_record.get("template_apply"))
        text_ok = text_record.get("status") == "technical_ok" and text_render.get("render_gate_pass") is True and text_render.get("finish_reason_valid") is True and text_render.get("reasoning_off_violation") is not True
        image_ok = image_record.get("status") == "technical_ok" and image_render.get("render_gate_pass") is True and image_render.get("finish_reason_valid") is True and image_render.get("reasoning_off_violation") is not True
        probe_path = SMOKE / f"{model['id']}.template_probe.summary.json"
        template_probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else {"text_template_ok": False, "image_template_ok": False, "status": "template_probe_missing"}
        summary = {
            "model_id": model["id"],
            "display_name": model["display_name"],
            "source_raw_jsonl": str(raw_path),
            "source_raw_jsonl_sha256": sha256_file(raw_path),
            "validator": str(RUNNER),
            "validator_sha256": validator_sha,
            "derivation_only": True,
            "request_replayed": False,
            "text_native_supported": text_ok,
            "image_native_supported": image_ok,
            "text_status": text_record.get("status", "missing"),
            "image_status": image_record.get("status", "missing"),
            "text_render_checks": text_render,
            "image_render_checks": image_render,
            "template_probe": template_probe,
            "model_identity": inventory["models"][model["id"]],
            "notes": "Corrected v2 derivation from immutable v1 smoke raw; no model request was replayed. The v1 run did not capture /apply-template; verbose prompt is retained as supplemental render evidence.",
        }
        write_json(SMOKE / f"{model['id']}.summary.v2.json", summary)
        summaries.append(summary)
    write_json(SMOKE / "summary.v2.json", {"derivation_only": True, "validator_sha256": validator_sha, "models": summaries})
    print(json.dumps({"text": sum(bool(item["text_native_supported"]) for item in summaries), "image": sum(bool(item["image_native_supported"]) for item in summaries), "models": len(summaries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
