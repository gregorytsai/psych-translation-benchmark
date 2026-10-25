#!/usr/bin/env python3
"""Prepare one exact frozen-schema pairwise Gemini UI prompt without sending it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/benchmark/translation_benchmark_20261023_new_materials/judge_evaluation")
PUBLIC = ROOT / "public"
RAW = ROOT / "raw" / "cloud"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_task(pair_job_id: str) -> dict:
    path = PUBLIC / "pairwise" / "judge-05.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("pair_job_id") == pair_job_id:
            return row
    raise SystemExit(f"pair job id not found in frozen judge-05 packet: {pair_job_id}")


def render_task(task: dict) -> str:
    if task.get("source_kind") == "image":
        source = "Source image: use the attached PNG pixels only. No OCR transcript or image ground truth is supplied."
    else:
        source = f"English source:\n{task['source_en']}"
    a = task["candidate_a"]
    b = task["candidate_b"]
    return (
        "Task kind: pairwise\n"
        f"Pair ID: {task['pair_job_id']}\n"
        f"Orientation: {task['orientation']}\n"
        f"{source}\n\n"
        f"A ({a['blind_id']}):\n{a['target_zh_tw']}\n\n"
        f"B ({b['blind_id']}):\n{b['target_zh_tw']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-job-id", required=True)
    args = parser.parse_args()
    instructions_path = PUBLIC / "instructions.md"
    schema_path = PUBLIC / "schemas" / "pairwise_result.schema.json"
    task = load_task(args.pair_job_id)
    instructions = instructions_path.read_text(encoding="utf-8")
    schema_text = schema_path.read_text(encoding="utf-8")
    prompt = (
        instructions.rstrip()
        + "\n\nSELECTED FROZEN PUBLIC JSON SCHEMA (authoritative; reproduce this contract exactly):\n"
        + schema_text.rstrip()
        + "\n\nTASK PAYLOAD (judge only this one task):\n"
        + render_task(task)
        + "\n\nReturn exactly one UTF-8 JSON object conforming to the selected frozen public schema, with no Markdown, explanation, or extra keys."
    )
    prompt_bytes = prompt.encode("utf-8")
    request_id = f"pairwise-{args.pair_job_id}"
    out_dir = RAW / "gemini_ui_requests"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / f"{request_id}.prompt.txt"
    meta_path = out_dir / f"{request_id}.request.json"
    prompt_path.write_bytes(prompt_bytes)
    metadata = {
        "schema_version": "new-cloud-gemini-ui-request-v2",
        "request_id": request_id,
        "pair_job_id": args.pair_job_id,
        "judge_slot": "judge-05",
        "judge_id": "gemini-3.8-flash-high-antigravity-new",
        "route": "antigravity_native_ui",
        "requested_model": "Gemini 3.8 Flash High",
        "visible_model_required": "Gemini 3.8 Flash High",
        "fresh_conversation_required": True,
        "source_kind": task.get("source_kind"),
        "image_alias": task.get("image_alias"),
        "instruction_sha256": sha256_file(instructions_path),
        "public_schema_path": str(schema_path),
        "public_schema_sha256": sha256_file(schema_path),
        "task_sha256": sha256_bytes(json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")),
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "prompt_path": str(prompt_path),
        "schema_embedded": True,
        "conversation_url": None,
        "visible_model_label": None,
        "sent_at": None,
        "capture_path": None,
        "response_sha256": None,
        "status": "prepared_not_sent",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"request_id": request_id, "prompt_path": str(prompt_path), "meta_path": str(meta_path), "prompt_sha256": metadata["prompt_sha256"], "instruction_sha256": metadata["instruction_sha256"], "public_schema_sha256": metadata["public_schema_sha256"], "source_kind": task.get("source_kind"), "image_alias": task.get("image_alias")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
