#!/usr/bin/env python3
"""Prepare an exact frozen-schema prompt for one native Gemini UI job.

This helper never contacts a model.  It renders the public frozen instructions,
the selected public schema, and one public pointwise task into a single prompt,
then records the bytes and hashes under raw/cloud for the CUA operator to send.
"""

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


def load_task(job_id: str) -> dict:
    path = PUBLIC / "pointwise" / "judge-05.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("job_id") == job_id:
            return row
    raise SystemExit(f"job id not found in frozen judge-05 pointwise packet: {job_id}")


def render_task(task: dict) -> str:
    if task.get("source_kind") == "image":
        source = "Source image: use the attached PNG pixels only. No OCR transcript or image ground truth is supplied."
    else:
        source = f"English source:\n{task['source_en']}"
    return (
        "Task kind: pointwise\n"
        f"Item ID: {task['blind_source_id']}\n"
        f"Source ID: {task['blind_source_id']}\n"
        f"{source}\n\n"
        f"Anonymous candidate {task['blind_id']}:\n{task['target_zh_tw']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    instructions_path = PUBLIC / "instructions.md"
    schema_path = PUBLIC / "schemas" / "pointwise_result.schema.json"
    task = load_task(args.job_id)
    instructions = instructions_path.read_text(encoding="utf-8")
    schema_text = schema_path.read_text(encoding="utf-8")
    # Keep schema bytes exactly as frozen on disk; the prompt includes the full
    # selected public schema, not a weaker transport derivation.
    prompt = (
        instructions.rstrip()
        + "\n\nROUTE CONTRACT (authoritative for this native UI request): Do not call any tools, inspect or search files, access a workspace or repository, browse the web, or request permission for any external action. This is a self-contained judging request; all required text is below and any image must be judged only from the native attachment. If the displayed task is insufficient, return the schema's blocked/missing object instead of attempting retrieval.\n"
        + "\n\nSELECTED FROZEN PUBLIC JSON SCHEMA (authoritative; reproduce this contract exactly):\n"
        + schema_text.rstrip()
        + "\n\nCANONICAL OUTPUT DETAIL: For status \"ok\", include every listed result property, including block_reason as the empty string \"\". For status \"blocked\" or \"missing\", use the schema's non-empty block_reason rule."
        + "\n\nTASK PAYLOAD (judge only this one task):\n"
        + render_task(task)
        + "\n\nReturn exactly one UTF-8 JSON object conforming to the selected frozen public schema, with no Markdown, explanation, or extra keys."
    )
    prompt_bytes = prompt.encode("utf-8")
    request_id = f"pointwise-{args.job_id}"
    out_dir = RAW / "gemini_ui_requests"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / f"{request_id}.prompt.txt"
    meta_path = out_dir / f"{request_id}.request.json"
    prompt_path.write_bytes(prompt_bytes)
    metadata = {
        "schema_version": "new-cloud-gemini-ui-request-v2",
        "request_id": request_id,
        "job_id": args.job_id,
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
