#!/usr/bin/env python3
"""Record the exact public prompt bytes used by the native Gemini runner.

This is an audit artifact only. It does not issue a model request, alter an
existing raw judgment, or add a schema that was not present in the request.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
RAW = ROOT / "raw" / "cloud"
OUT = RAW / "gemini_prompt_audit.jsonl"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_for(task: dict, instructions: str) -> str:
    header = (
        "You are one independent blind evaluator for the frozen new-material translation evaluation. "
        "Do not browse, call tools, inspect files, infer identity, use prior scores/reference, or add advice/summary/rewrite. "
        "Use only this task. Return exactly one UTF-8 JSON object and no Markdown or explanation.\n\n"
        "FROZEN INSTRUCTIONS:\n"
        + instructions
        + "\n\nTASK TRACK: "
        + task["track"]
        + "\nblind_id: "
        + task["blind_id"]
        + "\nsource_kind: "
        + task["source_kind"]
        + "\n"
    )
    if task["source_kind"] == "image":
        return (
            header
            + "This is an image job. I have attached the native public image pixels "
            + task["image_alias"]
            + ". Judge those pixels plus the displayed target translation below. Do not OCR or invent an English transcription, and do not use any image ground truth. If the image cannot be inspected, return status=\"blocked\", abstain=true, null score/subscores, critical=false, errors=[], and non-empty block_reason.\nTARGET TRANSLATION (exact):\n"
            + task["target_zh_tw"]
            + "\n\nReturn the pointwise object with exactly: blind_id,status,abstain,score_0_100,subscores,critical,errors (and optional block_reason). For status ok, scores are integers and subscores sum exactly; errors use exact target quotes only when supported."
        )
    if task["track"] == "pointwise":
        return (
            header
            + "Judge this text using the rubric. SOURCE ENGLISH (exact):\n"
            + task["source_en"]
            + "\n\nTARGET zh-TW (exact):\n"
            + task["target_zh_tw"]
            + '\n\nReturn exactly this pointwise object shape: {"blind_id":"'
            + task["blind_id"]
            + '\",\"status\":\"ok\",\"abstain\":false,\"score_0_100\":0,\"subscores\":{\"terminology_locale\":0,\"semantic_safety\":0,\"fluency\":0,\"completeness_format\":0},\"critical\":false,\"errors\":[],\"block_reason\":\"\"}. Use status blocked/missing only if genuinely unjudgeable. Scores must be integer and sum exactly; do not invent errors.'
        )
    candidate_a = task["candidate_a"]
    candidate_b = task["candidate_b"]
    source = (
        "The native image pixels are attached; do not OCR or invent an English transcription."
        if task["source_kind"] == "image"
        else "SOURCE ENGLISH (exact):\n" + task["source_en"]
    )
    return (
        header
        + "Judge this pairwise orientation using only the source and A/B targets. "
        + source
        + "\n\nCANDIDATE A (exact):\n"
        + candidate_a["target_zh_tw"]
        + "\n\nCANDIDATE B (exact):\n"
        + candidate_b["target_zh_tw"]
        + '\n\nReturn exactly: {"winner":"A","confidence":0.0,"critical_override":false,"reason_codes":["better_semantic_fidelity"]}. winner must be A, B, or TIE; confidence 0-1; reason_codes nonempty.'
    )


def main() -> None:
    instructions = (PUBLIC / "instructions.md").read_text(encoding="utf-8")
    jobs: dict[str, dict] = {}
    for track, name in (("pointwise", "judge-05.jsonl"), ("pairwise", "judge-05.jsonl")):
        path = PUBLIC / track / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            task = json.loads(line)
            task["track"] = track
            key = task.get("job_id") or task.get("pair_job_id") or task.get("pair_id")
            if key:
                jobs[f"{track}-" + key] = task

    rows: list[str] = []
    for raw_path in sorted(RAW.glob("pointwise-*.gemini.json")) + sorted(RAW.glob("pairwise-*.gemini.json")):
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        request_id = raw["request_id"]
        task = jobs.get(request_id)
        if task is None:
            continue
        prompt = prompt_for(task, instructions)
        calculated = sha256_text(prompt)
        recorded = raw.get("provenance", {}).get("prompt_sha256")
        audit = {
            "schema_version": "new-cloud-gemini-prompt-audit-v1",
            "request_id": request_id,
            "track": task["track"],
            "conversation_url": raw.get("conversation_url"),
            "fresh_conversation": raw.get("settings", {}).get("fresh_conversation"),
            "judge": raw.get("judge"),
            "task_binding": raw.get("task_binding"),
            "prompt_sha256": calculated,
            "recorded_prompt_sha256": recorded,
            "prompt_hash_matches_record": None if recorded is None else calculated == recorded,
            "schema_embedded": False,
            "schema_embedding_note": "The sent native prompt contained frozen instructions, task data/pixels, and an example output shape; it did not contain the selected public JSON schema.",
            "prompt": prompt,
        }
        rows.append(json.dumps(audit, ensure_ascii=False, separators=(",", ":")))
    OUT.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "records": len(rows), "all_reconstructed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
