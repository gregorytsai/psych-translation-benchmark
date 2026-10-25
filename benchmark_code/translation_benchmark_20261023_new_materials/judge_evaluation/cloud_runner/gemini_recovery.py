#!/usr/bin/env python3
"""Isolated, one-attempt recovery runner for the previously failed Gemini jobs.

This module is deliberately separate from the original Gemini checkpoint.  It
only reads the frozen public packet and the old checkpoint, and writes under
``recovery_20261025_failed_jobs/gemini``.  The CUA operator uses ``prepare``
to obtain a self-contained prompt, then ``capture`` and ``record`` to preserve
one native-UI response and validate it.  No model call is made by this file.

The recovery policy is intentionally narrow: exactly the five old ``failed``
and eighteen old ``missing`` jobs are eligible, each receives at most one new
attempt, and a clean first valid response is accepted without score selection
or regrading any old response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
PUBLIC = EVAL / "public"
BASELINE_RAW = EVAL / "raw" / "cloud"
OLD_CHECKPOINT = BASELINE_RAW / "gemini.checkpoint.json"
PACKET_READY = EVAL / "packet_ready.json"
FROZEN_MANIFEST = EVAL / "manifest.json"
INSTRUCTIONS = PUBLIC / "instructions.md"
POINTWISE = PUBLIC / "pointwise" / "judge-05.jsonl"
PAIRWISE = PUBLIC / "pairwise" / "judge-05.jsonl"
APPROVAL = EVAL / "ROOT_APPROVED"

RECOVERY = EVAL / "recovery_20261025_failed_jobs" / "gemini"
REQUESTS = RECOVERY / "requests"
MANIFEST = RECOVERY / "recovery_manifest.json"
CHECKPOINT = RECOVERY / "recovery_checkpoint.json"
PROGRESS = RECOVERY / "progress.md"

EXPECTED_JUDGE_ID = "gemini-3.8-flash-high-antigravity-new"
EXPECTED_MODEL = "Gemini 3.8 Flash High"
EXPECTED_ROUTE = "antigravity_native_ui"
RECOVERY_REGIME = "recovery_full_frozen_schema_v3_user_authorized_failed_job_retry"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise SystemExit(f"refusing to overwrite immutable recovery file: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise SystemExit(f"non-object public row at {path}:{line_no}")
        rows.append(item)
    return rows


def parent_raw_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return p
    # A few old missing records contain an already-prefixed relative path.
    direct = EVAL / p
    if direct.exists():
        return direct
    prefixed = EVAL / ".." / p
    if prefixed.exists():
        return prefixed.resolve()
    return direct


def task_key(track: str, task: dict[str, Any]) -> str:
    return str(task["job_id"] if track == "pointwise" else task["pair_job_id"])


def task_payload(track: str, task_id: str) -> dict[str, Any]:
    rows = load_jsonl(POINTWISE if track == "pointwise" else PAIRWISE)
    key = "job_id" if track == "pointwise" else "pair_job_id"
    found = [row for row in rows if row.get(key) == task_id]
    if len(found) != 1:
        raise SystemExit(f"expected exactly one frozen public {track} row for {task_id}, got {len(found)}")
    return found[0]


def canonical_task(track: str, task: dict[str, Any]) -> dict[str, Any]:
    # Keep the benchmark's validator as the single strict structural/sum/bounds
    # implementation.  Import only from the public cloud runner; no private map.
    sys.path.insert(0, str(HERE))
    from run_cloud_judge import _canonical_job  # type: ignore

    return _canonical_job(task, track=track)


def render_task(track: str, task: dict[str, Any]) -> str:
    source_kind = str(task.get("source_kind", "text")).lower()
    if source_kind == "image":
        source = "Source image: judge the attached PNG native pixels only. No OCR transcript, English source, or image ground truth is supplied."
    else:
        source = f"English source:\n{task['source_en']}"
    if track == "pointwise":
        return (
            "Task kind: pointwise\n"
            f"Item ID: {task['blind_source_id']}\n"
            f"Source ID: {task['blind_source_id']}\n"
            f"{source}\n\n"
            f"Anonymous candidate {task['blind_id']}:\n{task['target_zh_tw']}"
        )
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


def route_contract() -> str:
    return (
        "ROUTE CONTRACT (authoritative for this native UI request): Do not call any tools, inspect or search files, access a workspace or repository, browse the web, or request permission for any external action. This is a self-contained judging request; all required text is below and any image must be judged only from the native attachment. If the displayed task is insufficient, return the schema's blocked/missing object instead of attempting retrieval.\n"
    )


def verify_frozen_inputs() -> dict[str, Any]:
    if not APPROVAL.exists():
        raise SystemExit(f"ROOT_APPROVED marker missing: {APPROVAL}")
    approval = read_json(APPROVAL)
    manifest_sha = sha256_file(FROZEN_MANIFEST)
    if approval.get("root_approved") is not True:
        raise SystemExit("ROOT_APPROVED does not contain root_approved=true")
    if approval.get("manifest_sha256") != manifest_sha:
        raise SystemExit("ROOT_APPROVED manifest hash does not match frozen manifest")
    packet = read_json(PACKET_READY)
    if packet.get("packet_ready") is not True or packet.get("tests") != "PASS":
        raise SystemExit("packet_ready gate is not PASS")
    old = read_json(OLD_CHECKPOINT)
    required = {"packet_sha256", "manifest_sha256", "instruction_sha256", "results", "counts"}
    if not required.issubset(old):
        raise SystemExit("old Gemini checkpoint lacks required immutable provenance")
    if old["manifest_sha256"] != manifest_sha:
        raise SystemExit("old checkpoint manifest hash mismatch")
    instruction_sha = sha256_file(INSTRUCTIONS)
    if old["instruction_sha256"] != instruction_sha:
        raise SystemExit("old checkpoint instruction hash mismatch")
    return {
        "approval_sha256": sha256_file(APPROVAL),
        "packet_ready_sha256": sha256_file(PACKET_READY),
        "packet_sha256": old["packet_sha256"],
        "manifest_sha256": manifest_sha,
        "instruction_sha256": instruction_sha,
        "old_checkpoint_sha256": sha256_file(OLD_CHECKPOINT),
        "old_checkpoint": old,
    }


def init_recovery() -> None:
    frozen = verify_frozen_inputs()
    old = frozen.pop("old_checkpoint")
    if MANIFEST.exists() or CHECKPOINT.exists():
        # Idempotent restart is safe only when existing files are internally
        # consistent; never replace an existing recovery audit trail.
        if not MANIFEST.exists() or not CHECKPOINT.exists():
            raise SystemExit("one recovery file exists without its pair; refusing to initialize")
        existing = read_json(MANIFEST)
        if existing.get("schema_version") != "gemini-failed-job-recovery-manifest-v1":
            raise SystemExit("existing recovery manifest has an unknown schema")
        print(json.dumps({"status": "already_initialized", "manifest": str(MANIFEST), "checkpoint": str(CHECKPOINT), "manifest_sha256": sha256_file(MANIFEST), "checkpoint_sha256": sha256_file(CHECKPOINT)}))
        return

    old_results = old["results"]
    eligible = [r for r in old_results if r.get("status") in {"failed", "missing"}]
    if len(eligible) != 23:
        raise SystemExit(f"old checkpoint eligible count is {len(eligible)}, expected 23")
    request_ids = [str(r.get("request_id")) for r in eligible]
    if len(set(request_ids)) != 23:
        raise SystemExit("old eligible request IDs are not unique")
    jobs: list[dict[str, Any]] = []
    for recovery_index, parent in enumerate(eligible, 1):
        parent_id = str(parent["request_id"])
        track = "pairwise" if parent_id.startswith("pairwise-") else "pointwise"
        task_id = parent_id.split("-", 1)[1]
        task = task_payload(track, task_id)
        public_schema = PUBLIC / "schemas" / ("pointwise_result.schema.json" if track == "pointwise" else "pairwise_result.schema.json")
        old_path = parent_raw_path(parent.get("raw_path"))
        old_raw_sha = sha256_file(old_path) if old_path and old_path.exists() else None
        parent_attempt = None
        if old_path and old_path.exists():
            try:
                old_raw = read_json(old_path)
                parent_attempt = old_raw.get("attempt", old_raw.get("attempt_number"))
            except (OSError, json.JSONDecodeError):
                parent_attempt = None
        binding = {
            "job_id": task_id if track == "pointwise" else None,
            "pair_job_id": task_id if track == "pairwise" else None,
            "blind_id": task.get("blind_id") if track == "pointwise" else None,
            "blind_source_id": task.get("blind_source_id"),
            "pair_key": task.get("pair_key") if track == "pairwise" else None,
            "orientation": task.get("orientation") if track == "pairwise" else None,
            "source_kind": task.get("source_kind"),
            "image_alias": task.get("image_alias"),
        }
        jobs.append({
            "recovery_index": recovery_index,
            "request_id": f"recovery-{parent_id}-r1",
            "parent_request_id": parent_id,
            "track": track,
            "task_id": task_id,
            "parent_status": parent.get("status"),
            "parent_validation": parent.get("validation"),
            "parent_block_reason": parent.get("block_reason"),
            "parent_attempt": parent_attempt,
            "parent_raw_path": parent.get("raw_path"),
            "parent_raw_resolved_path": str(old_path) if old_path else None,
            "parent_raw_sha256": old_raw_sha,
            "retry_number": 1,
            "user_authorized_failed_job_retry": True,
            "prompt_regime": RECOVERY_REGIME,
            "source_kind": task.get("source_kind"),
            "image_alias": task.get("image_alias"),
            "task_sha256": sha256_json(task),
            "instruction_sha256": frozen["instruction_sha256"],
            "public_schema_path": str(public_schema),
            "public_schema_sha256": sha256_file(public_schema),
            "task_binding": binding,
            "public_task": task,
            "status": "pending",
            "attempts": [],
        })
    counts = {"planned": 23, "pending": 23, "ok": 0, "failed": 0, "missing": 0}
    manifest = {
        "schema_version": "gemini-failed-job-recovery-manifest-v1",
        "recovery_id": "recovery_20261025_failed_jobs/gemini",
        "created_at": now(),
        "status": "PREPARED",
        "authorization": {
            "label": "user_authorized_failed_job_retry",
            "scope": "exactly_old_failed_or_missing_only",
            "retry_number": 1,
            "first_valid_only": True,
            "no_valid_regrade": True,
            "no_score_selection": True,
        },
        "route": {
            "judge_slot": "judge-05",
            "judge_id": EXPECTED_JUDGE_ID,
            "provider": "google",
            "route": EXPECTED_ROUTE,
            "requested_model": EXPECTED_MODEL,
            "visible_model_required": EXPECTED_MODEL,
            "fresh_conversation_required": True,
            "native_image_only": True,
            "web_tools_workspace_forbidden": True,
        },
        "frozen_inputs": frozen,
        "parent_checkpoint": {
            "path": str(OLD_CHECKPOINT),
            "sha256": frozen["old_checkpoint_sha256"],
            "status": old.get("status"),
            "counts": old.get("counts"),
            "last_index": old.get("last_index"),
            "last_request_id": old.get("last_request_id"),
        },
        "counts": counts,
        "jobs": jobs,
        "immutable_artifacts": {
            "old_checkpoint_must_remain_unchanged": True,
            "old_raw_must_remain_unchanged": True,
            "recovery_responses_are_append_only": True,
        },
    }
    checkpoint = {
        "schema_version": "gemini-failed-job-recovery-checkpoint-v1",
        "recovery_id": manifest["recovery_id"],
        "status": "CHECKPOINT",
        "manifest_sha256": sha256_json(manifest),
        "manifest_path": str(MANIFEST),
        "parent_checkpoint_sha256": frozen["old_checkpoint_sha256"],
        "packet_sha256": frozen["packet_sha256"],
        "frozen_manifest_sha256": frozen["manifest_sha256"],
        "instruction_sha256": frozen["instruction_sha256"],
        "judge": manifest["route"],
        "authorization": manifest["authorization"],
        "planned": 23,
        "counts": counts,
        "last_recovery_index": 0,
        "last_request_id": None,
        "results": [],
        "updated_at": now(),
    }
    RECOVERY.mkdir(parents=True, exist_ok=False)
    REQUESTS.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, manifest, replace=False)
    # Checkpoint binds to the exact manifest bytes, not the pre-write object.
    checkpoint["manifest_sha256"] = sha256_file(MANIFEST)
    write_json(CHECKPOINT, checkpoint, replace=False)
    progress = (
        f"# Gemini failed-job recovery\n\n"
        f"Initialized {now()} with exactly 23 eligible jobs (5 failed + 18 missing).\n\n"
        f"- manifest: `{MANIFEST}` sha256 `{sha256_file(MANIFEST)}`\n"
        f"- checkpoint: `{CHECKPOINT}` sha256 `{sha256_file(CHECKPOINT)}`\n"
        f"- parent checkpoint sha256: `{frozen['old_checkpoint_sha256']}` (must remain unchanged)\n"
        f"- route: `{EXPECTED_ROUTE}` / `{EXPECTED_MODEL}` / fresh native UI\n"
        f"- policy: user_authorized_failed_job_retry, one retry per unique job, first-valid only\n"
    )
    PROGRESS.write_text(progress, encoding="utf-8")
    print(json.dumps({"status": "initialized", "manifest": str(MANIFEST), "checkpoint": str(CHECKPOINT), "planned": 23, "manifest_sha256": sha256_file(MANIFEST), "checkpoint_sha256": sha256_file(CHECKPOINT)}))


def load_manifest_checkpoint() -> tuple[dict[str, Any], dict[str, Any]]:
    if not MANIFEST.exists() or not CHECKPOINT.exists():
        raise SystemExit("recovery is not initialized; run init first")
    manifest = read_json(MANIFEST)
    checkpoint = read_json(CHECKPOINT)
    msha = sha256_file(MANIFEST)
    if checkpoint.get("manifest_sha256") != msha:
        raise SystemExit("recovery checkpoint manifest hash mismatch")
    if manifest.get("authorization", {}).get("label") != "user_authorized_failed_job_retry":
        raise SystemExit("recovery authorization label mismatch")
    if len(manifest.get("jobs", [])) != 23:
        raise SystemExit("recovery manifest must contain exactly 23 jobs")
    return manifest, checkpoint


def find_job(manifest: dict[str, Any], *, recovery_index: int | None, request_id: str | None) -> dict[str, Any]:
    jobs = manifest["jobs"]
    found = [j for j in jobs if (recovery_index is None or j["recovery_index"] == recovery_index) and (request_id is None or j["request_id"] == request_id)]
    if len(found) != 1:
        raise SystemExit(f"expected one recovery job, got {len(found)}")
    return found[0]


def prepare(recovery_index: int) -> None:
    manifest, checkpoint = load_manifest_checkpoint()
    job = find_job(manifest, recovery_index=recovery_index, request_id=None)
    if job["status"] != "pending":
        raise SystemExit(f"job {job['request_id']} is already {job['status']}")
    track = job["track"]
    task = job["public_task"]
    schema_path = PUBLIC / "schemas" / ("pointwise_result.schema.json" if track == "pointwise" else "pairwise_result.schema.json")
    schema_text = schema_path.read_text(encoding="utf-8")
    instructions = INSTRUCTIONS.read_text(encoding="utf-8")
    prompt = (
        instructions.rstrip()
        + "\n\n"
        + route_contract()
        + "\nSELECTED FROZEN PUBLIC JSON SCHEMA (authoritative; reproduce this contract exactly):\n"
        + schema_text.rstrip()
        + "\n\nCANONICAL OUTPUT DETAIL: For pointwise status \"ok\", include every listed result property, including block_reason as the empty string \"\". For pointwise status \"blocked\" or \"missing\", use the schema's non-empty block_reason rule. For pairwise, include every required property exactly once.\n"
        + "\nTASK PAYLOAD (judge only this one task; do not retrieve any other information):\n"
        + render_task(track, task)
        + "\n\nReturn exactly one UTF-8 JSON object conforming to the selected frozen public schema, with no Markdown, explanation, or extra keys."
    )
    # Safety invariant: image prompts never include source_en or any English
    # transcript; the only image content is supplied by the native attachment.
    if str(task.get("source_kind", "text")).lower() == "image" and "English source:" in prompt:
        raise SystemExit("image recovery prompt accidentally contains English source")
    request_id = job["request_id"]
    prompt_path = REQUESTS / f"{request_id}.prompt.txt"
    meta_path = REQUESTS / f"{request_id}.request.json"
    if prompt_path.exists() or meta_path.exists():
        raise SystemExit(f"recovery request already prepared: {request_id}")
    prompt_bytes = prompt.encode("utf-8")
    metadata = {
        "schema_version": "gemini-failed-job-recovery-request-v1",
        "request_id": request_id,
        "recovery_id": manifest["recovery_id"],
        "recovery_index": recovery_index,
        "parent_request_id": job["parent_request_id"],
        "parent_status": job["parent_status"],
        "parent_attempt": job.get("parent_attempt"),
        "retry_number": 1,
        "user_authorized_failed_job_retry": True,
        "prompt_regime": RECOVERY_REGIME,
        "track": track,
        "task_id": job["task_id"],
        "judge_slot": "judge-05",
        "judge_id": EXPECTED_JUDGE_ID,
        "provider": "google",
        "route": EXPECTED_ROUTE,
        "requested_model": EXPECTED_MODEL,
        "visible_model_required": EXPECTED_MODEL,
        "fresh_conversation_required": True,
        "schema_embedded": True,
        "web_tools_workspace_forbidden": True,
        "source_kind": task.get("source_kind"),
        "image_alias": task.get("image_alias"),
        "public_schema_path": str(schema_path),
        "public_schema_sha256": sha256_file(schema_path),
        "instruction_sha256": sha256_file(INSTRUCTIONS),
        "task_sha256": sha256_json(task),
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "prompt_path": str(prompt_path),
        "conversation_url": None,
        "visible_model_label": None,
        "fresh_conversation_confirmed": False,
        "route_status": None,
        "route_evidence": None,
        "sent_at": None,
        "captured_at": None,
        "capture_path": None,
        "full_ax_sha256": None,
        "response_sha256": None,
        "parsed": None,
        "status": "prepared_not_sent",
    }
    prompt_path.write_bytes(prompt_bytes)
    write_json(meta_path, metadata, replace=False)
    print(json.dumps({"status": "prepared", "request_id": request_id, "recovery_index": recovery_index, "prompt_path": str(prompt_path), "meta_path": str(meta_path), "prompt_sha256": metadata["prompt_sha256"], "track": track, "source_kind": task.get("source_kind"), "image_alias": task.get("image_alias"), "schema_embedded": True}, ensure_ascii=False))


def extract_response(ax_text: str, track: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if track == "pairwise":
        lookahead = r'(?=.*"winner")(?=.*"confidence")'
        markers = ("winner", "confidence")
    else:
        lookahead = r'(?=.*"blind_id")(?=.*"score_0_100")'
        markers = ("blind_id", "score_0_100")
    matches = re.findall(rf"\n\s*\d+\s+(?:文字|text)\s+(\{{{lookahead}.*?\}})\s*$", ax_text, flags=re.MULTILINE)
    nested = False
    if not matches:
        parts: list[str] = []
        active = False
        nested_matches: list[str] = []
        for line in ax_text.splitlines():
            node = re.match(r"^\s*\d+\s+(?:文字|text)\s+(.*)$", line)
            if not active:
                if node and node.group(1).strip() == "{":
                    active = True
                    parts = ["{"]
                continue
            if not node:
                continue
            fragment = node.group(1)
            parts.append(fragment)
            if fragment.strip() == "}":
                candidate = "".join(parts)
                if all(marker in candidate for marker in markers):
                    nested_matches.append(candidate)
                active = False
                parts = []
        if nested_matches:
            matches = nested_matches
            nested = True
    if not matches:
        return None, {"parse_status": "no_complete_json", "nested_ax": nested, "candidate_count": 0}
    response_raw = matches[-1]
    try:
        parsed = json.loads(response_raw)
        normalized = False
    except json.JSONDecodeError:
        try:
            parsed = json.loads(re.sub(r"\s{2,}", " ", response_raw))
            normalized = True
        except json.JSONDecodeError as exc:
            return None, {"parse_status": "json_decode_error", "parse_error": str(exc), "response_raw": response_raw, "nested_ax": nested, "candidate_count": len(matches)}
    if not isinstance(parsed, dict):
        return None, {"parse_status": "json_not_object", "response_raw": response_raw, "nested_ax": nested, "candidate_count": len(matches)}
    return parsed, {"parse_status": "parsed", "response_raw": response_raw, "normalized_ax_spacing": normalized, "nested_ax": nested, "candidate_count": len(matches)}


def capture(request_id: str, conversation_url: str, visible_model_label: str, ax_path_str: str, route_status: str, route_evidence: str, fresh_confirmed: bool) -> None:
    manifest, checkpoint = load_manifest_checkpoint()
    job = find_job(manifest, recovery_index=None, request_id=request_id)
    meta_path = REQUESTS / f"{request_id}.request.json"
    if not meta_path.exists():
        raise SystemExit(f"prepared metadata missing: {meta_path}")
    metadata = read_json(meta_path)
    if metadata.get("status") != "prepared_not_sent":
        raise SystemExit(f"request is not waiting for capture: {metadata.get('status')}")
    ax_path = Path(ax_path_str)
    if not ax_path.exists():
        raise SystemExit(f"AX capture missing: {ax_path}")
    if route_status not in {"clean", "contaminated", "unknown"}:
        raise SystemExit("route_status must be clean, contaminated, or unknown")
    if visible_model_label != EXPECTED_MODEL:
        route_status = "contaminated"
        route_evidence = (route_evidence + "; " if route_evidence else "") + f"visible model mismatch: {visible_model_label!r}"
    ax_bytes = ax_path.read_bytes()
    ax_text = ax_bytes.decode("utf-8", errors="replace")
    parsed, parse_info = extract_response(ax_text, job["track"])
    full_ax_path = REQUESTS / f"{request_id}.ax.full.txt"
    envelope_path = REQUESTS / f"{request_id}.response.envelope.json"
    if full_ax_path.exists() or envelope_path.exists():
        raise SystemExit(f"immutable capture already exists for {request_id}")
    shutil.copyfile(ax_path, full_ax_path)
    envelope = {
        "schema_version": "gemini-failed-job-recovery-response-envelope-v1",
        "request_id": request_id,
        "recovery_index": job["recovery_index"],
        "parent_request_id": job["parent_request_id"],
        "track": job["track"],
        "task_id": job["task_id"],
        "user_authorized_failed_job_retry": True,
        "retry_number": 1,
        "prompt_regime": RECOVERY_REGIME,
        "conversation_url": conversation_url,
        "visible_model_label": visible_model_label,
        "requested_model": EXPECTED_MODEL,
        "route": EXPECTED_ROUTE,
        "sent_at": metadata.get("sent_at"),
        "route_status": route_status,
        "route_evidence": route_evidence,
        "fresh_conversation_confirmed": bool(fresh_confirmed),
        "captured_at": now(),
        "ax_source_path": str(ax_path),
        "full_ax_path": str(full_ax_path),
        "full_ax_sha256": sha256_bytes(ax_bytes),
        "parse": parse_info,
        "parsed": parsed,
    }
    # response_sha256 is over the exact extracted JSON bytes, not normalized
    # content; the full native AX remains the primary capture evidence.
    if isinstance(parse_info.get("response_raw"), str):
        envelope["response_sha256"] = sha256_bytes(parse_info["response_raw"].encode("utf-8"))
    else:
        envelope["response_sha256"] = None
    write_json(envelope_path, envelope, replace=False)
    metadata.update({
        "conversation_url": conversation_url,
        "visible_model_label": visible_model_label,
        "fresh_conversation_confirmed": bool(fresh_confirmed),
        "route_status": route_status,
        "route_evidence": route_evidence,
        "captured_at": envelope["captured_at"],
        "capture_path": str(full_ax_path),
        "full_ax_sha256": envelope["full_ax_sha256"],
        "response_envelope_path": str(envelope_path),
        "response_envelope_sha256": sha256_file(envelope_path),
        "response_sha256": envelope["response_sha256"],
        "parsed": parsed,
        "parse": parse_info,
        "status": "captured_native_ui" if parsed is not None else "captured_native_ui_invalid",
    })
    write_json(meta_path, metadata)
    print(json.dumps({"status": metadata["status"], "request_id": request_id, "route_status": route_status, "visible_model_label": visible_model_label, "full_ax_sha256": envelope["full_ax_sha256"], "response_sha256": envelope["response_sha256"], "parse_status": parse_info.get("parse_status"), "envelope": str(envelope_path)}, ensure_ascii=False))


def record(request_id: str) -> None:
    manifest, checkpoint = load_manifest_checkpoint()
    job = find_job(manifest, recovery_index=None, request_id=request_id)
    if job["status"] != "pending":
        raise SystemExit(f"job {request_id} already recorded as {job['status']}")
    meta_path = REQUESTS / f"{request_id}.request.json"
    if not meta_path.exists():
        raise SystemExit(f"request metadata missing: {meta_path}")
    metadata = read_json(meta_path)
    if metadata.get("status") not in {"captured_native_ui", "captured_native_ui_invalid"}:
        raise SystemExit(f"request must be captured before record: {metadata.get('status')}")
    envelope_path = Path(str(metadata.get("response_envelope_path", "")))
    if not envelope_path.is_absolute():
        envelope_path = RECOVERY / envelope_path
    envelope = read_json(envelope_path)
    parsed = envelope.get("parsed")
    route_status = envelope.get("route_status")
    structural_errors: list[str] = []
    evidence: list[str] = []
    block_reason = None
    status = "ok"
    if envelope.get("visible_model_label") != EXPECTED_MODEL:
        status = "missing"
        block_reason = f"route_model_mismatch: visible {envelope.get('visible_model_label')!r}, required {EXPECTED_MODEL!r}"
    elif not envelope.get("fresh_conversation_confirmed"):
        status = "missing"
        block_reason = "fresh_conversation_not_confirmed"
    elif route_status != "clean":
        status = "missing"
        block_reason = f"route_{route_status or 'unknown'}: {envelope.get('route_evidence') or 'route cleanliness not proven'}"
    elif parsed is None:
        status = "missing"
        block_reason = f"no_unambiguous_json: {envelope.get('parse', {}).get('parse_status', 'unknown')}"
    else:
        canonical = canonical_task(job["track"], job["public_task"])
        sys.path.insert(0, str(HERE))
        from run_cloud_judge import validate_result_diagnostics  # type: ignore
        diagnostics = validate_result_diagnostics(job["track"], parsed, canonical)
        structural_errors = diagnostics["structural_errors"]
        evidence = diagnostics["evidence_diagnostics"]
        if structural_errors:
            status = "failed"
            block_reason = "strict_schema_or_score_validation_failed"
    result = {
        "recovery_index": job["recovery_index"],
        "request_id": request_id,
        "parent_request_id": job["parent_request_id"],
        "track": job["track"],
        "task_id": job["task_id"],
        "status": status,
        "parsed": parsed,
        "score_0_100": parsed.get("score_0_100") if status == "ok" and isinstance(parsed, dict) and job["track"] == "pointwise" else None,
        "winner": parsed.get("winner") if status == "ok" and isinstance(parsed, dict) and job["track"] == "pairwise" else None,
        "validation": {"status": "ok" if not structural_errors and status == "ok" else "failed", "structural_errors": structural_errors, "evidence_diagnostics": evidence},
        "block_reason": block_reason,
        "parent_status": job["parent_status"],
        "parent_attempt": job.get("parent_attempt"),
        "retry_number": 1,
        "user_authorized_failed_job_retry": True,
        "prompt_regime": RECOVERY_REGIME,
        "route": envelope.get("route"),
        "route_status": route_status,
        "requested_model": EXPECTED_MODEL,
        "visible_model_label": envelope.get("visible_model_label"),
        "conversation_url": envelope.get("conversation_url"),
        "fresh_conversation_confirmed": envelope.get("fresh_conversation_confirmed"),
        "prompt_sha256": metadata.get("prompt_sha256"),
        "instruction_sha256": metadata.get("instruction_sha256"),
        "public_schema_sha256": metadata.get("public_schema_sha256"),
        "task_sha256": metadata.get("task_sha256"),
        "response_envelope_path": str(envelope_path),
        "response_envelope_sha256": sha256_file(envelope_path),
        "full_ax_path": envelope.get("full_ax_path"),
        "full_ax_sha256": envelope.get("full_ax_sha256"),
        "response_sha256": envelope.get("response_sha256"),
        "recorded_at": now(),
    }
    # Append to checkpoint atomically, then update manifest job status.  A
    # duplicate request ID can never be inserted even after a process crash.
    if any(r.get("request_id") == request_id for r in checkpoint.get("results", [])):
        raise SystemExit(f"checkpoint already contains {request_id}; refusing duplicate")
    checkpoint["results"].append(result)
    checkpoint["counts"]["pending"] -= 1
    checkpoint["counts"][status] += 1
    checkpoint["last_recovery_index"] = job["recovery_index"]
    checkpoint["last_request_id"] = request_id
    checkpoint["updated_at"] = now()
    write_json(CHECKPOINT, checkpoint)
    job["status"] = status
    job["attempts"].append({
        "retry_number": 1,
        "request_id": request_id,
        "status": status,
        "result_recorded_at": result["recorded_at"],
        "response_envelope_sha256": result["response_envelope_sha256"],
    })
    manifest["counts"]["pending"] -= 1
    manifest["counts"][status] += 1
    manifest["updated_at"] = now()
    write_json(MANIFEST, manifest)
    # The checkpoint must bind to the current immutable manifest after its
    # status update; update only the binding field, never prior result data.
    checkpoint = read_json(CHECKPOINT)
    checkpoint["manifest_sha256"] = sha256_file(MANIFEST)
    checkpoint["updated_at"] = now()
    write_json(CHECKPOINT, checkpoint)
    print(json.dumps({"status": status, "request_id": request_id, "recovery_index": job["recovery_index"], "score_0_100": result["score_0_100"], "winner": result["winner"], "structural_errors": structural_errors, "evidence_diagnostics": evidence, "block_reason": block_reason, "checkpoint_counts": checkpoint["counts"]}, ensure_ascii=False))


def show() -> None:
    manifest, checkpoint = load_manifest_checkpoint()
    print(json.dumps({"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST), "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT), "counts": checkpoint.get("counts"), "jobs": [{"recovery_index": j["recovery_index"], "request_id": j["request_id"], "parent_request_id": j["parent_request_id"], "status": j["status"]} for j in manifest["jobs"]]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    p = sub.add_parser("prepare")
    p.add_argument("--recovery-index", type=int, required=True)
    c = sub.add_parser("capture")
    c.add_argument("--request-id", required=True)
    c.add_argument("--conversation-url", required=True)
    c.add_argument("--visible-model-label", required=True)
    c.add_argument("--ax-path", required=True)
    c.add_argument("--route-status", choices=["clean", "contaminated", "unknown"], required=True)
    c.add_argument("--route-evidence", default="")
    c.add_argument("--fresh-confirmed", action="store_true")
    r = sub.add_parser("record")
    r.add_argument("--request-id", required=True)
    sub.add_parser("show")
    args = parser.parse_args()
    if args.command == "init":
        init_recovery()
    elif args.command == "prepare":
        prepare(args.recovery_index)
    elif args.command == "capture":
        capture(args.request_id, args.conversation_url, args.visible_model_label, args.ax_path, args.route_status, args.route_evidence, args.fresh_confirmed)
    elif args.command == "record":
        record(args.request_id)
    else:
        show()


if __name__ == "__main__":
    main()
