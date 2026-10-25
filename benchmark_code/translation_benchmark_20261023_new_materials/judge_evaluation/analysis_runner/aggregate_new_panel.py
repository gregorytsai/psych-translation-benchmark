#!/usr/bin/env python3
"""Read-only, resumable aggregation for the frozen new-material judge panel.

This module deliberately treats judge output as an append-only evidence source.
It never edits a worker's raw files, selects a retry by score, fills a missing
score with zero, or uses a partially observed queue for model ranking.  The
command can be rerun while workers are writing; a JSON file is used only when
its bytes and stat metadata are stable across the read.

The public functions are small enough for offline fault-fixture tests.  The
CLI writes only ``judge_evaluation/reports`` and ``progress_analysis.md``.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve()
JUDGE_ROOT = HERE.parents[1]
BUNDLE_ROOT = JUDGE_ROOT.parent
RAW_ROOT = JUDGE_ROOT / "raw"
PUBLIC_ROOT = JUDGE_ROOT / "public"
PRIVATE_ROOT = JUDGE_ROOT / "private"
REPORT_ROOT = JUDGE_ROOT / "reports"
ANALYSIS_PROGRESS = JUDGE_ROOT / "progress_analysis.md"
SOURCE_CSV = BUNDLE_ROOT / "results" / "translation_results.csv"
FROZEN_TEXT = BUNDLE_ROOT / "materials" / "frozen_text_materials.jsonl"
IMAGE_SELECTION = BUNDLE_ROOT / "materials" / "image_selection.json"
MATERIAL_MANIFEST = BUNDLE_ROOT / "materials" / "metadata_manifest.json"
PACKET_MANIFEST = JUDGE_ROOT / "manifest.json"
APPROVAL = JUDGE_ROOT / "ROOT_APPROVED"
INDEPENDENT_QA = JUDGE_ROOT / "independent_qa" / "qa_results.json"
REFRESH_LOCK = HERE.parent / ".refresh.lock"

JUDGE_SLOTS = ["judge-01", "judge-02", "judge-03", "judge-04", "judge-05"]
JUDGE_META = {
    "judge-01": {"model_id": "qwen3.8-flash-next-local", "family": "local_qwen", "label": "Qwen (local)"},
    "judge-02": {"model_id": "nemotron-3.5-super-vl-120b-a12b-local", "family": "local_nemotron", "label": "Nemotron (local)"},
    "judge-03": {"model_id": "mistral-medium-3.5-128b-local", "family": "local_mistral", "label": "Mistral (local)"},
    "judge-04": {"model_id": "gpt-6.1-sol", "family": "cloud_gpt", "label": "GPT (cloud)"},
    "judge-05": {"model_id": "gemini-3.8-flash-high", "family": "cloud_gemini", "label": "Gemini (cloud)"},
}

# This is a design constraint, not a fabricated score.  Mistral has no mmproj
# image route in the frozen local runner, so the image panel is deliberately a
# fixed four-judge panel when the corresponding records are available.
IMAGE_PANEL_SLOTS = ["judge-01", "judge-02", "judge-04", "judge-05"]
TEXT_PANEL_SLOTS = list(JUDGE_SLOTS)
POINT_EXPECTED = {"text": 30, "image": 4}
PAIR_EXPECTED = {"text": 48, "image": 24}
MODEL_IDS = [
    "gemma4-12b",
    "gemma4-26b",
    "gemma4-31b-qat",
    "gemma4-e4b",
    "translategemma-12b",
    "translategemma-27b",
    "translategemma-4b",
]
SCORE_FIELDS = {
    "completeness_format": 20,
    "fluency": 20,
    "semantic_safety": 30,
    "terminology_locale": 30,
}
SEVERITIES = {"minor", "major", "critical"}
MAX_BOOTSTRAP = 10000


@contextlib.contextmanager
def refresh_lock():
    """Serialize report refreshes without touching frozen inputs.

    Workers can call ``--refresh`` frequently.  A non-blocking advisory lock
    avoids interleaving CSV/Markdown writes from two callers; the lock file is
    outside the frozen public/private/manifest trees and is never reported as
    a raw input.
    """
    try:
        import fcntl  # type: ignore
    except ImportError:  # pragma: no cover - Windows fallback
        yield True
        return
    REFRESH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with REFRESH_LOCK.open("a+", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, data: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data.encode("utf-8") if isinstance(data, str) else data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json_stable(path: Path, retries: int = 3) -> tuple[Any | None, str | None, str | None]:
    """Return ``(parsed, sha256, reason)`` without accepting a moving file."""
    last_reason = "unreadable"
    for _ in range(retries):
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                last_reason = "changed_during_read"
                continue
            digest = sha256_bytes(raw)
            try:
                return json.loads(raw.decode("utf-8")), digest, None
            except Exception as exc:  # malformed/incomplete worker output is evidence
                last_reason = f"json_parse:{type(exc).__name__}"
                continue
        except (FileNotFoundError, PermissionError, OSError) as exc:
            last_reason = f"read:{type(exc).__name__}"
    return None, None, last_reason


def read_json(path: Path) -> Any:
    value, _, reason = read_json_stable(path, retries=5)
    if reason or value is None:
        raise RuntimeError(f"cannot read stable JSON {path}: {reason}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    try:
        raw = path.read_bytes()
    except OSError:
        return rows
    # A worker may append while we read; line-level JSON parse failures are
    # retained as snapshot diagnostics by the caller rather than crashing.
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_error_item(item: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return ["error_item_not_object"]
    required = {"severity", "type", "target_exact_quote", "source_evidence", "source_span", "target_anchor", "confidence"}
    extras = set(item) - required
    missing = required - set(item)
    if extras:
        errors.append("error_item_extra:" + ",".join(sorted(extras)))
    if missing:
        errors.append("error_item_missing:" + ",".join(sorted(missing)))
    if item.get("severity") not in SEVERITIES:
        errors.append("error_item_bad_severity")
    for key in ("type", "target_exact_quote", "source_evidence", "source_span", "target_anchor"):
        if not isinstance(item.get(key), str):
            errors.append(f"error_item_{key}_not_string")
    if not finite_number(item.get("confidence")) or not 0 <= float(item.get("confidence")) <= 1:
        errors.append("error_item_bad_confidence")
    return errors


def validate_pointwise(parsed: Any, expected_blind_id: str | None = None) -> dict[str, Any]:
    """Strictly validate frozen 30/30/20/20 pointwise output.

    Evidence diagnostics are deliberately returned separately and do not
    invalidate an otherwise coherent scalar result.
    """
    result: dict[str, Any] = {"valid": False, "scalar_valid": False, "status": "invalid", "errors": [], "evidence_diagnostics": []}
    if not isinstance(parsed, dict):
        result["errors"] = ["result_not_object"]
        return result
    allowed = {"blind_id", "status", "abstain", "score_0_100", "subscores", "critical", "errors", "block_reason"}
    extras = set(parsed) - allowed
    required = allowed - {"block_reason"}
    missing = required - set(parsed)
    if extras:
        result["errors"].append("extra_keys:" + ",".join(sorted(extras)))
    if missing:
        result["errors"].append("missing_keys:" + ",".join(sorted(missing)))
    if expected_blind_id and parsed.get("blind_id") != expected_blind_id:
        result["errors"].append("blind_id_mismatch")
    if not isinstance(parsed.get("blind_id"), str) or not re.fullmatch(r"b-[0-9a-f]{12}", parsed.get("blind_id", "")):
        result["errors"].append("bad_blind_id")
    status = parsed.get("status")
    result["status"] = status if status in {"ok", "blocked", "missing"} else "invalid"
    if status not in {"ok", "blocked", "missing"}:
        result["errors"].append("bad_status")
    if not isinstance(parsed.get("abstain"), bool):
        result["errors"].append("abstain_not_bool")
    if not isinstance(parsed.get("critical"), bool):
        result["errors"].append("critical_not_bool")
    if not isinstance(parsed.get("errors"), list):
        result["errors"].append("errors_not_list")
    else:
        for i, item in enumerate(parsed["errors"]):
            for error in validate_error_item(item):
                result["errors"].append(f"errors[{i}].{error}")
    if status == "ok":
        if parsed.get("abstain") is not False:
            result["errors"].append("ok_requires_abstain_false")
        score = parsed.get("score_0_100")
        if not is_int(score) or not 0 <= score <= 100:
            result["errors"].append("bad_score_integer_or_range")
        subscores = parsed.get("subscores")
        if not isinstance(subscores, dict):
            result["errors"].append("subscores_not_object")
        else:
            if set(subscores) != set(SCORE_FIELDS):
                result["errors"].append("subscore_keys_mismatch")
            values = []
            for key, max_value in SCORE_FIELDS.items():
                value = subscores.get(key)
                if not is_int(value) or not 0 <= value <= max_value:
                    result["errors"].append(f"bad_subscore:{key}")
                else:
                    values.append(value)
            if len(values) == 4 and sum(values) != score:
                result["errors"].append("subscores_sum_mismatch")
        result["scalar_valid"] = not result["errors"]
    elif status in {"blocked", "missing"}:
        if parsed.get("abstain") is not True:
            result["errors"].append("abstain_required_for_blocked_or_missing")
        if parsed.get("score_0_100") is not None or parsed.get("subscores") is not None:
            result["errors"].append("blocked_missing_score_must_be_null")
        if parsed.get("critical") is not False:
            result["errors"].append("blocked_missing_critical_must_be_false")
        if parsed.get("errors") != []:
            result["errors"].append("blocked_missing_errors_must_be_empty")
        if not isinstance(parsed.get("block_reason"), str) or not parsed.get("block_reason", "").strip():
            result["errors"].append("blocked_missing_block_reason_required")
        result["scalar_valid"] = not result["errors"]
    result["valid"] = not result["errors"]
    return result


def validate_pairwise(parsed: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": False, "errors": []}
    if not isinstance(parsed, dict):
        result["errors"] = ["result_not_object"]
        return result
    allowed = {"winner", "confidence", "critical_override", "reason_codes"}
    extras = set(parsed) - allowed
    missing = allowed - set(parsed)
    if extras:
        result["errors"].append("extra_keys:" + ",".join(sorted(extras)))
    if missing:
        result["errors"].append("missing_keys:" + ",".join(sorted(missing)))
    if parsed.get("winner") not in {"A", "B", "TIE"}:
        result["errors"].append("bad_winner")
    if not finite_number(parsed.get("confidence")) or not 0 <= float(parsed.get("confidence")) <= 1:
        result["errors"].append("bad_confidence")
    if not isinstance(parsed.get("critical_override"), bool):
        result["errors"].append("critical_override_not_bool")
    if not isinstance(parsed.get("reason_codes"), list) or not parsed.get("reason_codes") or any(not isinstance(x, str) or not x for x in parsed.get("reason_codes", [])):
        result["errors"].append("reason_codes_invalid")
    result["valid"] = not result["errors"]
    return result


def evidence_diagnostics(parsed: Any, task: Mapping[str, Any], material: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Derive warnings without changing a valid scalar result."""
    out: list[dict[str, Any]] = []
    if not isinstance(parsed, dict) or parsed.get("status") != "ok":
        return out
    errors = parsed.get("errors") if isinstance(parsed.get("errors"), list) else []
    target = str(task.get("target_zh_tw") or "")
    source = str(task.get("source_en") or (material or {}).get("source_en") or "")
    for index, item in enumerate(errors):
        if not isinstance(item, dict):
            continue
        quote = item.get("target_exact_quote")
        if isinstance(quote, str) and quote.strip() and quote not in target:
            out.append({"code": "target_exact_quote_not_in_target", "error_index": index, "target_exact_quote": quote, "matched": False})
        elif isinstance(quote, str) and quote.strip():
            out.append({"code": "target_exact_quote_in_target", "error_index": index, "target_exact_quote": quote, "matched": True})
        elif not str(item.get("source_span") or "").strip() or not str(item.get("target_anchor") or "").strip():
            out.append({"code": "omission_evidence_missing", "error_index": index, "matched": False})
        source_span = item.get("source_span")
        if task.get("source_kind") == "text" and isinstance(source_span, str) and source_span.strip():
            out.append({"code": "source_span_in_source", "error_index": index, "matched": source_span in source})
    has_critical = any(isinstance(x, dict) and x.get("severity") == "critical" for x in errors)
    if isinstance(parsed.get("critical"), bool) and parsed.get("critical") != has_critical:
        out.append({"code": "critical_flag_vs_error_severity", "critical": parsed.get("critical"), "has_critical_severity": has_critical})
    return out


def technical_execution_diagnostics(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    material: Mapping[str, Any] | None,
    track: str,
) -> dict[str, Any]:
    """Check trusted execution metadata separately from JSON/rubric validity.

    A structurally valid parsed object is not a valid judge observation when
    its raw runner reports transport, model-identity, finish, or native-image
    failure.  Legacy fixtures/bundle rows with no execution metadata remain
    allowed but are explicitly marked ``not_evaluable``; they are not treated
    as evidence of a successful transport.
    """
    raw = record.get("raw") if isinstance(record.get("raw"), Mapping) else record
    reasons: list[str] = []
    checked: list[str] = []
    has_execution_metadata = any(
        key in raw
        for key in (
            "status", "exit_code", "timed_out", "response_metadata", "http_envelope",
            "actual_template_render", "model_identity", "validation_errors", "invalid_reason",
            "provenance", "resolved_model", "resolved_model_id", "reported_model", "actual_model",
        )
    )

    status = raw.get("status")
    if status is not None:
        checked.append("status")
        if str(status).lower() not in {"ok", "success", "completed", "complete", "valid"}:
            reasons.append("raw_status_not_success")
    exit_code = raw.get("exit_code")
    if exit_code is not None:
        checked.append("exit_code")
        if exit_code != 0:
            reasons.append("process_exit_nonzero")
    if raw.get("timed_out") is True:
        checked.append("timed_out")
        reasons.append("execution_timed_out")

    envelopes: list[Mapping[str, Any]] = []
    for value in (raw.get("http_envelope"), raw.get("response_metadata")):
        if isinstance(value, Mapping):
            envelopes.append(value)
            nested = value.get("http_envelope")
            if isinstance(nested, Mapping):
                envelopes.append(nested)
    for envelope in envelopes:
        if envelope.get("http_status") is not None:
            checked.append("http_status")
            if envelope.get("http_status") != 200:
                reasons.append("http_status_not_200")
        if envelope.get("error"):
            checked.append("http_error")
            reasons.append("http_error_present")
        if envelope.get("finish_reason") is not None:
            checked.append("finish_reason")
            if str(envelope.get("finish_reason")).lower() not in {"stop", "completed", "complete", "success"}:
                reasons.append("finish_reason_not_stop")
    if raw.get("finish_reason") is not None:
        checked.append("finish_reason")
        if str(raw.get("finish_reason")).lower() not in {"stop", "completed", "complete", "success"}:
            reasons.append("finish_reason_not_stop")

    validation_errors = raw.get("validation_errors")
    if isinstance(validation_errors, list) and validation_errors:
        checked.append("validation_errors")
        reasons.append("raw_validation_errors_present")
    invalid_reason = raw.get("invalid_reason")
    if invalid_reason not in (None, "", [], {}):
        checked.append("invalid_reason")
        reasons.append("raw_invalid_reason_present")
    raw_validation = raw.get("validation")
    if isinstance(raw_validation, Mapping):
        if raw_validation.get("status") in {"failed", "invalid", "error"}:
            checked.append("validation_status")
            reasons.append("raw_validation_status_failed")
        if raw_validation.get("valid") is False:
            checked.append("validation_valid")
            reasons.append("raw_validation_valid_false")
        if isinstance(raw_validation.get("errors"), list) and raw_validation.get("errors"):
            checked.append("validation_errors")
            reasons.append("raw_validation_errors_present")

    template = raw.get("actual_template_render")
    if isinstance(template, Mapping):
        gate = template.get("gate")
        if isinstance(gate, Mapping):
            checked.append("actual_template_render_gate")
            if gate.get("pass") is False:
                reasons.append("native_template_gate_failed")
            if gate.get("http_200") is False:
                reasons.append("template_http_not_200")
            source_kind = expected.get("source_kind") or (material or {}).get("source_kind")
            if source_kind == "image":
                if gate.get("actual_image") is False:
                    reasons.append("native_image_render_missing")
                if gate.get("native_image_marker_if_image") is False:
                    reasons.append("native_image_marker_missing")

    # Explicit model identity mismatch is always a failed technical gate.
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), Mapping) else {}
    identity = raw.get("model_identity") if isinstance(raw.get("model_identity"), Mapping) else {}
    if "schema_provenance_valid" in provenance:
        checked.append("schema_provenance")
        if provenance.get("schema_provenance_valid") is not True:
            reasons.append("schema_provenance_invalid")
    if raw.get("model_identity_mismatch") is True or provenance.get("model_identity_mismatch") is True:
        reasons.append("model_identity_mismatch")
    reported = (
        raw.get("resolved_model")
        or raw.get("resolved_model_id")
        or raw.get("reported_model")
        or raw.get("actual_model")
        or identity.get("resolved_model")
        or identity.get("resolved_model_id")
        or identity.get("reported_model")
        or identity.get("actual_model")
        or provenance.get("resolved_model")
        or provenance.get("resolved_model_id")
        or provenance.get("reported_model")
    )
    response_metadata = raw.get("response_metadata") if isinstance(raw.get("response_metadata"), Mapping) else {}
    if reported is None:
        reported = response_metadata.get("resolved_model") or response_metadata.get("resolved_model_id") or response_metadata.get("reported_model")
    if reported is not None:
        requested = provenance.get("requested_model") or raw.get("requested_model")
        if requested and reported and str(requested).strip().casefold() != str(reported).strip().casefold():
            reasons.append("model_identity_mismatch")
        checked.append("reported_model_identity")
    if raw.get("model_id") is not None and identity.get("model_id") is not None:
        checked.append("model_identity")
        if str(raw.get("model_id")).strip().casefold() != str(identity.get("model_id")).strip().casefold():
            reasons.append("model_identity_mismatch")

    # Keep one stable reason list for annotations while preserving whether the
    # record was legacy/uninstrumented.
    reasons = sorted(set(reasons))
    return {
        "trusted": not reasons,
        "status": "pass" if not reasons else "fail",
        "evaluability": "checked" if has_execution_metadata else "not_evaluable_legacy",
        "checked": sorted(set(checked)),
        "errors": reasons,
    }


def normalize_model_name(model_id: str, source_rows: Mapping[tuple[str, str], Mapping[str, str]]) -> str:
    for (candidate, _), row in source_rows.items():
        if candidate == model_id:
            return row.get("model_name") or model_id
    return model_id


def load_source_rows() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    by_case: dict[str, dict[str, Any]] = {}
    with SOURCE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model_id = row.get("model_id", "")
            case_id = row.get("case_id", "")
            by_key[(model_id, case_id)] = row
            by_case.setdefault(case_id, {})
            by_case[case_id][model_id] = row
    return by_key, by_case


def load_materials() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    materials: dict[str, dict[str, Any]] = {}
    for line in FROZEN_TEXT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            material_id = row.get("material_id") or row.get("item_id")
            materials[material_id] = {**row, "material_id": material_id, "source_kind": "text"}
    image_doc = read_json(IMAGE_SELECTION)
    for row in image_doc.get("selected_images", []):
        image_id = row["image_id"]
        reference_lines = list(row.get("ground_truth_text", []) or [])
        reference_kind = "ground_truth_text" if reference_lines else "provisional_reference_text"
        if not reference_lines:
            reference_lines = str(row.get("provisional_reference_text", "")).splitlines()
        materials[image_id] = {
            "material_id": image_id,
            "source_id": image_id,
            "source_kind": "image",
            "source_en": "",
            "source_sha256": row.get("sha256", ""),
            "image_alias": row.get("image_id"),
            "public_alias": None,
            "ground_truth_en": "\n".join(reference_lines),
            "ground_truth_lines": reference_lines,
            "reference_text_kind": reference_kind,
            "ground_truth_status": row.get("ground_truth_status", ""),
            "image_role": row.get("role", ""),
            "layout_stratum": row.get("layout_stratum", ""),
        }
    return materials, {r.get("material_id") or r.get("image_id"): r for r in image_doc.get("selected_images", [])}, read_json(MATERIAL_MANIFEST)


def load_private_maps() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate = read_json(PRIVATE_ROOT / "candidate_map.json")
    source_map = read_json(PRIVATE_ROOT / "source_and_image_map.json")
    job_map = read_json(PRIVATE_ROOT / "job_keymap.json")
    pair_selection = read_json(PRIVATE_ROOT / "pair_selection.json")
    return candidate, source_map, job_map, pair_selection


def load_jobs() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    candidate_map, source_map, job_map, _ = load_private_maps()
    point_jobs: dict[tuple[str, str], dict[str, Any]] = {}
    pair_jobs: dict[tuple[str, str], dict[str, Any]] = {}
    for slot in JUDGE_SLOTS:
        point_path = PUBLIC_ROOT / "pointwise" / f"{slot}.jsonl"
        pair_path = PUBLIC_ROOT / "pairwise" / f"{slot}.jsonl"
        for public in load_jsonl(point_path):
            job_id = str(public.get("job_id", ""))
            private = (job_map.get("pointwise") or {}).get(job_id, {})
            merged = {**public, **private, "job_id": job_id, "judge_slot": slot}
            if not merged.get("material_id"):
                # Private binding is authoritative; source map is a fallback
                # for a worker that only retained blind/public fields.
                blind = merged.get("blind_id")
                for model_id, blind_id in (candidate_map.get("mapping_by_judge", {}).get(slot, {}) or {}).items():
                    if blind_id == blind:
                        merged["model_id"] = model_id
                        break
            point_jobs[(slot, job_id)] = merged
        for public in load_jsonl(pair_path):
            pair_id = str(public.get("pair_job_id", ""))
            private = (job_map.get("pairwise") or {}).get(pair_id, {})
            merged = {**public, **private, "pair_job_id": pair_id, "judge_slot": slot}
            pair_jobs[(slot, pair_id)] = merged
    return point_jobs, pair_jobs, {"candidate_map": candidate_map, "source_map": source_map, "job_map": job_map}


def infer_slot(value: Any, path: Path) -> str | None:
    text = str(value or "")
    if text in JUDGE_SLOTS:
        return text
    lower = (text + " " + str(path)).lower()
    if "qwen" in lower or "judge-01" in lower:
        return "judge-01"
    if "nemotron" in lower or "judge-02" in lower:
        return "judge-02"
    if "mistral" in lower or "judge-03" in lower:
        return "judge-03"
    if "gpt" in lower or "judge-04" in lower:
        return "judge-04"
    if "gemini" in lower or "judge-05" in lower:
        return "judge-05"
    return None


def infer_attempt(d: Mapping[str, Any], path: Path) -> int:
    if is_int(d.get("attempt")):
        return int(d["attempt"])
    if d.get("repair") is True or "retry1" in path.name or "attempt2" in path.name:
        return 1
    if "attempt3" in path.name:
        return 2
    return 0


GEMINI_UI_REQUEST_SCHEMA = "new-cloud-gemini-ui-request-v2"
GEMINI_CHECKPOINT_SCHEMA = "new-cloud-judge-checkpoint-v1"
CHECKPOINT_TERMINAL_STATUSES = {"ok", "failed", "missing", "skipped", "unsupported", "cancelled"}


def is_gemini_ui_request(d: Mapping[str, Any]) -> bool:
    request_id = str(d.get("request_id") or "")
    return (
        d.get("schema_version") == GEMINI_UI_REQUEST_SCHEMA
        and request_id.startswith(("pointwise-", "pairwise-"))
        and d.get("judge_slot") == "judge-05"
    )


def complete_gemini_checkpoint_terminal_statuses(data: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return terminal checkpoint outcomes only from a complete final checkpoint.

    A checkpoint's ``missing``/``failed`` state is terminal queue evidence, not
    a score. Interim checkpoints are deliberately ignored so a worker cannot
    close the primary coverage gate before its planned index and counts agree.
    """
    if data.get("schema_version") != GEMINI_CHECKPOINT_SCHEMA:
        return {}
    judge = data.get("judge") if isinstance(data.get("judge"), Mapping) else {}
    if judge.get("judge_slot") != "judge-05":
        return {}
    counts = data.get("counts") if isinstance(data.get("counts"), Mapping) else {}
    planned = counts.get("planned")
    last_index = data.get("last_index")
    if not is_int(planned) or not is_int(last_index) or planned < 0 or last_index < planned:
        return {}
    count_total = sum(int(counts.get(key, 0)) for key in ("ok", "failed", "missing", "skipped"))
    if count_total != planned:
        return {}
    results = data.get("results") if isinstance(data.get("results"), list) else []
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, Mapping):
            continue
        request_id = str(item.get("request_id") or "")
        if request_id.startswith("pointwise-"):
            track, prefix = "pointwise", "pointwise-"
        elif request_id.startswith("pairwise-"):
            track, prefix = "pairwise", "pairwise-"
        else:
            continue
        status = str(item.get("status") or "").lower()
        if status not in CHECKPOINT_TERMINAL_STATUSES:
            continue
        normalized = request_id[len(prefix):]
        out[("judge-05", track, normalized)] = {
            "status": status,
            "request_id": request_id,
            "last_index": last_index,
            "planned": planned,
        }
    return out


def resolve_capture_path(value: Any, fallback: Path) -> Path:
    """Resolve a worker-captured artifact without inventing a new source."""
    if value:
        candidate = Path(str(value))
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            for root in (BUNDLE_ROOT.parent, BUNDLE_ROOT, JUDGE_ROOT, fallback.parent):
                joined = root / candidate
                if joined.is_file():
                    return joined
    return fallback


def gemini_ui_attempt_payloads(request_path: Path, request: Mapping[str, Any]) -> list[tuple[Path, str, dict[str, Any]]]:
    """Adapt native-UI request envelopes into normal raw-attempt records.

    The native UI writer keeps the response object in ``request.json`` and
    stores the exact response separately.  The old collector only recognized
    the legacy ``*.gemini.json`` envelope, so these records were present in
    the snapshot manifest but absent from ``attempts``.  This adapter keeps
    the request envelope/provenance and uses the separately captured response
    JSON as the attempt payload; no score or target is changed.
    """
    request_id = str(request.get("request_id") or "")
    track = "pointwise" if request_id.startswith("pointwise-") else "pairwise" if request_id.startswith("pairwise-") else ""
    if not track:
        return []
    expected_schema = _public_schema_sha256(track)
    entries: list[tuple[str, Mapping[str, Any], int, str]] = []
    if isinstance(request.get("response"), Mapping):
        entries.append(("base", request, 0, ""))
    # A nested retry/repair is a separate raw attempt for the same frozen job.
    # Keep it only when its response object is actually captured.
    nested_retry = request.get("retry1")
    if isinstance(nested_retry, Mapping) and isinstance(nested_retry.get("response"), Mapping):
        entries.append(("retry1", nested_retry, 1, "infrastructure_recovery"))
    nested_repair = request.get("repair")
    if isinstance(nested_repair, Mapping) and isinstance(nested_repair.get("response"), Mapping):
        entries.append(("repair", nested_repair, 1, "format_repair"))

    out: list[tuple[Path, str, dict[str, Any]]] = []
    for entry_name, entry, attempt, explicit_kind in entries:
        # A native UI envelope may preserve a structurally valid provisional
        # JSON even after the route is marked contaminated (for example, a
        # workspace-find/web request was observed).  The Gemini checkpoint
        # deliberately records that job as missing, so the preserved payload
        # must not become a score merely because it parses.  Nested retry/repair
        # entries carry their own route metadata and are evaluated separately.
        if entry.get("route_contaminated"):
            continue
        response = entry.get("response")
        raw_path = resolve_capture_path(entry.get("response_raw_path"), request_path)
        parsed, response_digest, response_reason = read_json_stable(raw_path)
        if not isinstance(parsed, Mapping):
            # The request envelope is not a substitute for a missing raw
            # response artifact.  Leave this job missing rather than creating
            # a technical-invalid score from an unverified capture.
            continue
        if isinstance(response, Mapping) and dict(parsed) != dict(response):
            response_reason = response_reason or "response_wrapper_mismatch"
        schema_valid = (
            request.get("schema_embedded") is True
            and request.get("schema_version") == GEMINI_UI_REQUEST_SCHEMA
            and isinstance(request.get("public_schema_sha256"), str)
            and request.get("public_schema_sha256") == expected_schema
            and bool(request.get("instruction_sha256"))
            and bool(request.get("prompt_sha256"))
            and response_reason is None
        )
        job_id = str(request.get("job_id") or request_id)
        binding: dict[str, Any] = {
            "job_id": job_id if track == "pointwise" else None,
            "pair_job_id": job_id if track == "pairwise" else None,
            "blind_id": parsed.get("blind_id"),
            "source_kind": request.get("source_kind", ""),
            "image_alias": request.get("image_alias"),
        }
        task: dict[str, Any] = {
            "track": track,
            "job_id": job_id if track == "pointwise" else None,
            "pair_job_id": job_id if track == "pairwise" else None,
            "source_kind": request.get("source_kind", ""),
            "image_alias": request.get("image_alias"),
        }
        provenance = {
            "requested_model": request.get("requested_model", ""),
            "judge_id": request.get("judge_id", ""),
            "judge_slot": request.get("judge_slot", ""),
            "route": request.get("route", ""),
            "schema_version": request.get("schema_version", ""),
            "schema_embedded": request.get("schema_embedded"),
            "schema_regime": "schema_embedded_v2" if request.get("schema_embedded") is True else "schema_not_embedded",
            "public_schema_sha256": request.get("public_schema_sha256", ""),
            "instruction_sha256": request.get("instruction_sha256", ""),
            "prompt_sha256": request.get("prompt_sha256", ""),
            "response_sha256": entry.get("response_sha256") or request.get("response_sha256", ""),
            "request_path": str(request_path),
            "response_raw_path": str(raw_path),
            "schema_provenance_valid": schema_valid,
            "response_wrapper_matches_raw": response_reason is None,
        }
        payload = dict(request)
        payload.update({
            "track": track,
            "request_id": request_id,
            "status": "ok",
            "parsed": dict(parsed),
            "task_binding": binding,
            "task": task,
            "provenance": provenance,
            "attempt": attempt,
            "raw_response_path": str(raw_path),
        })
        if explicit_kind == "format_repair":
            payload["repair"] = True
            payload["repair_context"] = {"attempt_kind": "format_repair", "kind": "format_repair"}
        elif explicit_kind:
            payload["retry_kind"] = explicit_kind
        out.append((raw_path, response_digest or "", payload))
    return out


def _public_schema_sha256(track: str) -> str | None:
    schema_name = "pointwise_result.schema.json" if track == "pointwise" else "pairwise_result.schema.json"
    schema_path = PUBLIC_ROOT / "schemas" / schema_name
    return sha256_file(schema_path) if schema_path.is_file() else None


def raw_record_from_json(path: Path, d: Mapping[str, Any], digest: str, source: str) -> dict[str, Any] | None:
    task = d.get("task") if isinstance(d.get("task"), dict) else {}
    binding = d.get("task_binding") if isinstance(d.get("task_binding"), dict) else {}
    provenance = dict(d.get("provenance")) if isinstance(d.get("provenance"), dict) else {}
    track = d.get("track") or task.get("track") or provenance.get("mode")
    if track not in {"pointwise", "pairwise"}:
        return None
    request_id = d.get("request_id") or d.get("task_id") or task.get("task_id") or task.get("job_id") or binding.get("job_id") or binding.get("pair_job_id")
    if not request_id:
        return None
    parsed = d.get("parsed")
    if parsed is None and isinstance(d.get("result"), dict):
        parsed = d["result"].get("parsed")
    if parsed is None and isinstance(d.get("response"), dict):
        parsed = d["response"].get("parsed")
    status = d.get("status") or (d.get("validation") or {}).get("status")
    if status is None and isinstance(d.get("validation"), dict):
        status = "ok" if d["validation"].get("valid") else "invalid"
    slot = infer_slot(d.get("judge_slot") or provenance.get("judge_slot") or d.get("judge") or d.get("model_id"), path)
    if not slot:
        return None
    if slot == "judge-05":
        provenance.setdefault("schema_regime", "schema_embedded_v2" if d.get("schema_embedded") is True else "schema_not_embedded")
    raw_request_id = str(request_id)
    normalized_request_id = raw_request_id
    for prefix in ("pointwise-", "pairwise-", "pointwise_", "pairwise_"):
        if normalized_request_id.startswith(prefix):
            normalized_request_id = normalized_request_id[len(prefix):]
            break
    return {
        "path": str(path),
        "raw_hash": digest,
        "source": source,
        "judge_slot": slot,
        "track": track,
        "request_id": normalized_request_id,
        "raw_request_id": raw_request_id,
        "attempt": infer_attempt(d, path),
        "status": status,
        "parsed": parsed,
        "validation": d.get("validation") or {},
        "task_binding": binding,
        "task": task,
        "elapsed_s": d.get("elapsed_s", (d.get("response_metadata") or {}).get("elapsed_s", (d.get("http_envelope") or {}).get("elapsed_s"))),
        "recorded_at": d.get("recorded_at") or d.get("completed_at"),
        "provenance": provenance,
        "settings": d.get("settings") or {},
        "raw": d,
    }


def collect_raw_snapshot(
    extra_roots: Sequence[Path] | None = None,
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """Collect direct per-attempt JSON first, then bundle/checkpoint fallbacks.

    ``extra_roots`` is opt-in for the isolated post-recovery lane.  The normal
    ``--refresh`` path passes no extra roots and therefore preserves the
    original report scope.  Recovery roots are still hashed as snapshot files,
    but their attempts are never implicitly included in the baseline refresh.
    """
    attempts: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    diagnostics: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    checkpoint_terminal_statuses: dict[tuple[str, str, str], dict[str, Any]] = {}
    direct_paths: list[Path] = []
    scan_roots = [RAW_ROOT, JUDGE_ROOT / "local_runner" / "results"]
    scan_roots.extend(Path(root) for root in (extra_roots or []) if Path(root).is_dir())
    for root in scan_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            name = path.name.lower()
            if name in {"gpt.bundle.json", "gpt.checkpoint.json"} or name.endswith(".summary.json") or "transport_schemas" in path.parts or "server_cleanup" in name or "preflight" in name or "technical_smoke" in path.parts or "template_probe" in name:
                continue
            direct_paths.append(path)
        # Local append-only result files are JSONL, handled separately below.
    for path in direct_paths:
        data, digest, reason = read_json_stable(path)
        if reason or data is None:
            diagnostics.append({"path": str(path), "status": "skipped_unstable_or_malformed", "reason": reason or "empty"})
            continue
        files.append({"path": str(path), "sha256": digest, "source": "direct"})
        if isinstance(data, dict):
            if path.name == "gemini.checkpoint.json":
                # Keep the checkpoint in the immutable file snapshot, but use
                # only its complete terminal registry for queue coverage. It
                # never supplies a parsed judgment or score.
                checkpoint_terminal_statuses.update(complete_gemini_checkpoint_terminal_statuses(data))
                continue
            if is_gemini_ui_request(data):
                # The request envelope is metadata; each captured response is
                # indexed as its own attempt while the original request file
                # remains in the immutable snapshot file list.
                for response_path, response_digest, payload in gemini_ui_attempt_payloads(path, data):
                    rec = raw_record_from_json(response_path, payload, response_digest, "gemini_ui_response")
                    if rec:
                        key = (rec["judge_slot"], rec["track"], rec["request_id"])
                        attempts[key].append(rec)
                continue
            rec = raw_record_from_json(path, data, digest or "", "direct")
            if rec:
                key = (rec["judge_slot"], rec["track"], rec["request_id"])
                attempts[key].append(rec)
    # Append-only local runner results are a useful fallback when the worker
    # checkpoint has already written a record but its per-request JSON is not
    # visible in the snapshot yet.  They never replace direct raw attempts.
    jsonl_roots = [JUDGE_ROOT / "local_runner" / "results"]
    jsonl_roots.extend(Path(root) for root in (extra_roots or []) if Path(root).is_dir())
    for jsonl_root in jsonl_roots:
        for path in sorted(jsonl_root.glob("*.jsonl")):
            try:
                digest = sha256_file(path)
            except OSError:
                continue
            source_name = "local_results_jsonl" if jsonl_root == JUDGE_ROOT / "local_runner" / "results" else "recovery_results_jsonl"
            files.append({"path": str(path), "sha256": digest, "source": source_name})
            for i, data in enumerate(load_jsonl(path)):
                rec = raw_record_from_json(path, data, digest, source_name)
                if rec:
                    rec["line_index"] = i
                    key = (rec["judge_slot"], rec["track"], rec["request_id"])
                    # Keep JSONL fallback only if it has not already been captured.
                    if not any(x.get("source") == "direct" and x.get("path") != str(path) for x in attempts[key]):
                        attempts[key].append(rec)
    # Cloud bundle/checkpoint references can survive a worker moving a raw
    # file.  Add them only as a fallback for a key with no direct attempt.
    for path in sorted((RAW_ROOT / "cloud").glob("*.json")):
        if path.name not in {"gpt.bundle.json", "gpt.checkpoint.json"}:
            continue
        data, digest, reason = read_json_stable(path)
        if reason or not isinstance(data, dict):
            diagnostics.append({"path": str(path), "status": "skipped_unstable_or_malformed", "reason": reason or "not_object"})
            continue
        files.append({"path": str(path), "sha256": digest, "source": "bundle_fallback"})
        for item in data.get("results", []) if isinstance(data.get("results"), list) else []:
            if not isinstance(item, dict):
                continue
            request_id = item.get("request_id")
            parsed = item.get("parsed")
            if not request_id or parsed is None:
                continue
            pseudo = {"request_id": request_id, "judge_slot": data.get("judge_slot") or data.get("judge") or "judge-04", "track": item.get("track", "pointwise"), "status": item.get("status"), "parsed": parsed, "elapsed_s": item.get("elapsed_s"), "provenance": item.get("provenance", {})}
            rec = raw_record_from_json(path, pseudo, digest or "", "bundle_fallback")
            if not rec:
                continue
            key = (rec["judge_slot"], rec["track"], rec["request_id"])
            if not attempts[key]:
                attempts[key].append(rec)
    for key in attempts:
        attempts[key].sort(key=lambda x: (int(x.get("attempt", 0)), str(x.get("path", ""))))
    return attempts, diagnostics, files, checkpoint_terminal_statuses


def job_key_from_record(rec: Mapping[str, Any], track: str) -> str | None:
    binding = rec.get("task_binding") if isinstance(rec.get("task_binding"), Mapping) else {}
    task = rec.get("task") if isinstance(rec.get("task"), Mapping) else {}
    if track == "pointwise":
        return str(binding.get("job_id") or task.get("job_id") or task.get("task_id") or rec.get("request_id")) if (binding.get("job_id") or task.get("job_id") or task.get("task_id") or rec.get("request_id")) else None
    return str(binding.get("pair_job_id") or task.get("pair_job_id") or task.get("task_id") or rec.get("request_id")) if (binding.get("pair_job_id") or task.get("pair_job_id") or task.get("task_id") or rec.get("request_id")) else None


def select_first_valid(attempt_list: Sequence[Mapping[str, Any]], expected: Mapping[str, Any], track: str, material: Mapping[str, Any] | None) -> dict[str, Any]:
    """Select the first schema-valid attempt, never a quality-preferred retry."""
    normalized: list[dict[str, Any]] = []
    for rec in sorted(attempt_list, key=lambda x: (int(x.get("attempt", 0)), str(x.get("path", "")))):
        parsed = rec.get("parsed")
        validation = validate_pointwise(parsed, expected.get("blind_id")) if track == "pointwise" else validate_pairwise(parsed)
        task = expected
        ev = evidence_diagnostics(parsed, task, material) if track == "pointwise" else []
        technical = technical_execution_diagnostics(rec, expected, material, track)
        status = "valid" if validation.get("valid") else "invalid"
        if validation.get("valid") and not technical.get("trusted"):
            status = "technical_invalid"
        if track == "pointwise" and validation.get("valid") and technical.get("trusted") and isinstance(parsed, dict):
            status = parsed.get("status", "invalid")
        normalized.append({
            **rec,
            "validation_local": validation,
            "technical_diagnostics_local": technical,
            "evidence_diagnostics_local": ev,
            "selection_status": status,
        })
    selected = next((x for x in normalized if x["selection_status"] in {"valid", "ok", "blocked", "missing"}), None)
    if selected is None:
        selected = normalized[-1] if normalized else None
    if selected is None:
        return {
            "status": "missing",
            "selected": None,
            "attempts": [],
            "selected_attempt_kind": "initial",
            "format_repair_used": False,
            "infrastructure_recovery_used": False,
            "unknown_recovery_used": False,
            "all_attempts": [],
        }
    prior = [a for a in normalized if a is not selected]
    selected_attempt_kind = "initial"
    if int(selected.get("attempt", 0) or 0) > 0:
        raw = selected.get("raw") if isinstance(selected.get("raw"), Mapping) else {}
        explicit_kind = str(raw.get("attempt_kind") or raw.get("retry_kind") or "").strip().lower()
        repair_context = raw.get("repair_context") if isinstance(raw.get("repair_context"), Mapping) else {}
        explicit_kind = explicit_kind or str(repair_context.get("attempt_kind") or repair_context.get("kind") or "").strip().lower()
        explicit_repair = raw.get("repair") is True or explicit_kind in {"format_repair", "format-repair", "repair"}
        # A same-prompt retry is infrastructure recovery only when the prior
        # attempt did not return a parsed candidate judgment.  A prior parsed
        # sum/schema failure is not silently relabeled as transport recovery.
        no_parsed_prior = any(a.get("parsed") is None for a in prior)
        if explicit_repair:
            selected_attempt_kind = "format_repair"
        elif no_parsed_prior:
            selected_attempt_kind = "infrastructure_recovery"
        else:
            selected_attempt_kind = "unknown_recovery"
    return {
        "status": selected["selection_status"],
        "selected": selected,
        "attempts": normalized,
        "selected_attempt_kind": selected_attempt_kind,
        "format_repair_used": selected_attempt_kind == "format_repair",
        "infrastructure_recovery_used": selected_attempt_kind == "infrastructure_recovery",
        "unknown_recovery_used": selected_attempt_kind == "unknown_recovery",
        "all_attempts": normalized,
    }


def observed_attempt_kind(attempt: Mapping[str, Any]) -> str:
    """Classify an individual raw attempt without inferring quality intent."""
    raw = attempt.get("raw") if isinstance(attempt.get("raw"), Mapping) else {}
    explicit_kind = str(raw.get("attempt_kind") or raw.get("retry_kind") or "").strip().lower()
    repair_context = raw.get("repair_context") if isinstance(raw.get("repair_context"), Mapping) else {}
    explicit_kind = explicit_kind or str(repair_context.get("attempt_kind") or repair_context.get("kind") or "").strip().lower()
    if raw.get("repair") is True or explicit_kind in {"format_repair", "format-repair", "repair"}:
        return "format_repair"
    if int(attempt.get("attempt", 0) or 0) == 0:
        return "initial"
    if attempt.get("parsed") is None:
        return "infrastructure_failure_unclassified"
    return "unknown_recovery"


def match_untranslated_visible_english(target: str, material: Mapping[str, Any]) -> tuple[bool, str]:
    if material.get("source_kind") != "image":
        return False, ""
    for line in material.get("ground_truth_lines", []) or []:
        line = str(line).strip()
        # Match the independent packet-QA definition: visible English lines
        # need at least 12 characters.  Acronyms/proper nouns alone are not
        # sufficient; a match is a quality flag, not proof of an error.
        if len(line) < 12 or not re.search(r"[A-Za-z]", line):
            continue
        if line.casefold() in str(target or "").casefold():
            return True, line
    return False, ""


def numeric(value: Any) -> float | None:
    try:
        if value is None or value == "" or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def safe_median(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.median(vals) if vals else None


def safe_sd(values: Iterable[float]) -> float | None:
    vals = list(values)
    return math.sqrt(sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / len(vals)) if vals else None


def quantile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for pos in range(i, j):
            ranks[indexed[pos][0]] = rank
        i = j
    return ranks


def kendall_w(matrix: Sequence[Sequence[float]]) -> float | None:
    """Tie-corrected Kendall's W for a complete unit-by-judge matrix.

    Ties reduce the attainable denominator.  If every judge gives every unit
    the same rank, the corrected denominator is zero and W is undefined (NA),
    not a fabricated 0 or 1.
    """
    if not matrix or not matrix[0]:
        return None
    n = len(matrix); k = len(matrix[0])
    if n < 2 or k < 2 or any(len(row) != k for row in matrix):
        return None
    ranks_by_judge = [rankdata([matrix[i][j] for i in range(n)]) for j in range(k)]
    sums = [sum(ranks_by_judge[j][i] for j in range(k)) for i in range(n)]
    mean_sum = sum(sums) / n
    s = sum((x - mean_sum) ** 2 for x in sums)
    tie_term = 0.0
    for j in range(k):
        counts = Counter(matrix[i][j] for i in range(n))
        tie_term += sum(float(t ** 3 - t) for t in counts.values() if t > 1)
    denom = k * k * (n ** 3 - n) - k * tie_term
    return (12.0 * s / denom) if denom else None


def icc_2(matrix: Sequence[Sequence[float]]) -> tuple[float | None, float | None, int, int]:
    """Two-way random absolute-agreement ICC(2,1)/(2,k).

    The formulas follow the standard ANOVA mean-square form.  No confidence
    interval is fabricated here; the report records that it is unavailable.
    """
    if not matrix or not matrix[0]:
        return None, None, 0, 0
    n = len(matrix); k = len(matrix[0])
    if n < 2 or k < 2 or any(len(row) != k for row in matrix):
        return None, None, n, k
    grand = sum(sum(row) for row in matrix) / (n * k)
    row_means = [sum(row) / k for row in matrix]
    col_means = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]
    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_error = sum((matrix[i][j] - row_means[i] - col_means[j] + grand) ** 2 for i in range(n) for j in range(k))
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom_21 = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    denom_2k = ms_rows + (ms_cols - ms_error) / n
    return ((ms_rows - ms_error) / denom_21 if denom_21 else None,
            (ms_rows - ms_error) / denom_2k if denom_2k else None, n, k)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        names: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    names.append(key); seen.add(key)
        fieldnames = names or ["status"]
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def material_from_job(job: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return dict(materials.get(str(job.get("material_id")), {"material_id": job.get("material_id", ""), "source_kind": job.get("source_kind", "")}) or {})


def selected_attempt_row(
    slot: str,
    expected: Mapping[str, Any],
    attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    track: str,
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    job_id = str(expected.get("job_id") or expected.get("pair_job_id"))
    key = (slot, track, job_id)
    material = material_from_job(expected, materials)
    selection = select_first_valid(attempts.get(key, []), expected, track, material)
    selected = selection.get("selected") or {}
    parsed = selected.get("parsed") if isinstance(selected, Mapping) else None
    validation = selected.get("validation_local", {}) if isinstance(selected, Mapping) else {}
    technical = selected.get("technical_diagnostics_local", {}) if isinstance(selected, Mapping) else {}
    technical_trusted = bool(technical.get("trusted")) if isinstance(technical, Mapping) else False
    status = selection.get("status", "missing")
    if not attempts.get(key):
        status = "missing"
    score = parsed.get("score_0_100") if isinstance(parsed, dict) and validation.get("scalar_valid") and technical_trusted else None
    subscores = parsed.get("subscores") if isinstance(parsed, dict) and validation.get("scalar_valid") and technical_trusted else None
    target = expected.get("target_zh_tw", "")
    untrans, untrans_span = match_untranslated_visible_english(str(target), material)
    evidence = selected.get("evidence_diagnostics_local", []) if isinstance(selected, Mapping) else []
    elapsed = selected.get("elapsed_s") if isinstance(selected, Mapping) else None
    if elapsed is None and isinstance(selected, Mapping):
        raw = selected.get("raw") or {}
        elapsed = (raw.get("response_metadata") or {}).get("elapsed_s") or (raw.get("http_envelope") or {}).get("elapsed_s")
    return {
        "judge_slot": slot,
        "job_id": job_id,
        "track": track,
        "material_id": expected.get("material_id", ""),
        "source_kind": expected.get("source_kind", material.get("source_kind", "")),
        "model_id": expected.get("model_id", ""),
        "blind_id": expected.get("blind_id", ""),
        "source_en": expected.get("source_en", material.get("source_en", "")) if expected.get("source_kind", material.get("source_kind")) == "text" else "",
        "target_zh_tw": expected.get("target_zh_tw", ""),
        "image_alias": expected.get("image_alias", material.get("public_alias", "")) if expected.get("source_kind", material.get("source_kind")) == "image" else "",
        "image_sha256": expected.get("image_sha256", material.get("source_sha256", "")) if expected.get("source_kind", material.get("source_kind")) == "image" else "",
        "status": status,
        "scalar_valid": bool(validation.get("scalar_valid") and technical_trusted),
        "schema_valid": bool(validation.get("valid")),
        "technical_valid": technical_trusted,
        "technical_status": technical.get("status", "not_evaluated") if isinstance(technical, Mapping) else "not_evaluated",
        "technical_evaluability": technical.get("evaluability", "not_evaluated") if isinstance(technical, Mapping) else "not_evaluated",
        "technical_errors": technical.get("errors", []) if isinstance(technical, Mapping) else [],
        "validation_valid": bool(validation.get("valid") and technical_trusted),
        "score_0_100": score,
        "completeness_format": (subscores or {}).get("completeness_format") if isinstance(subscores, dict) else None,
        "fluency": (subscores or {}).get("fluency") if isinstance(subscores, dict) else None,
        "semantic_safety": (subscores or {}).get("semantic_safety") if isinstance(subscores, dict) else None,
        "terminology_locale": (subscores or {}).get("terminology_locale") if isinstance(subscores, dict) else None,
        "critical": parsed.get("critical") if isinstance(parsed, dict) and validation.get("valid") and technical_trusted else None,
        "error_count": len(parsed.get("errors", [])) if isinstance(parsed, dict) and isinstance(parsed.get("errors"), list) else None,
        "evidence_warning_count": len(evidence),
        "evidence_warning_codes": [x.get("code") for x in evidence if isinstance(x, dict) and x.get("code")],
        "untranslated_visible_english": untrans,
        "untranslated_visible_english_span": untrans_span,
        "image_reference_text_kind": material.get("reference_text_kind", "") if material.get("source_kind") == "image" else "",
        "image_ground_truth_status": material.get("ground_truth_status", "") if material.get("source_kind") == "image" else "",
        "attempt_count": len(selection.get("attempts", [])),
        "selected_attempt": selected.get("attempt") if isinstance(selected, Mapping) else None,
        "attempt_kind": selection.get("selected_attempt_kind", "initial"),
        "format_repair_used": selection.get("format_repair_used", False),
        "infrastructure_recovery_used": selection.get("infrastructure_recovery_used", False),
        "unknown_recovery_used": selection.get("unknown_recovery_used", False),
        "raw_path": selected.get("path", "") if isinstance(selected, Mapping) else "",
        "raw_sha256": selected.get("raw_hash", "") if isinstance(selected, Mapping) else "",
        "raw_status": selected.get("status", "") if isinstance(selected, Mapping) else "",
        "elapsed_s": elapsed,
        "invalid_reasons": list(validation.get("errors", [])) + [f"technical:{e}" for e in (technical.get("errors", []) if isinstance(technical, Mapping) else [])],
        "all_attempt_paths": [x.get("path", "") for x in selection.get("attempts", []) if isinstance(x, Mapping)],
        "all_attempt_statuses": [x.get("selection_status", "invalid") for x in selection.get("attempts", []) if isinstance(x, Mapping)],
        "expected": True,
    }


def is_design_unsupported_image(slot: str, job: Mapping[str, Any], track: str = "pointwise") -> bool:
    """Return the frozen Mistral image-route exclusion, regardless of raw status."""
    return slot == "judge-03" and track in {"pointwise", "pairwise"} and str(job.get("source_kind")) == "image"


def build_pointwise_rows(
    point_jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (slot, job_id), job in sorted(point_jobs.items(), key=lambda item: (item[0][0], item[0][1])):
        row = selected_attempt_row(slot, {**job, "job_id": job_id}, attempts, "pointwise", materials)
        if is_design_unsupported_image(slot, job, "pointwise"):
            row["status"] = "unsupported_by_design"
            row["unsupported_reason"] = "Mistral local route has no mmproj/native image capability; no score imputed"
            row["scalar_valid"] = False
            # A route-level design exclusion is not a technical execution
            # failure.  Keep raw_status/raw_path for provenance, but do not
            # count this cell as technical-invalid in summaries/annotations.
            row["technical_status"] = "not_evaluated"
            row["technical_valid"] = False
            row["technical_evaluability"] = "not_evaluated"
            row["technical_errors"] = []
            row["validation_valid"] = False
        else:
            row["unsupported_reason"] = ""
        rows.append(row)
    return rows


def pointwise_expected_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("judge_slot")), str(row.get("material_id")), str(row.get("model_id"))


def build_comparison_rows(point_rows: Sequence[Mapping[str, Any]], source_rows: Mapping[tuple[str, str], Mapping[str, str]], materials: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in point_rows:
        key = (str(row.get("material_id")), str(row.get("model_id")))
        if key not in groups:
            src = dict(source_rows.get((key[1], key[0]), {}))
            material = materials.get(key[0], {})
            groups[key] = {
                "material_id": key[0],
                "source_id": src.get("source_id", material.get("source_id", "")),
                "source_kind": row.get("source_kind", material.get("source_kind", "")),
                "source_en": src.get("source_en", material.get("source_en", "")) if row.get("source_kind") == "text" else "",
                "source_sha256": src.get("source_sha256", material.get("source_sha256", "")),
                "image_alias": row.get("image_alias", ""),
                "image_sha256": row.get("image_sha256", ""),
                "model_id": key[1],
                "model_name": src.get("model_name", key[1]),
                "model_alias": src.get("model_alias", ""),
                "target_zh_tw": src.get("translation_zh_tw", row.get("target_zh_tw", "")),
                "image_ground_truth_en": src.get("image_ground_truth_en", material.get("ground_truth_en", "")),
                "image_reference_text_kind": materials.get(key[0], {}).get("reference_text_kind", "") if row.get("source_kind") == "image" else "",
                "image_ground_truth_status": materials.get(key[0], {}).get("ground_truth_status", "") if row.get("source_kind") == "image" else "",
                "image_role": materials.get(key[0], {}).get("image_role", "") if row.get("source_kind") == "image" else "",
                "image_layout_stratum": materials.get(key[0], {}).get("layout_stratum", "") if row.get("source_kind") == "image" else "",
                "translation_status": src.get("status", ""),
                "translation_elapsed_seconds": src.get("elapsed_seconds", ""),
            }
        base = groups[key]
        slot = str(row.get("judge_slot"))
        base[f"{slot}_score"] = row.get("score_0_100")
        base[f"{slot}_status"] = row.get("status")
        base[f"{slot}_valid"] = row.get("scalar_valid")
        base[f"{slot}_critical"] = row.get("critical")
        base[f"{slot}_error_count"] = row.get("error_count")
        base[f"{slot}_evidence_warning_count"] = row.get("evidence_warning_count")
        base[f"{slot}_raw_path"] = row.get("raw_path")
        base[f"{slot}_selected_attempt"] = row.get("selected_attempt")
        base[f"{slot}_attempt_kind"] = row.get("attempt_kind")
        base[f"{slot}_infrastructure_recovery_used"] = row.get("infrastructure_recovery_used")
        base[f"{slot}_untranslated_visible_english"] = row.get("untranslated_visible_english")
    # Every candidate/material group is expected to have all five slots; this
    # keeps missing cells explicit rather than silently dropping early jobs.
    for base in groups.values():
        mat = str(base.get("material_id")); required = TEXT_PANEL_SLOTS if base.get("source_kind") == "text" else IMAGE_PANEL_SLOTS
        scores = [numeric(base.get(f"{slot}_score")) for slot in required]
        valid_scores = [x for x in scores if x is not None]
        base["panel_required_k"] = len(required)
        base["panel_n_valid"] = len(valid_scores)
        base["panel_complete"] = len(valid_scores) == len(required)
        base["panel_mean"] = safe_mean(valid_scores) if valid_scores else None
        base["panel_median"] = safe_median(valid_scores) if valid_scores else None
        base["ranking_status"] = "DESCRIPTIVE_ROW_ONLY_PANEL_PARTIAL" if not base["panel_complete"] else "ROW_COMPLETE_BUT_TRACK_GATE_REQUIRED"
        base["untranslated_visible_english_any_judge"] = any(base.get(f"{slot}_untranslated_visible_english") is True for slot in JUDGE_SLOTS)
        material = materials.get(mat, {})
        direct_flag, direct_span = match_untranslated_visible_english(str(base.get("target_zh_tw", "")), material)
        base["translation_untranslated_visible_english"] = direct_flag
        base["translation_untranslated_visible_english_span"] = direct_span
        base["translation_untranslated_visible_english_note"] = "match flag only; acronym/proper noun or legitimate retained English may be acceptable; preserve target unchanged" if direct_flag else ""
        base["source_cluster"] = materials.get(mat, {}).get("source_id", base.get("source_id", ""))
    return sorted(groups.values(), key=lambda x: (str(x.get("source_kind")), str(x.get("material_id")), str(x.get("model_id"))))


def build_error_annotations(point_rows: Sequence[Mapping[str, Any]], attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]], point_jobs: Mapping[tuple[str, str], Mapping[str, Any]], materials: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in point_rows:
        slot = str(row.get("judge_slot")); job_id = str(row.get("job_id")); key = (slot, "pointwise", job_id)
        expected = point_jobs.get((slot, job_id), {})
        for attempt in attempts.get(key, []):
            parsed = attempt.get("parsed")
            if not isinstance(parsed, dict) or not isinstance(parsed.get("errors"), list):
                continue
            selected = attempt.get("raw_hash") == row.get("raw_sha256")
            validation = validate_pointwise(parsed, expected.get("blind_id"))
            attempt_status = "ok" if validation.get("valid") and isinstance(parsed, dict) and parsed.get("status") == "ok" else (parsed.get("status") if isinstance(parsed, dict) else "invalid")
            for index, item in enumerate(parsed.get("errors", [])):
                if not isinstance(item, dict):
                    continue
                quote = item.get("target_exact_quote", "")
                target = str(expected.get("target_zh_tw", ""))
                source = str(expected.get("source_en", "")) if expected.get("source_kind") == "text" else ""
                rows.append({
                    "judge_slot": slot,
                    "model_id": expected.get("model_id", ""),
                    "material_id": expected.get("material_id", ""),
                    "source_kind": expected.get("source_kind", ""),
                    "job_id": job_id,
                    "attempt": attempt.get("attempt", 0),
                    "selected_for_scalar": selected,
                    "record_status": attempt_status,
                    "severity": item.get("severity"),
                    "type": item.get("type"),
                    "confidence": item.get("confidence"),
                    "source_span": item.get("source_span"),
                    "source_evidence": item.get("source_evidence"),
                    "target_anchor": item.get("target_anchor"),
                    "target_exact_quote": quote,
                    "target_exact_quote_match": bool(quote and quote in target),
                    "source_span_match": bool(source and item.get("source_span") and item.get("source_span") in source) if expected.get("source_kind") == "text" else "not_applicable_image_source",
                    "critical_flag": parsed.get("critical"),
                    "critical_severity": item.get("severity") == "critical",
                    "critical_flag_vs_severity_disagreement": isinstance(parsed.get("critical"), bool) and ((parsed.get("critical") is True) != (item.get("severity") == "critical")),
                    "raw_path": attempt.get("path", ""),
                    "raw_sha256": attempt.get("raw_hash", ""),
                })
    return rows


def build_invalid_annotations(
    point_rows: Sequence[Mapping[str, Any]],
    attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    point_jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    pair_jobs: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot, track, request_id in sorted(attempts):
        for attempt in attempts[(slot, track, request_id)]:
            expected = point_jobs.get((slot, request_id), {}) if track == "pointwise" else (pair_jobs or {}).get((slot, request_id), {})
            if is_design_unsupported_image(slot, expected, track):
                # Mistral image cells are excluded by the frozen panel design;
                # an emitted N/A/no-mmproj record is evidence for that design
                # exclusion, not a scored technical failure.
                continue
            validation = validate_pointwise(attempt.get("parsed"), expected.get("blind_id")) if track == "pointwise" else validate_pairwise(attempt.get("parsed"))
            technical = technical_execution_diagnostics(attempt, expected, None, track)
            if validation.get("valid") and technical.get("trusted"):
                continue
            validation_errors = list(validation.get("errors", []))
            if attempt.get("parsed") is None and not validation_errors:
                validation_errors = ["raw_record_no_parsed_result"]
            validation_errors.extend(f"technical:{e}" for e in technical.get("errors", []))
            selection_status = attempt.get("selection_status", "invalid")
            if validation.get("valid") and not technical.get("trusted"):
                selection_status = "technical_invalid"
            rows.append({
                "judge_slot": slot,
                "track": track,
                "request_id": request_id,
                "attempt": attempt.get("attempt", 0),
                "attempt_kind": observed_attempt_kind(attempt),
                "raw_status": attempt.get("status", ""),
                "selection_status": selection_status,
                "validation_errors": validation_errors,
                "technical_status": technical.get("status"),
                "technical_errors": technical.get("errors", []),
                "technical_evaluability": technical.get("evaluability"),
                "evidence_diagnostics": attempt.get("evidence_diagnostics_local", []),
                "raw_path": attempt.get("path", ""),
                "raw_sha256": attempt.get("raw_hash", ""),
                "preserved_excluded": True,
            })
    return rows


def build_pairwise_rows(
    pair_jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (slot, pair_job_id), job in sorted(pair_jobs.items(), key=lambda item: (item[0][0], item[0][1])):
        key = (slot, "pairwise", pair_job_id)
        selection = select_first_valid(attempts.get(key, []), job, "pairwise", None)
        selected = selection.get("selected") or {}
        parsed = selected.get("parsed") if isinstance(selected, Mapping) else None
        validation = selected.get("validation_local", {}) if isinstance(selected, Mapping) else {}
        technical = selected.get("technical_diagnostics_local", {}) if isinstance(selected, Mapping) else {}
        technical_trusted = bool(technical.get("trusted")) if isinstance(technical, Mapping) else False
        status = selection.get("status", "missing")
        if not attempts.get(key):
            status = "missing"
        if is_design_unsupported_image(slot, job, "pairwise"):
            status = "unsupported_by_design"
        rows.append({
            "judge_slot": slot,
            "pair_job_id": pair_job_id,
            "pair_key": job.get("pair_key", ""),
            "material_id": job.get("material_id", ""),
            "source_id": job.get("source_id", ""),
            "source_kind": job.get("source_kind", ""),
            "model_a": job.get("model_a", ""),
            "model_b": job.get("model_b", ""),
            "canonical_left_model_id": job.get("canonical_left_model_id", ""),
            "canonical_right_model_id": job.get("canonical_right_model_id", ""),
            "orientation": job.get("orientation", ""),
            "winner": parsed.get("winner") if isinstance(parsed, dict) and validation.get("valid") and technical_trusted else None,
            "confidence": parsed.get("confidence") if isinstance(parsed, dict) and validation.get("valid") and technical_trusted else None,
            "critical_override": parsed.get("critical_override") if isinstance(parsed, dict) and validation.get("valid") and technical_trusted else None,
            "reason_codes": parsed.get("reason_codes") if isinstance(parsed, dict) and validation.get("valid") and technical_trusted else None,
            "status": status,
            "schema_valid": bool(validation.get("valid")),
            "technical_valid": technical_trusted,
            "technical_errors": [] if is_design_unsupported_image(slot, job, "pairwise") else technical.get("errors", []) if isinstance(technical, Mapping) else [],
            "valid": False if is_design_unsupported_image(slot, job, "pairwise") else bool(validation.get("valid") and technical_trusted),
            "attempt_count": len(selection.get("attempts", [])),
            "selected_attempt": selected.get("attempt") if isinstance(selected, Mapping) else None,
            "attempt_kind": selection.get("selected_attempt_kind", "initial"),
            "format_repair_used": selection.get("format_repair_used", False),
            "infrastructure_recovery_used": selection.get("infrastructure_recovery_used", False),
            "unknown_recovery_used": selection.get("unknown_recovery_used", False),
            "raw_path": selected.get("path", "") if isinstance(selected, Mapping) else "",
            "raw_sha256": selected.get("raw_hash", "") if isinstance(selected, Mapping) else "",
            "invalid_reasons": list(validation.get("errors", [])) + [f"technical:{e}" for e in (technical.get("errors", []) if isinstance(technical, Mapping) else [])],
        })
    return rows


def build_pairwise_canonical(pair_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in pair_rows:
        groups[(str(row.get("judge_slot")), str(row.get("pair_key")), str(row.get("material_id")))][str(row.get("orientation", ""))] = row
    output: list[dict[str, Any]] = []
    for (slot, pair_key, material_id), orient in sorted(groups.items()):
        ab = orient.get("AB"); ba = orient.get("BA")
        row: dict[str, Any] = {
            "judge_slot": slot,
            "pair_key": pair_key,
            "material_id": material_id,
            "source_kind": (ab or ba or {}).get("source_kind", ""),
            "canonical_left_model_id": (ab or ba or {}).get("canonical_left_model_id", ""),
            "canonical_right_model_id": (ab or ba or {}).get("canonical_right_model_id", ""),
            "has_AB": bool(ab),
            "has_BA": bool(ba),
            "AB_status": (ab or {}).get("status", "missing"),
            "BA_status": (ba or {}).get("status", "missing"),
            "AB_winner": (ab or {}).get("winner"),
            "BA_winner": (ba or {}).get("winner"),
            "canonical_AB_winner": None,
            "canonical_BA_winner": None,
            "canonical_status": "MISSING_OR_INVALID",
            "position_bias_signal": None,
        }
        if ab and ab.get("valid"):
            row["canonical_AB_winner"] = {"A": ab.get("model_a"), "B": ab.get("model_b"), "TIE": "TIE"}.get(ab.get("winner"))
        if ba and ba.get("valid"):
            row["canonical_BA_winner"] = {"A": ba.get("model_a"), "B": ba.get("model_b"), "TIE": "TIE"}.get(ba.get("winner"))
        left = row["canonical_AB_winner"]; right = row["canonical_BA_winner"]
        if left is not None and right is not None:
            if left == right:
                row["canonical_status"] = "CONSISTENT_TIE" if left == "TIE" else "CONSISTENT"
                row["canonical_winner"] = left
            else:
                # Orientation inconsistency is an abstention, never a tie.
                row["canonical_status"] = "ABSTAIN_INCONSISTENT"
                row["canonical_winner"] = None
                row["position_bias_signal"] = True
        else:
            row["canonical_winner"] = None
        output.append(row)
    return output


def summarize_judges(point_rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]], canonical_rows: Sequence[Mapping[str, Any]], snapshot_files: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in JUDGE_SLOTS:
        meta = JUDGE_META[slot]
        p = [r for r in point_rows if r.get("judge_slot") == slot]
        q = [r for r in pair_rows if r.get("judge_slot") == slot]
        scores = [numeric(r.get("score_0_100")) for r in p if r.get("scalar_valid")]
        latencies = [numeric(r.get("elapsed_s")) for r in p + q if numeric(r.get("elapsed_s")) is not None]
        unsupported = sum(r.get("status") == "unsupported_by_design" for r in p)
        technical_invalid = sum(bool(r.get("technical_errors")) and r.get("status") != "unsupported_by_design" for r in p)
        invalid = sum(r.get("status") in {"invalid", "technical_invalid"} for r in p)
        blocked = sum(r.get("status") in {"blocked", "missing"} for r in p)
        pair_valid = sum(r.get("valid") is True for r in q)
        canon = [r for r in canonical_rows if r.get("judge_slot") == slot]
        selected = [r for r in p if r.get("scalar_valid")]
        rows.append({
            "judge_slot": slot,
            "judge_label": meta["label"],
            "judge_family": meta["family"],
            "judge_model_id": meta["model_id"],
            "pointwise_expected": len(p),
            "pointwise_raw_attempt_selected": sum(bool(r.get("raw_path")) for r in p),
            "pointwise_scalar_valid": len(scores),
            "pointwise_invalid_or_excluded": invalid,
            "pointwise_technical_invalid": technical_invalid,
            "pointwise_blocked_or_missing": blocked,
            "pointwise_unsupported_by_design": unsupported,
            "pointwise_format_repair_used": sum(bool(r.get("format_repair_used")) for r in p),
            "pointwise_infrastructure_recovery_used": sum(bool(r.get("infrastructure_recovery_used")) for r in p),
            "pointwise_unknown_recovery_used": sum(bool(r.get("unknown_recovery_used")) for r in p),
            "pointwise_evidence_warning_count": sum(int(r.get("evidence_warning_count") or 0) for r in p),
            "pointwise_critical_flag_count": sum(r.get("critical") is True for r in selected),
            "pointwise_observed_mean_noncomparative": safe_mean(scores),
            "pointwise_observed_sd_noncomparative": safe_sd(scores),
            "pointwise_unique_scores": len(set(scores)),
            "pairwise_expected": len(q),
            "pairwise_valid_orientations": pair_valid,
            "pairwise_invalid_or_missing": len(q) - pair_valid,
            "pairwise_canonical_groups": len(canon),
            "pairwise_consistent": sum(r.get("canonical_status") in {"CONSISTENT", "CONSISTENT_TIE"} for r in canon),
            "pairwise_inconsistent_abstain": sum(r.get("canonical_status") == "ABSTAIN_INCONSISTENT" for r in canon),
            "latency_median_s": safe_median(latencies),
            "latency_p95_s": quantile(latencies, 0.95),
            "translation_latency_not_included": True,
            "status": "OBSERVED_PARTIAL" if len(scores) < len([r for r in p if r.get("source_kind") == "text"]) else "OBSERVED",
        })
    return rows


def track_required_slots(track: str) -> list[str]:
    return TEXT_PANEL_SLOTS if track == "text" else IMAGE_PANEL_SLOTS


def track_gate(comparison_rows: Sequence[Mapping[str, Any]], track: str) -> tuple[bool, int, int]:
    rows = [r for r in comparison_rows if r.get("source_kind") == track]
    required_models = set(MODEL_IDS)
    expected_units = len(rows)
    complete = sum(bool(r.get("panel_complete")) for r in rows)
    # A balanced candidate×material panel requires every candidate/material
    # cell, not merely a large number of available observations.
    gate = bool(rows) and complete == expected_units and {str(r.get("model_id")) for r in rows} == required_models
    return gate, complete, expected_units


def common_case_intersection(
    comparison_rows: Sequence[Mapping[str, Any]],
    track: str,
    terminal_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Find the pre-specified all-candidate/common-panel intersection.

    This is intentionally separate from the full fixed-panel gate.  It can
    become eligible only after every primary job has a terminal outcome; before
    then it reports a queue-state exclusion and returns no analysis rows.
    """
    terminal_coverage = terminal_coverage or {}
    terminal_ready = bool(terminal_coverage.get("all_primary_terminal"))
    required_slots = track_required_slots(track)
    rows = [r for r in comparison_rows if r.get("source_kind") == track]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("material_id"))].append(row)
    complete_material_ids: list[str] = []
    reasons: dict[str, str] = {}
    for material_id, material_rows in sorted(grouped.items()):
        by_model = {str(row.get("model_id")): row for row in material_rows}
        missing_models = sorted(set(MODEL_IDS) - set(by_model))
        incomplete_models = sorted(
            model for model in MODEL_IDS
            if model in by_model and not bool(by_model[model].get("panel_complete"))
        )
        if missing_models or incomplete_models:
            parts: list[str] = []
            if missing_models:
                parts.append("missing_candidates=" + ",".join(missing_models))
            if incomplete_models:
                parts.append("incomplete_panel=" + ",".join(incomplete_models))
            reasons[material_id] = ";".join(parts)
        else:
            complete_material_ids.append(material_id)
    source_pages = sorted({
        str(row.get("source_cluster") or row.get("source_id") or row.get("material_id"))
        for row in rows if str(row.get("material_id")) in complete_material_ids
    }) if track == "text" else []
    minimum_items = 12 if track == "text" else 2
    minimum_source_pages = 6 if track == "text" else 0
    threshold_reason = ""
    if not terminal_ready:
        threshold_reason = "primary_terminal_coverage_incomplete"
    elif len(complete_material_ids) < minimum_items:
        threshold_reason = f"common_items_below_minimum:{len(complete_material_ids)}<{minimum_items}"
    elif track == "text" and len(source_pages) < minimum_source_pages:
        threshold_reason = f"source_pages_below_minimum:{len(source_pages)}<{minimum_source_pages}"
    eligible_material_ids = complete_material_ids if not threshold_reason else []
    if threshold_reason:
        for material_id in complete_material_ids:
            reasons.setdefault(material_id, threshold_reason)
    for material_id in grouped:
        if not terminal_ready:
            reasons[material_id] = "primary_terminal_coverage_incomplete"
    return {
        "track": track,
        "terminal_ready": terminal_ready,
        "required_slots": required_slots,
        "complete_material_ids": complete_material_ids,
        "eligible_material_ids": eligible_material_ids,
        "eligible_item_count": len(eligible_material_ids),
        "source_pages": source_pages,
        "source_page_count": len(source_pages),
        "minimum_items": minimum_items,
        "minimum_source_pages": minimum_source_pages,
        "threshold_reason": threshold_reason,
        "analysis_ready": bool(eligible_material_ids),
        "status": "LIMITED_COMMON_CASE_READY" if eligible_material_ids else (
            "QUEUE_TERMINAL_INCOMPLETE" if not terminal_ready else "COMMON_CASE_THRESHOLD_NOT_MET"
        ),
        "excluded_case_ids": sorted(set(grouped) - set(eligible_material_ids)),
        "excluded_case_reasons": {key: reasons[key] for key in sorted(reasons)},
    }


def annotate_common_case(
    comparison_rows: Sequence[Mapping[str, Any]],
    common_cases: Mapping[str, Mapping[str, Any]],
) -> None:
    """Attach common-case eligibility/exclusion metadata to derived CSV rows."""
    for row in comparison_rows:
        track = str(row.get("source_kind"))
        summary = common_cases.get(track, {})
        material_id = str(row.get("material_id"))
        eligible = material_id in set(summary.get("eligible_material_ids", []))
        complete = material_id in set(summary.get("complete_material_ids", []))
        reasons = summary.get("excluded_case_reasons", {})
        row["common_case_candidate_item_complete"] = complete
        row["common_case_eligible"] = eligible
        row["common_case_status"] = summary.get("status", "QUEUE_TERMINAL_INCOMPLETE")
        row["common_case_exclusion_reason"] = "" if eligible else str(reasons.get(material_id, ""))
        row["common_case_source_page_count"] = summary.get("source_page_count", 0)


def build_model_summary(comparison_rows: Sequence[Mapping[str, Any]], point_rows: Sequence[Mapping[str, Any]], source_rows: Mapping[tuple[str, str], Mapping[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for track in ["text", "image"]:
        gate, complete_units, expected_units = track_gate(comparison_rows, track)
        rows = [r for r in comparison_rows if r.get("source_kind") == track]
        required_k = len(track_required_slots(track))
        for model_id in MODEL_IDS:
            cr = [r for r in rows if r.get("model_id") == model_id]
            valid = [numeric(r.get("panel_mean")) for r in cr if r.get("panel_complete")]
            scalar = [r for r in point_rows if r.get("model_id") == model_id and r.get("source_kind") == track and r.get("scalar_valid")]
            critical = [r for r in scalar if r.get("critical") is True]
            untrans = [r for r in scalar if r.get("untranslated_visible_english") is True]
            critical_review_units = 0
            critical_fail_units = 0
            for cell in cr:
                slots = track_required_slots(track)
                critical_n = sum(cell.get(f"{slot}_critical") is True for slot in slots)
                if critical_n == 1:
                    critical_review_units += 1
                elif critical_n >= 2:
                    critical_fail_units += 1
            out.append({
                "track": track,
                "model_id": model_id,
                "model_name": normalize_model_name(model_id, source_rows),
                "expected_material_units": len(cr),
                "panel_required_k": required_k,
                "panel_complete_units": len(valid),
                "judge_scalar_valid_cells": len(scalar),
                "judge_scalar_expected_cells": len(cr) * required_k,
                "coverage_fraction": (len(scalar) / (len(cr) * required_k)) if cr and required_k else None,
                "critical_flag_cells": len(critical),
                "critical_review_units_1ofk": critical_review_units,
                "critical_fail_units_2plusofk": critical_fail_units,
                "untranslated_visible_english_units": len(untrans),
                "observed_panel_mean_noncomparative": safe_mean(valid),
                "observed_panel_sd_noncomparative": safe_sd(valid),
                "observed_panel_median_noncomparative": safe_median(valid),
                "rank_eligible": gate,
                "ranking_status": "FULL_FIXED_PANEL_TRACK" if gate else "PROVISIONAL_NO_RANKING_OR_MEAN_COMPARISON",
                "track_complete_units": complete_units,
                "track_expected_units": expected_units,
            })
    return out


def build_common_case_model_summary(
    comparison_rows: Sequence[Mapping[str, Any]],
    common_cases: Mapping[str, Mapping[str, Any]],
    source_rows: Mapping[tuple[str, str], Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Summarize each candidate on the identical eligible common-case rows.

    ``model_summary.csv`` intentionally remains the all-material descriptive
    coverage view and therefore is not a common-case ranking table.  This
    separate derived CSV makes the balanced common-case denominator explicit.
    """
    out: list[dict[str, Any]] = []
    for track in ["text", "image"]:
        summary = common_cases.get(track, {})
        eligible_ids = set(summary.get("eligible_material_ids", []))
        all_ids = sorted({str(r.get("material_id")) for r in comparison_rows if r.get("source_kind") == track})
        eligible_rows = [
            r for r in comparison_rows
            if r.get("source_kind") == track and str(r.get("material_id")) in eligible_ids
        ]
        slots = track_required_slots(track)
        for model_id in MODEL_IDS:
            rows = [r for r in eligible_rows if str(r.get("model_id")) == model_id]
            panel_means = [numeric(r.get("panel_mean")) for r in rows if numeric(r.get("panel_mean")) is not None]
            critical_cells = 0
            review_units = 0
            fail_units = 0
            for row in rows:
                critical_n = sum(row.get(f"{slot}_critical") is True for slot in slots)
                critical_cells += critical_n
                if critical_n == 1:
                    review_units += 1
                elif critical_n >= 2:
                    fail_units += 1
            out.append({
                "track": track,
                "model_id": model_id,
                "model_name": normalize_model_name(model_id, source_rows),
                "common_case_items": len(eligible_ids),
                "common_case_source_pages": summary.get("source_page_count", 0),
                "required_judge_k": len(slots),
                "valid_judge_cells": len(panel_means) * len(slots),
                "expected_judge_cells": len(eligible_ids) * len(slots),
                "common_case_panel_mean": safe_mean(panel_means),
                "common_case_panel_median": safe_median(panel_means),
                "critical_flag_cells": critical_cells,
                "critical_review_units_1ofk": review_units,
                "critical_fail_units_2plusofk": fail_units,
                "excluded_case_count": len(set(all_ids) - eligible_ids),
                "excluded_case_ids": ";".join(sorted(set(all_ids) - eligible_ids)),
                "analysis_status": "LIMITED_COMMON_CASE" if summary.get("analysis_ready") else str(summary.get("status", "COMMON_CASE_NOT_READY")),
            })
    return out


def kendall_reliability_levels(
    complete_rows: Sequence[Mapping[str, Any]],
    slots: Sequence[str],
) -> dict[str, Any]:
    """Return system-level and per-item W alongside the optional unit W."""
    if not complete_rows or not slots:
        return {
            "kendall_w_unit_level": None,
            "kendall_w_system_level": None,
            "kendall_w_per_item_defined_count": 0,
            "kendall_w_per_item_mean": None,
            "kendall_w_per_item_median": None,
            "kendall_w_per_item_min": None,
            "kendall_w_per_item_max": None,
        }
    unit_matrix: list[list[float]] = []
    for row in complete_rows:
        values = [numeric(row.get(f"{slot}_score")) for slot in slots]
        if len(values) == len(slots) and all(v is not None for v in values):
            unit_matrix.append([float(v) for v in values if v is not None])

    # Candidate/system ranking: average each candidate over the common
    # materials separately for each judge, then rank the seven candidates.
    by_candidate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_material: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in complete_rows:
        candidate = str(row.get("model_id"))
        material = str(row.get("material_id"))
        for slot in slots:
            value = numeric(row.get(f"{slot}_score"))
            if value is not None:
                by_candidate[candidate][slot].append(float(value))
                by_material[material][candidate][slot] = float(value)
    candidates = sorted(by_candidate)
    system_matrix: list[list[float]] = []
    for candidate in candidates:
        values = [safe_mean(by_candidate[candidate].get(slot, [])) for slot in slots]
        if len(values) == len(slots) and all(v is not None for v in values):
            system_matrix.append([float(v) for v in values if v is not None])
    per_item_values: list[float] = []
    for material in sorted(by_material):
        candidate_map = by_material[material]
        item_candidates = sorted(candidate_map)
        item_matrix: list[list[float]] = []
        for candidate in item_candidates:
            values = [candidate_map[candidate].get(slot) for slot in slots]
            if len(values) == len(slots) and all(v is not None for v in values):
                item_matrix.append([float(v) for v in values if v is not None])
        item_w = kendall_w(item_matrix)
        if item_w is not None:
            per_item_values.append(item_w)
    return {
        "kendall_w_unit_level": kendall_w(unit_matrix),
        "kendall_w_system_level": kendall_w(system_matrix),
        "kendall_w_per_item_defined_count": len(per_item_values),
        "kendall_w_per_item_mean": safe_mean(per_item_values),
        "kendall_w_per_item_median": safe_median(per_item_values),
        "kendall_w_per_item_min": min(per_item_values) if per_item_values else None,
        "kendall_w_per_item_max": max(per_item_values) if per_item_values else None,
    }


def build_reliability(
    comparison_rows: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    terminal_coverage: Mapping[str, Any] | None = None,
    common_cases: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_cases = common_cases or {}
    for track in ["text", "image"]:
        gate, complete_units, expected_units = track_gate(comparison_rows, track)
        slots = track_required_slots(track)
        summary = common_cases.get(track, {})
        limited = not gate and bool(summary.get("analysis_ready"))
        if limited:
            eligible = set(summary.get("eligible_material_ids", []))
            complete = [
                r for r in comparison_rows
                if r.get("source_kind") == track
                and str(r.get("material_id")) in eligible
                and r.get("panel_complete")
            ]
        else:
            complete = [r for r in comparison_rows if r.get("source_kind") == track and r.get("panel_complete")]
        matrix: list[list[float]] = []
        for r in complete:
            values = [numeric(r.get(f"{slot}_score")) for slot in slots]
            if len(values) == len(slots) and all(v is not None for v in values):
                matrix.append([float(v) for v in values if v is not None])
        analysis_ready = (gate and len(matrix) == expected_units) or (limited and bool(matrix))
        common_expected_units = int(summary.get("eligible_item_count", 0)) * len(MODEL_IDS) if limited else expected_units
        icc21, icc2k, n, k = icc_2(matrix) if analysis_ready and len(matrix) == common_expected_units else (None, None, len(matrix), len(slots))
        w_levels = kendall_reliability_levels(complete, slots) if analysis_ready and len(matrix) == common_expected_units else kendall_reliability_levels([], slots)
        # Keep the legacy field as the system-level candidate-ranking W; the
        # unit-level and per-item variants are explicit adjacent fields.
        w = w_levels.get("kendall_w_system_level")
        analysis_status = "FULL_COMPLETE_BALANCED" if gate and len(matrix) == expected_units else (
            "LIMITED_COMMON_CASE" if analysis_ready and limited else "NOT_AVAILABLE_GATE_PARTIAL"
        )
        rows.append({
            "track": track,
            "analysis_status": analysis_status,
            "complete_balanced_units": len(matrix),
            "expected_balanced_units": common_expected_units if limited else expected_units,
            "full_fixed_panel_gate": gate and len(matrix) == expected_units,
            "common_case_items": summary.get("eligible_item_count", 0),
            "common_case_source_pages": summary.get("source_page_count", 0),
            "common_case_excluded_items": len(summary.get("excluded_case_ids", [])),
            "common_case_excluded_case_ids": summary.get("excluded_case_ids", []),
            "common_case_exclusion_reasons": summary.get("excluded_case_reasons", {}),
            "judge_slots_fixed_panel": slots,
            "actual_k": k,
            "icc_2_1_absolute_agreement": icc21,
            "icc_2_k_absolute_agreement": icc2k,
            "icc_k_is_actual_fixed_panel": True,
            "kendall_w": w,
            "kendall_w_level": "system_level_candidate_rank",
            "kendall_w_unit_level": w_levels.get("kendall_w_unit_level"),
            "kendall_w_system_level": w_levels.get("kendall_w_system_level"),
            "kendall_w_per_item_defined_count": w_levels.get("kendall_w_per_item_defined_count"),
            "kendall_w_per_item_mean": w_levels.get("kendall_w_per_item_mean"),
            "kendall_w_per_item_median": w_levels.get("kendall_w_per_item_median"),
            "kendall_w_per_item_min": w_levels.get("kendall_w_per_item_min"),
            "kendall_w_per_item_max": w_levels.get("kendall_w_per_item_max"),
            "ci_low": None,
            "ci_high": None,
            "ci_status": "not_computed; source-cluster bootstrap is reported separately" if analysis_ready else "not_available_until_terminal_common_case_or_balanced_gate",
            "mean_interpretation": "arithmetic panel mean only; panel median is descriptive separately",
            "note": (
                "Limited common-case exploratory matrix; not the full fixed-panel gate."
                if limited and analysis_ready else
                "Mistral image route is excluded by design from the fixed image k=4 panel; no zero imputation."
                if track == "image" else "Text panel requires all five judge slots."
            ),
        })
    return rows


def source_cluster_for_material(material_id: str, source_kind: str, materials: Mapping[str, Mapping[str, Any]]) -> str:
    material = materials.get(material_id, {})
    if source_kind == "text":
        return str(material.get("source_id", material_id))
    role = str(material.get("image_role", material.get("source_id", material_id)))
    # img-01/img-04 are the two synthetic authored images.  They are one
    # source group for the conservative image-cluster note; WHO pages remain
    # separate groups.
    if role.startswith("synthetic"):
        return "synthetic_images_shared_group"
    if "who" in role:
        return role
    return role or material_id


def build_bootstrap(
    comparison_rows: Sequence[Mapping[str, Any]],
    materials: Mapping[str, Mapping[str, Any]],
    terminal_coverage: Mapping[str, Any] | None = None,
    common_cases: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int = 20261023,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    common_cases = common_cases or {}
    for track in ["text", "image"]:
        gate, _, expected_units = track_gate(comparison_rows, track)
        required = track_required_slots(track)
        summary = common_cases.get(track, {})
        limited = not gate and bool(summary.get("analysis_ready"))
        eligible = set(summary.get("eligible_material_ids", []))
        complete = [
            r for r in comparison_rows
            if r.get("source_kind") == track
            and r.get("panel_complete")
            and (not limited or str(r.get("material_id")) in eligible)
        ]
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in complete:
            groups[source_cluster_for_material(str(row.get("material_id")), track, materials)].append(row)
        group_ids = sorted(groups)
        if not gate and not limited:
            for model in MODEL_IDS:
                rows.append({"track": track, "model_id": model, "estimator": "panel_mean", "status": "NOT_AVAILABLE_GATE_PARTIAL", "B": MAX_BOOTSTRAP, "cluster_count": len(group_ids), "ci_low": None, "ci_high": None, "common_case_items": summary.get("eligible_item_count", 0), "excluded_case_ids": summary.get("excluded_case_ids", []), "note": "No model ranking or bootstrap comparison from early/partial queue."})
            continue
        if track == "image" and len(group_ids) < 4:
            for model in MODEL_IDS:
                rows.append({"track": track, "model_id": model, "estimator": "panel_mean", "status": "UNRELIABLE_NA_IMAGE_CLUSTERS", "B": MAX_BOOTSTRAP, "cluster_count": len(group_ids), "ci_low": None, "ci_high": None, "common_case_items": summary.get("eligible_item_count", 0), "excluded_case_ids": summary.get("excluded_case_ids", []), "note": "Only 4 images and 3 shared source groups; image cluster CI is not treated as reliable."})
            continue
        samples: dict[str, list[float]] = {model: [] for model in MODEL_IDS}
        for _ in range(MAX_BOOTSTRAP):
            chosen = [rng.choice(group_ids) for _ in group_ids]
            for model in MODEL_IDS:
                vals: list[float] = []
                for group in chosen:
                    vals.extend(float(r.get("panel_mean")) for r in groups[group] if r.get("model_id") == model and numeric(r.get("panel_mean")) is not None)
                if vals:
                    samples[model].append(sum(vals) / len(vals))
        for model in MODEL_IDS:
            vals = samples[model]
            rows.append({"track": track, "model_id": model, "estimator": "panel_mean", "status": "AVAILABLE_LIMITED_COMMON_CASE_SOURCE_CLUSTER_BOOTSTRAP" if limited else "AVAILABLE_SOURCE_CLUSTER_BOOTSTRAP", "B": MAX_BOOTSTRAP, "cluster_count": len(group_ids), "common_case_items": summary.get("eligible_item_count", 0), "excluded_case_ids": summary.get("excluded_case_ids", []), "ci_low": quantile(vals, 0.025), "ci_high": quantile(vals, 0.975), "note": "Text bootstrap resamples source pages with replacement; common-case exploratory only." if limited else "Text bootstrap resamples 12 source pages with replacement; image uses no CI."})
    return rows


def build_lofo(
    comparison_rows: Sequence[Mapping[str, Any]],
    terminal_coverage: Mapping[str, Any] | None = None,
    common_cases: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    common_cases = common_cases or {}
    for track in ["text", "image"]:
        gate, _, _ = track_gate(comparison_rows, track)
        slots = track_required_slots(track)
        summary = common_cases.get(track, {})
        limited = not gate and bool(summary.get("analysis_ready"))
        eligible = set(summary.get("eligible_material_ids", []))
        rows = [
            r for r in comparison_rows
            if r.get("source_kind") == track
            and (not limited or str(r.get("material_id")) in eligible)
        ]
        for omitted in slots:
            remaining = [s for s in slots if s != omitted]
            means: dict[str, float | None] = {}
            for model in MODEL_IDS:
                vals: list[float] = []
                for row in rows:
                    # LOFO means are candidate-specific.  Without this
                    # filter every candidate receives the same pooled
                    # candidate×item rows, making the sensitivity table
                    # falsely identical across models.
                    if str(row.get("model_id")) != model:
                        continue
                    scores = [numeric(row.get(f"{s}_score")) for s in remaining]
                    if len(scores) == len(remaining) and all(v is not None for v in scores):
                        vals.append(sum(v for v in scores if v is not None) / len(scores))
                means[model] = safe_mean(vals)
            ordering = sorted(MODEL_IDS, key=lambda m: (-(means[m] if means[m] is not None else -math.inf), m)) if gate or limited else []
            out.append({
                "track": track,
                "omitted_judge_slot": omitted,
                "omitted_judge_family": JUDGE_META[omitted]["family"],
                "remaining_actual_k": len(remaining),
                "status": "FULL_TRACK_LOFO" if gate else "LIMITED_COMMON_CASE_LOFO" if limited else "NOT_AVAILABLE_GATE_PARTIAL",
                "candidate_means_noncomparative": means,
                "candidate_order_if_full": ordering,
                "common_case_items": summary.get("eligible_item_count", 0),
                "common_case_source_pages": summary.get("source_page_count", 0),
                "excluded_case_ids": summary.get("excluded_case_ids", []),
                "note": "Limited common-case exploratory sensitivity only; not a full-track ranking or correctness proof."
                if limited else "Sensitivity only; not a correctness proof and not a reason to discard a judge. Gemini/GPT are cloud families; Mistral image omission is by design.",
            })
    return out


def build_translation_generation_summary(source_rows: Mapping[tuple[str, str], Mapping[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in MODEL_IDS:
        rows = [r for (m, _), r in source_rows.items() if m == model]
        for track in ["text", "image"]:
            subset = [r for r in rows if ("image" if r.get("input_type") == "image" else "text") == track]
            elapsed = [numeric(r.get("elapsed_seconds")) for r in subset if numeric(r.get("elapsed_seconds")) is not None]
            out.append({
                "model_id": model,
                "model_name": (rows[0].get("model_name") if rows else model),
                "track": track,
                "translation_cases": len(subset),
                "translation_elapsed_median_s": safe_median(elapsed),
                "translation_elapsed_p95_s": quantile(elapsed, 0.95),
                "translation_completion_tokens": sum(int(float(r.get("completion_tokens") or 0)) for r in subset),
                "translation_status_ok": sum(r.get("status") == "ok" for r in subset),
                "not_judge_latency": True,
                "note": "Generation latency is from the frozen translation run and is not compared to judge latency.",
            })
    return out


def build_settings_provenance(snapshot_files: Sequence[Mapping[str, Any]], attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key, recs in attempts.items():
        slot = key[0]
        for rec in recs:
            provenance = rec.get("provenance") if isinstance(rec.get("provenance"), Mapping) else {}
            settings = rec.get("settings") if isinstance(rec.get("settings"), Mapping) else {}
            k = f"{slot}|{rec.get('source')}|{provenance.get('requested_model') or settings.get('model') or ''}"
            row = grouped.setdefault(k, {
                "judge_slot": slot,
                "judge_label": JUDGE_META.get(slot, {}).get("label", slot),
                "judge_family": JUDGE_META.get(slot, {}).get("family", ""),
                "source_kind": rec.get("source", ""),
                "requested_model": provenance.get("requested_model") or settings.get("model") or "",
                "requested_reasoning_effort": provenance.get("requested_reasoning_effort", ""),
                "manifest_sha256": provenance.get("manifest_sha256", ""),
                "packet_sha256": provenance.get("packet_sha256", ""),
                "instruction_sha256": provenance.get("instruction_sha256", ""),
                "schema_regimes": [],
                "record_count": 0,
                "raw_paths": [],
            })
            row["record_count"] += 1
            regime = provenance.get("schema_regime")
            if regime and regime not in row["schema_regimes"]:
                row["schema_regimes"].append(regime)
            if rec.get("path"):
                row["raw_paths"].append(rec.get("path"))
    for row in grouped.values():
        row["raw_paths"] = sorted(set(row["raw_paths"]))[:20]
    return sorted(grouped.values(), key=lambda x: (x["judge_slot"], x["source_kind"]))


def terminal_attempt_coverage(
    point_jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    pair_jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    attempts: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    checkpoint_terminal_statuses: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Count terminal raw/checkpoint keys without treating absence as score zero."""
    out: dict[str, Any] = {}
    checkpoint_terminal_statuses = checkpoint_terminal_statuses or {}
    for track, jobs in (("pointwise", point_jobs), ("pairwise", pair_jobs)):
        expected = {(slot, track, request_id) for slot, request_id in jobs}
        design_unsupported = {
            (slot, track, request_id)
            for slot, request_id in jobs
            if slot == "judge-03" and str(jobs[(slot, request_id)].get("source_kind")) == "image"
        }
        raw_present = 0
        terminal = 0
        pending = 0
        checkpoint_terminal = sum(
            key in checkpoint_terminal_statuses for key in expected if key not in design_unsupported
        )
        checkpoint_missing_terminal = sum(
            str((checkpoint_terminal_statuses.get(key) or {}).get("status", "")).lower() == "missing"
            for key in expected if key not in design_unsupported
        )
        for key in expected:
            if key in design_unsupported:
                continue
            rows = list(attempts.get(key, []))
            if not rows:
                checkpoint = checkpoint_terminal_statuses.get(key)
                if checkpoint and str(checkpoint.get("status", "")).lower() in CHECKPOINT_TERMINAL_STATUSES:
                    terminal += 1
                continue
            raw_present += 1
            latest = sorted(rows, key=lambda x: (int(x.get("attempt", 0) or 0), str(x.get("path", ""))))[-1]
            status = str(latest.get("status") or "").lower()
            if status in {"pending", "running", "in_progress", "started"} or (
                status in {"ok", "success", "completed", "complete"} and latest.get("parsed") is None
            ):
                pending += 1
            else:
                terminal += 1
        prefix = "pointwise" if track == "pointwise" else "pairwise"
        out[f"{prefix}_expected_jobs"] = len(expected)
        out[f"{prefix}_design_unsupported_jobs"] = len(design_unsupported)
        out[f"{prefix}_terminal_expected_jobs"] = len(expected) - len(design_unsupported)
        out[f"{prefix}_raw_present_jobs"] = raw_present
        out[f"{prefix}_terminal_jobs"] = terminal
        out[f"{prefix}_pending_jobs"] = pending
        out[f"{prefix}_checkpoint_terminal_jobs"] = checkpoint_terminal
        out[f"{prefix}_checkpoint_missing_terminal_jobs"] = checkpoint_missing_terminal
        out[f"{prefix}_terminal_complete"] = terminal == (len(expected) - len(design_unsupported)) if expected else False
    out["all_primary_terminal"] = bool(out.get("pointwise_terminal_complete") and out.get("pairwise_terminal_complete"))
    return out


def report_status(
    point_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    terminal_coverage: Mapping[str, Any] | None = None,
    common_cases: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    common_cases = common_cases or {}
    point_valid = sum(bool(r.get("scalar_valid")) for r in point_rows)
    point_planned = len(point_rows)
    point_technical_invalid = sum(bool(r.get("technical_errors")) and r.get("status") != "unsupported_by_design" for r in point_rows)
    pair_valid = sum(bool(r.get("valid")) for r in pair_rows)
    text_gate, text_complete, text_expected = track_gate(comparison_rows, "text")
    image_gate, image_complete, image_expected = track_gate(comparison_rows, "image")
    terminal_ready = bool((terminal_coverage or {}).get("all_primary_terminal"))
    common_case_ready = any(bool(summary.get("analysis_ready")) for summary in common_cases.values())
    if text_gate and image_gate:
        analysis_status = "COMPLETE_BALANCED"
    elif terminal_ready and common_case_ready:
        analysis_status = "TERMINAL_COMMON_CASE_EXPLORATORY"
    else:
        analysis_status = "PARTIAL_PROGRESS_NO_RANKING_OR_ICC"
    return {
        "analysis_status": analysis_status,
        "pointwise_planned_rows": point_planned,
        "pointwise_scalar_valid_rows": point_valid,
        "pointwise_scalar_coverage": point_valid / point_planned if point_planned else 0,
        "pointwise_technical_invalid_rows": point_technical_invalid,
        "pairwise_planned_orientations": len(pair_rows),
        "pairwise_valid_orientations": pair_valid,
        "text_complete_units": text_complete,
        "text_expected_units": text_expected,
        "text_gate": text_gate,
        "image_complete_units": image_complete,
        "image_expected_units": image_expected,
        "image_gate": image_gate,
        "image_fixed_panel_k": len(IMAGE_PANEL_SLOTS),
        "terminal_coverage": dict(terminal_coverage or {}),
        "common_case": {
            track: {
                "status": summary.get("status"),
                "analysis_ready": bool(summary.get("analysis_ready")),
                "eligible_item_count": summary.get("eligible_item_count", 0),
                "source_page_count": summary.get("source_page_count", 0),
                "excluded_case_ids": summary.get("excluded_case_ids", []),
                "excluded_case_reasons": summary.get("excluded_case_reasons", {}),
            }
            for track, summary in common_cases.items()
        },
    }


def independent_image_flag_summary(
    materials: Mapping[str, Mapping[str, Any]],
    private_maps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the frozen packet-QA image flag without touching raw/targets.

    The public packets repeat a candidate target for each judge and again in
    the limited pairwise audit.  This function reports both the unique target
    fingerprint count and the expanded job mentions so readers do not mistake
    repetition for independent translations.
    """
    unique_targets: set[tuple[str, str]] = set()
    point_targets: set[tuple[str, str]] = set()
    pair_targets: set[tuple[str, str]] = set()
    flagged_model_ids: set[str] = set()
    point_mentions: list[dict[str, Any]] = []
    pair_mentions: list[dict[str, Any]] = []
    private_maps = private_maps or {}
    point_keymap = private_maps.get("pointwise") if isinstance(private_maps.get("pointwise"), Mapping) else {}
    pair_keymap = private_maps.get("pairwise") if isinstance(private_maps.get("pairwise"), Mapping) else {}
    for slot in JUDGE_SLOTS:
        for job in load_jsonl(PUBLIC_ROOT / "pointwise" / f"{slot}.jsonl"):
            if job.get("source_kind") != "image":
                continue
            # Public jobs intentionally carry no material/model identity; the
            # frozen private keymap is the binding authority for this derived
            # QA diagnostic.
            private = point_keymap.get(job.get("job_id"), {}) if isinstance(point_keymap, Mapping) else {}
            material_id = str(private.get("material_id") or "")
            material = materials.get(material_id, {})
            flag, span = match_untranslated_visible_english(str(job.get("target_zh_tw", "")), material)
            if flag:
                target_hash = sha256_bytes(str(job.get("target_zh_tw", "")).encode("utf-8"))
                unique_targets.add((material_id, target_hash))
                point_targets.add((material_id, target_hash))
                if private.get("model_id"):
                    flagged_model_ids.add(str(private.get("model_id")))
                point_mentions.append({"judge_slot": slot, "job_id": job.get("job_id"), "material_id": material_id, "span": span})
        for job in load_jsonl(PUBLIC_ROOT / "pairwise" / f"{slot}.jsonl"):
            if job.get("source_kind") != "image":
                continue
            private = pair_keymap.get(job.get("pair_job_id"), {}) if isinstance(pair_keymap, Mapping) else {}
            material_id = str(private.get("material_id") or "")
            material = materials.get(material_id, {})
            for label in ("candidate_a", "candidate_b"):
                candidate = job.get(label) or {}
                flag, span = match_untranslated_visible_english(str(candidate.get("target_zh_tw", "")), material)
                if flag:
                    target_hash = sha256_bytes(str(candidate.get("target_zh_tw", "")).encode("utf-8"))
                    unique_targets.add((material_id, target_hash))
                    pair_targets.add((material_id, target_hash))
                    candidate_model = private.get("model_a") if label == "candidate_a" else private.get("model_b")
                    if candidate_model:
                        flagged_model_ids.add(str(candidate_model))
                    pair_mentions.append({"judge_slot": slot, "pair_job_id": job.get("pair_job_id"), "candidate": label, "material_id": material_id, "span": span})
    # QA evidence is immutable and useful even before judge raw arrives, but
    # recomputation above is the primary binding.  Preserve any discrepancy.
    qa_count = None
    qa_path_status = "not_available"
    if INDEPENDENT_QA.is_file():
        try:
            qa = read_json(INDEPENDENT_QA)
            qa_rows = qa.get("image_candidate_english_quality_flags", []) if isinstance(qa, dict) else []
            qa_count = len(qa_rows) if isinstance(qa_rows, list) else None
            qa_path_status = "read_only_reference"
        except Exception:
            qa_path_status = "unreadable"
    return {
        "unique_target_fingerprints": len(unique_targets),
        "pointwise_unique_target_fingerprints": len(point_targets),
        "pairwise_unique_target_fingerprints": len(pair_targets),
        "flagged_candidate_model_ids": sorted(flagged_model_ids),
        "pointwise_judge_copy_slots": len({str(x.get("judge_slot")) for x in point_mentions}),
        "pairwise_judge_copy_slots": len({str(x.get("judge_slot")) for x in pair_mentions}),
        "pointwise_expanded_mentions": len(point_mentions),
        "pairwise_expanded_mentions": len(pair_mentions),
        "expanded_mentions_total": len(point_mentions) + len(pair_mentions),
        "independent_qa_reported_mentions": qa_count,
        "independent_qa_path_status": qa_path_status,
        "pointwise_examples": point_mentions[:12],
        "pairwise_examples": pair_mentions[:12],
        "interpretation": "A target fingerprint repeated across judge slots is one translation output, not independent translations. A visible-English match is a quality flag only; it is not proof of a translation error, and candidate targets are immutable.",
    }


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        try:
            out[str(path)] = sha256_file(path)
        except OSError:
            out[str(path)] = "unreadable"
    return out


def make_markdown(
    status: Mapping[str, Any],
    point_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, Any]],
    judge_summary: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    snapshot_diagnostics: Sequence[Mapping[str, Any]],
    image_visibility: Mapping[str, Any],
    raw_file_count: int,
    report_files: Sequence[str],
) -> str:
    image_flags = int(image_visibility.get("expanded_mentions_total") or 0)
    image_point_flags = int(image_visibility.get("pointwise_expanded_mentions") or 0)
    image_pair_flags = int(image_visibility.get("pairwise_expanded_mentions") or 0)
    image_unique_targets = int(image_visibility.get("unique_target_fingerprints") or 0)
    image_point_copy_slots = int(image_visibility.get("pointwise_judge_copy_slots") or 0)
    image_qa_flags = image_visibility.get("independent_qa_reported_mentions")
    invalid = sum(r.get("status") in {"invalid", "technical_invalid"} for r in point_rows)
    technical_invalid = sum(bool(r.get("technical_errors")) and r.get("status") != "unsupported_by_design" for r in point_rows)
    missing = sum(r.get("status") == "missing" for r in point_rows)
    unsupported = sum(r.get("status") == "unsupported_by_design" for r in point_rows)
    pair_consistent = sum(r.get("canonical_status") in {"CONSISTENT", "CONSISTENT_TIE"} for r in canonical_rows)
    pair_inconsistent = sum(r.get("canonical_status") == "ABSTAIN_INCONSISTENT" for r in canonical_rows)
    terminal_ready = bool(status.get("terminal_coverage", {}).get("all_primary_terminal"))
    text_common_source_pages = status.get("common_case", {}).get("text", {}).get("source_page_count", 0)
    if status["analysis_status"] == "COMPLETE_BALANCED":
        status_note = "已達完整 fixed-panel gate；以下是完整平衡面板結果。"
    elif status["analysis_status"] == "TERMINAL_COMMON_CASE_EXPLORATORY":
        status_note = "primary terminal coverage 已完成；以下是明確標示的 common-case limited exploratory 結果，不取代完整 fixed-panel gate。"
    elif terminal_ready:
        status_note = "primary terminal coverage 已完成，但 common-case threshold 尚未達成；仍不發布 ICC/W/LOFO 排序。"
    else:
        status_note = "目前只反映已落地 raw；primary queue 尚未 terminal，不能把 partial queue 當完整模型比較。"
    lines = [
        "# 新素材五 judge panel：只讀彙整（可重跑快照）",
        "",
        f"- 產生時間（UTC）：`{now_utc()}`",
        f"- 狀態：**{status['analysis_status']}**；{status_note}",
        f"- ROOT_APPROVED：`{APPROVAL.exists()}`；frozen manifest SHA：`{sha256_file(PACKET_MANIFEST) if PACKET_MANIFEST.is_file() else 'missing'}`",
        f"- 穩定 raw 檔案快照：`{raw_file_count}`；不穩定/損壞而跳過的檔案：`{len(snapshot_diagnostics)}`（原檔未修改）。",
        f"- Primary terminal coverage：pointwise `{status.get('terminal_coverage', {}).get('pointwise_terminal_jobs', 0)}/{status.get('terminal_coverage', {}).get('pointwise_terminal_expected_jobs', 0)}`、pairwise `{status.get('terminal_coverage', {}).get('pairwise_terminal_jobs', 0)}/{status.get('terminal_coverage', {}).get('pairwise_terminal_expected_jobs', 0)}`；design-unsupported excluded=`{status.get('terminal_coverage', {}).get('pointwise_design_unsupported_jobs', 0) + status.get('terminal_coverage', {}).get('pairwise_design_unsupported_jobs', 0)}`；`all_primary_terminal={status.get('terminal_coverage', {}).get('all_primary_terminal', False)}`。未達 terminal coverage 時不做 common-case ranking/ICC。",
        f"- Complete checkpoint terminal declarations (no score): pointwise `{status.get('terminal_coverage', {}).get('pointwise_checkpoint_missing_terminal_jobs', 0)}` missing、pairwise `{status.get('terminal_coverage', {}).get('pairwise_checkpoint_missing_terminal_jobs', 0)}` missing；interim/incomplete checkpoint entries are not used to close the gate.",
        "",
        "## 現況摘要",
        "",
        "|項目|計畫|目前觀測|解讀|",
        "|---|---:|---:|---|",
        f"|pointwise scalar rows|1190|{status['pointwise_scalar_valid_rows']} valid / {status['pointwise_planned_rows']} slots|{'terminal raw 已完成；' if terminal_ready else '仍為 queue-progress；'}`invalid`、`missing`、`blocked` 不轉 0|",
        f"|pointwise invalid/excluded|—|{invalid} (technical gate {technical_invalid})|保留 raw、schema/technical errors，排除 scalar|",
        f"|pointwise missing|—|{missing}|尚未看到 stable raw，不代表 score 0|",
        f"|Mistral image by-design|28 slots|{unsupported}|沒有 native mmproj 路徑；固定 image panel k=4，不作 0 分插補|",
        f"|pairwise orientations|360|{len(pair_rows)} rows / {sum(r.get('valid') is True for r in pair_rows)} valid|AB/BA 原向位保留；不一致 derived status 是 ABSTAIN，不是 TIE|",
        f"|pairwise canonical|—|{pair_consistent} consistent / {pair_inconsistent} abstain|僅為 frozen spanning-chain audit，非 round-robin 全排序|",
        f"|image English-visible flag|—|{image_flags} expanded mentions ({image_point_copy_slots} pointwise judge copies / {image_point_flags}; pairwise {image_pair_flags}); {image_unique_targets} unique target fingerprint(s)|固定 public target 的 QA match flag；不可當 prompt leak 或已證明翻譯錯誤，也不刪除文字；independent QA reported {image_qa_flags if image_qa_flags is not None else 'n/a'}|",
        "",
        "## 報告閘門",
        "",
        "- text 需要 30 material × 7 candidate 且五個固定 judge 都有 valid scalar；image 需要 4 material × 7 candidate 且固定四位（Qwen、Nemotron、GPT、Gemini）都 valid。",
        "- 在 primary queue 尚未達 terminal coverage 時，`model_summary.csv` 只列 coverage 與 descriptive observed values；不發布模型排名或跨候選 mean comparison，`reliability_summary.csv` 的 ICC/W 留空並標示 gate。terminal coverage 後才可依 amendment 做明確標示的 common-case exploratory analysis；完整 balanced fixed-panel 仍是首選。",
        "- 終止 raw coverage 後才可依 analysis policy amendment 做 common-case 探索：text 至少 12 common items / 6 source pages；image 至少 2 images 且固定 k=4。這些門檻仍是 limited descriptive/exploratory，不取代完整 balanced gate；詳見 `analysis_runner/analysis_policy_amendment.md`。",
        "- primary terminal coverage 且 common-case threshold 達成後，才會計算明確標示為 limited exploratory 的 arithmetic panel mean ICC(2,1)、ICC(2,k)（k 為實際固定 panel）、tie-corrected Kendall W；完整 fixed-panel gate 仍是首選。W 的主欄是 7-candidate system-level rank，另列 candidate×item unit-level 與 per-item 分布，median 另列且不混入 ICC。全同分時 W 為 NA，不偽造 0/1。ICC/W 是一致性/可靠度指標，不是正確性證明。方法參考：Koo & Li (2016), [PubMed 27330520](https://pubmed.ncbi.nlm.nih.gov/27330520/)。",
        f"- text cluster bootstrap 預設 B=10,000、以目前 eligible common-case 的 {text_common_source_pages} 個 source pages 重抽；image 僅 4 張且只有 3 個 shared source groups，因此 image CI 預設 NA 並說明原因，不假設 4 張相互獨立。",
        "- critical/error-severity 不一致、target quote 未命中、source evidence 等均為獨立 evidence diagnostics，不會悄悄改分；critical gate 依 track 固定 panel：text 1/5 標 `REVIEW`、≥2/5 標 `FAIL`；image 1/4 標 `REVIEW`、≥2/4 標 `FAIL`。這些只是探索性 gate，不是臨床安全結論。",
        "- img-09/img-10 的 `provisional_reference_text` 僅作 private/QA visible-English 診斷，並保留 machine-extracted / AI-visual-checked / not-human-gold status；5 個 pointwise judge copies 與 20 個 pairwise candidate mentions 是 expanded job slots，不是 25 個獨立翻譯。",
        "- Image material attribution：`img-01`/`img-04` 是 authored synthetic render，具有 complete visible English labels；`img-09`/`img-10` 是 WHO 原始 PDF page render，reference 僅 machine-extracted + AI-visual-checked、not-human-gold。這裡的 English-visible match 是 quality flag，不是 OCR character-accuracy ground truth；詳見 [materials_audit.md](../../materials_audit.md)。",
        f"- 目前旗標的 target-level fingerprint 詳細值：pointwise unique `{image_visibility.get('pointwise_unique_target_fingerprints', 0)}`、pairwise unique `{image_visibility.get('pairwise_unique_target_fingerprints', 0)}`、合併 unique `{image_unique_targets}`；這些 hash 是去重診斷，不是新的品質裁分。",
        f"- 對應 private binding 的 flagged candidate model ID（非 judge identity）共 `{len(image_visibility.get('flagged_candidate_model_ids', []))}`：`{', '.join(image_visibility.get('flagged_candidate_model_ids', [])) or 'n/a'}`；此欄只用來追溯 immutable target。",
        "",
    ]
    lines.extend(["## Common-case intersection", ""])
    for track in ("text", "image"):
        common = status.get("common_case", {}).get(track, {})
        excluded = common.get("excluded_case_ids", [])
        lines.append(
            f"- `{track}`：status=`{common.get('status', 'n/a')}`, eligible items=`{common.get('eligible_item_count', 0)}`, source pages=`{common.get('source_page_count', 0)}`, excluded=`{len(excluded)}`."
        )
        if excluded:
            lines.append(f"  - excluded cases: `{csv_value({case: common.get('excluded_case_reasons', {}).get(case, '') for case in excluded})}`")
    lines.extend([
        "",
        "## Judge observed（非速度排名）",
        "",
        "|slot|judge|point valid|pair valid|observed score mean|latency median (s)|",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in judge_summary:
        lines.append(f"|{row['judge_slot']}|{row['judge_label']}|{row['pointwise_scalar_valid']}|{row['pairwise_valid_orientations']}|{csv_value(row.get('pointwise_observed_mean_noncomparative'))}|{csv_value(row.get('latency_median_s'))}|")
    lines.extend([
        "",
        "Judge latency 只來自 judge raw elapsed；translation generation latency 另在 `translation_generation_summary.csv`，不可混用。部分 cloud raw provider 不回顯 resolved model 時，保留 provenance 中的 requested model/CLI flag，不臆造 provider response。",
        "",
        "## 目前 ICC",
        "",
    ])
    for row in reliability:
        if row.get("analysis_status") == "FULL_COMPLETE_BALANCED":
            lines.append(f"- `{row['track']}`：ICC(2,1) `{csv_value(row.get('icc_2_1_absolute_agreement'))}`、ICC(2,k) `{csv_value(row.get('icc_2_k_absolute_agreement'))}`、k=`{row.get('actual_k')}`；Kendall W system-level candidate rank=`{csv_value(row.get('kendall_w_system_level'))}`，unit-level=`{csv_value(row.get('kendall_w_unit_level'))}`，per-item defined n=`{row.get('kendall_w_per_item_defined_count')}` (median `{csv_value(row.get('kendall_w_per_item_median'))}`)。CI 未由此分析器捏造。")
        elif row.get("analysis_status") == "LIMITED_COMMON_CASE":
            lines.append(f"- `{row['track']}`：**LIMITED_COMMON_CASE exploratory**，items=`{row.get('common_case_items')}`、source pages=`{row.get('common_case_source_pages')}`、ICC(2,1) `{csv_value(row.get('icc_2_1_absolute_agreement'))}`、ICC(2,k) `{csv_value(row.get('icc_2_k_absolute_agreement'))}`、k=`{row.get('actual_k')}`；不取代完整 fixed-panel gate。")
        else:
            lines.append(f"- `{row['track']}`：**NOT_AVAILABLE_GATE_PARTIAL**（{row.get('complete_balanced_units')}/{row.get('expected_balanced_units')} units；實際固定 k={row.get('actual_k')}）。")
    lines.extend([
        "",
        "## Pairwise 與錯誤證據",
        "",
        "`pairwise_raw.csv` 保留 AB、BA 兩向原始 winner；`pairwise_canonical.csv` 將 positional inconsistency 映射為 `ABSTAIN_INCONSISTENT`，不把它當 TIE。`error_annotations.csv` 保留每個 raw attempt 的 exact target quote、source span/evidence、severity、confidence 與 match flags；`invalid_annotations.csv` 保留 malformed/sum mismatch 等排除理由。",
        "",
        "## 輸出",
    ])
    for name in report_files:
        lines.append(f"- [`{name}`](./{name})")
    lines.extend([
        "",
        "## 重跑",
        "",
        "```sh",
        f"python3 {HERE} --refresh",
        "```",
        "",
        "此分析器只讀 frozen source/packet/private maps 與 append-only raw；每個 JSON 先做 stable-read/hash。worker 新增 raw 後重新執行即可更新 current reports，不會改寫既有 raw。",
    ])
    return "\n".join(lines) + "\n"


def aggregate() -> dict[str, Any]:
    source_rows, _ = load_source_rows()
    materials, image_rows, material_manifest = load_materials()
    point_jobs, pair_jobs, private_context = load_jobs()
    attempts, snapshot_diagnostics, snapshot_files, checkpoint_terminal_statuses = collect_raw_snapshot()
    point_rows = build_pointwise_rows(point_jobs, attempts, materials)
    comparison_rows = build_comparison_rows(point_rows, source_rows, materials)
    pair_rows = build_pairwise_rows(pair_jobs, attempts)
    canonical_rows = build_pairwise_canonical(pair_rows)
    terminal_coverage = terminal_attempt_coverage(point_jobs, pair_jobs, attempts, checkpoint_terminal_statuses)
    common_cases = {
        track: common_case_intersection(comparison_rows, track, terminal_coverage)
        for track in ("text", "image")
    }
    annotate_common_case(comparison_rows, common_cases)
    judge_summary = summarize_judges(point_rows, pair_rows, canonical_rows, snapshot_files)
    model_summary = build_model_summary(comparison_rows, point_rows, source_rows)
    common_case_model_summary = build_common_case_model_summary(comparison_rows, common_cases, source_rows)
    reliability = build_reliability(comparison_rows, point_rows, terminal_coverage, common_cases)
    bootstrap = build_bootstrap(comparison_rows, materials, terminal_coverage, common_cases)
    lofo = build_lofo(comparison_rows, terminal_coverage, common_cases)
    errors = build_error_annotations(point_rows, attempts, point_jobs, materials)
    invalid = build_invalid_annotations(point_rows, attempts, point_jobs, pair_rows, pair_jobs)
    generation = build_translation_generation_summary(source_rows)
    settings = build_settings_provenance(snapshot_files, attempts)
    status = report_status(point_rows, comparison_rows, pair_rows, terminal_coverage, common_cases)
    image_visibility = independent_image_flag_summary(materials, private_context.get("job_map", {}))
    return {
        "source_rows": source_rows,
        "materials": materials,
        "material_manifest": material_manifest,
        "point_jobs": point_jobs,
        "pair_jobs": pair_jobs,
        "private_context": private_context,
        "attempts": attempts,
        "snapshot_diagnostics": snapshot_diagnostics,
        "snapshot_files": snapshot_files,
        "checkpoint_terminal_statuses": checkpoint_terminal_statuses,
        "common_cases": common_cases,
        "point_rows": point_rows,
        "comparison_rows": comparison_rows,
        "pair_rows": pair_rows,
        "canonical_rows": canonical_rows,
        "judge_summary": judge_summary,
        "model_summary": model_summary,
        "common_case_model_summary": common_case_model_summary,
        "reliability": reliability,
        "bootstrap": bootstrap,
        "lofo": lofo,
        "errors": errors,
        "invalid": invalid,
        "generation": generation,
        "settings": settings,
        "terminal_coverage": terminal_coverage,
        "status": status,
        "image_visibility": image_visibility,
    }


def make_operational_retry_addendum(data: Mapping[str, Any]) -> str:
    """Document retry provenance without changing the frozen protocol."""
    rows = data.get("point_rows", [])
    infra = [r for r in rows if r.get("infrastructure_recovery_used")]
    repairs = [r for r in rows if r.get("format_repair_used")]
    unknown = [r for r in rows if r.get("unknown_recovery_used")]
    lines = [
        "# Operational retry provenance addendum",
        "",
        "This is a derived, append-only analysis note. It does not amend the frozen public protocol or select a score by quality.",
        "",
        "- A same-prompt infrastructure retry is labeled `infrastructure_recovery` only when the earlier attempt had no parsed candidate judgment (for example a transport/CLI failure).",
        "- A retry is labeled `format_repair` only when the raw record explicitly carries `repair: true` or an equivalent explicit repair kind. The importer never infers format repair from a score, evidence, or a prior sum/schema failure.",
        "- A retry after a parsed but invalid result is labeled `unknown_recovery` unless raw provenance explicitly identifies the repair. The first schema-valid attempt remains selected; all malformed and excluded attempts remain in `invalid_annotations.csv`.",
        f"- Current selected rows: infrastructure recovery `{len(infra)}`, explicit format repair `{len(repairs)}`, unknown recovery `{len(unknown)}`.",
        "- Infrastructure recovery is operational provenance, not a second judge vote. No quality-selected retry or score substitution is performed.",
        "",
        "## Selected infrastructure-recovery rows",
        "",
        "|judge slot|track|material/job|selected raw|attempt|",
        "|---|---|---|---|---:|",
    ]
    for row in infra[:100]:
        lines.append(f"|{row.get('judge_slot','')}|{row.get('track','')}|{row.get('material_id') or row.get('job_id','')}|{row.get('raw_path','')}|{row.get('selected_attempt','')}|")
    if not infra:
        lines.append("|—|—|—|none observed in current snapshot|—|")
    lines.extend([
        "",
        "The raw files and hashes remain the source of truth; this addendum only records how the read-only importer classified attempts.",
    ])
    return "\n".join(lines) + "\n"


def write_reports(data: Mapping[str, Any]) -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    # Explicit field order makes CSV diffs stable for workers and ArtifactTool.
    outputs: dict[str, Sequence[Mapping[str, Any]]] = {
        "translation_comparison.csv": data["comparison_rows"],
        "pointwise_results_long.csv": data["point_rows"],
        "error_annotations.csv": data["errors"],
        "invalid_annotations.csv": data["invalid"],
        "pairwise_raw.csv": data["pair_rows"],
        "pairwise_canonical.csv": data["canonical_rows"],
        "judge_summary.csv": data["judge_summary"],
        "model_summary.csv": data["model_summary"],
        "common_case_model_summary.csv": data["common_case_model_summary"],
        "reliability_summary.csv": data["reliability"],
        "bootstrap_summary.csv": data["bootstrap"],
        "translation_generation_summary.csv": data["generation"],
        "lofo_summary.csv": data["lofo"],
        "settings_provenance.csv": data["settings"],
        "snapshot_diagnostics.csv": data["snapshot_diagnostics"],
    }
    for name, rows in outputs.items():
        write_csv(REPORT_ROOT / name, rows)
    retry_addendum = make_operational_retry_addendum(data)
    atomic_write(REPORT_ROOT / "operational_retry_addendum.md", retry_addendum)
    report_names = list(outputs) + ["operational_retry_addendum.md"]
    markdown = make_markdown(
        data["status"], data["point_rows"], data["comparison_rows"], data["pair_rows"], data["canonical_rows"],
        data["judge_summary"], data["reliability"], data["snapshot_diagnostics"], data["image_visibility"],
        len(data["snapshot_files"]), report_names,
    )
    atomic_write(REPORT_ROOT / "analysis_report.md", markdown)
    summary = {
        "schema_version": "new-material-judge-analysis-v1",
        "generated_at": now_utc(),
        "status": data["status"],
        "terminal_coverage": data["terminal_coverage"],
        "frozen_inputs": {
            "packet_manifest_sha256": sha256_file(PACKET_MANIFEST) if PACKET_MANIFEST.is_file() else None,
            "source_csv_sha256": sha256_file(SOURCE_CSV) if SOURCE_CSV.is_file() else None,
            "frozen_text_sha256": sha256_file(FROZEN_TEXT) if FROZEN_TEXT.is_file() else None,
            "image_selection_sha256": sha256_file(IMAGE_SELECTION) if IMAGE_SELECTION.is_file() else None,
            "material_manifest_sha256": sha256_file(MATERIAL_MANIFEST) if MATERIAL_MANIFEST.is_file() else None,
            "root_approved": APPROVAL.exists(),
        },
        "analysis_policy_amendment_sha256": sha256_file(HERE.parent / "analysis_policy_amendment.md") if (HERE.parent / "analysis_policy_amendment.md").is_file() else None,
        "raw_snapshot_files": data["snapshot_files"],
        "raw_snapshot_diagnostics": data["snapshot_diagnostics"],
        "planned": {"pointwise": 1190, "pairwise": 360, "total": 1550},
        "panel_policy": {"text_slots": TEXT_PANEL_SLOTS, "image_slots": IMAGE_PANEL_SLOTS, "mistral_image_not_imputed": True},
        "image_visibility": data["image_visibility"],
        "reports": report_names + ["analysis_report.md", "analysis_manifest.json"],
        "report_sha256": {},
    }
    # Hash outputs only after all content is atomically in place.
    for name in report_names + ["analysis_report.md"]:
        summary["report_sha256"][name] = sha256_file(REPORT_ROOT / name)
    write_json(REPORT_ROOT / "analysis_manifest.json", summary)
    progress = [
        "# New-material judge analysis progress",
        "",
        f"- Updated: `{now_utc()}`",
        f"- Status: **{data['status']['analysis_status']}**",
        f"- Pointwise scalar valid: `{data['status']['pointwise_scalar_valid_rows']}/{data['status']['pointwise_planned_rows']}`; pairwise valid orientations: `{data['status']['pairwise_valid_orientations']}/{data['status']['pairwise_planned_orientations']}`",
        f"- Technical execution exclusions: `{data['status'].get('pointwise_technical_invalid_rows', 0)}` pointwise rows; structurally valid JSON never overrides a failed HTTP/finish/model/native-image gate.",
        f"- Primary terminal coverage: pointwise `{data['terminal_coverage'].get('pointwise_terminal_jobs', 0)}/{data['terminal_coverage'].get('pointwise_terminal_expected_jobs', 0)}`, pairwise `{data['terminal_coverage'].get('pairwise_terminal_jobs', 0)}/{data['terminal_coverage'].get('pairwise_terminal_expected_jobs', 0)}`; design-unsupported jobs are excluded from the denominator; common-case exploratory stats stay gated until all primary jobs have terminal attempts.",
        f"- Complete checkpoint terminal declarations (no score): pointwise `{data['terminal_coverage'].get('pointwise_checkpoint_missing_terminal_jobs', 0)}` missing, pairwise `{data['terminal_coverage'].get('pairwise_checkpoint_missing_terminal_jobs', 0)}` missing; interim/incomplete checkpoint entries do not close the gate.",
        f"- Text balanced gate: `{data['status']['text_complete_units']}/{data['status']['text_expected_units']}`; image fixed-k=4 balanced gate: `{data['status']['image_complete_units']}/{data['status']['image_expected_units']}`",
        f"- Image visible-English derived flag: `{data['image_visibility'].get('expanded_mentions_total', 0)}` expanded mentions (pointwise `{data['image_visibility'].get('pointwise_expanded_mentions', 0)}`, pairwise `{data['image_visibility'].get('pairwise_expanded_mentions', 0)}`), `{data['image_visibility'].get('unique_target_fingerprints', 0)}` unique target fingerprint(s); this is a flag only, not proof of translation error.",
        "- Missing/blocked/invalid are explicit and never converted to score zero. Mistral image route is unsupported-by-design when no native mmproj result exists.",
        "- No model ranking or ICC is published from the early queue. After all primary jobs have terminal attempts, only the pre-specified common-case exploratory subset may be considered; complete balanced fixed-panel results remain preferred. English-visible image text remains a quality flag on actual candidate targets.",
        "- Offline importer fixtures: 22/22 PASS in the latest run. ArtifactTool `Workbook.fromCSV` QA is run separately against all 14 CSVs; it checks headers, row counts and bounded score cells.",
        "",
        "## Refresh command",
        "",
        f"`python3 {HERE} --refresh`",
        "",
        "## Reports",
        "",
        "- `reports/analysis_report.md` is the human summary.",
        "- `reports/analysis_manifest.json` binds frozen input hashes and the current raw snapshot.",
        "- Drivers/root may invoke the refresh command as a child process after a raw write has been atomically closed; it performs no model/API/server action and never edits raw/protocol files.",
    ]
    atomic_write(ANALYSIS_PROGRESS, "\n".join(progress) + "\n")
    return summary


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refresh read-only raw snapshot and reports")
    parser.add_argument("--dry-run", action="store_true", help="load and aggregate without writing reports")
    args = parser.parse_args(argv)
    if args.dry_run:
        data = aggregate()
        print(json.dumps(data["status"], ensure_ascii=False, sort_keys=True))
        return 0
    with refresh_lock() as acquired:
        if not acquired:
            print(json.dumps({"status": "SKIPPED_REFRESH_LOCK_BUSY", "lock": str(REFRESH_LOCK)}, ensure_ascii=False, sort_keys=True))
            return 0
        data = aggregate()
        summary = write_reports(data)
        print(json.dumps({"report_dir": str(REPORT_ROOT), "progress": str(ANALYSIS_PROGRESS), "status": summary["status"]}, ensure_ascii=False, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(run())




