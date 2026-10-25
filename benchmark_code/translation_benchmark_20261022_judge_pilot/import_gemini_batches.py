#!/usr/bin/env python3
"""Lossless adapter/assembler for Gemini prompt-format-v2 captures.

The Gemini UI is an external capture surface, so its raw response is kept
byte-for-byte and is never repaired in place.  This module only adds an
operator-authenticated envelope, validates the batch against the frozen
public packet, and records raw/prompt/manifest provenance.  It deliberately
does not repair scores, subscores, error labels, quotes, offsets, or text.

Use ``adapt`` once per fresh-chat capture and ``assemble`` to combine the six
independent v2 batches.  Missing batches may be imported with
``--allow-partial`` for exploratory diagnostics; missing records are never
filled with score zero or an inferred judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import make_gemini_prompts_v2 as renderer


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKET = ROOT / "judge_packet.json"
DEFAULT_INSTRUCTIONS = ROOT / "judge_instructions.txt"
DEFAULT_MANIFEST = ROOT / "gemini_copy_paste_v2" / "manifest.json"
ADAPTED_SCHEMA = "gemini-batch-adapted-v1"
ASSEMBLED_SCHEMA = "gemini-bundle-assembled-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
PLACEHOLDER = re.compile(r"^<[^<>]+>$")
FORBIDDEN_KEYS = {
    "api_key",
    "authorization",
    "bearer_token",
    "candidate_mapping",
    "model_mapping",
    "expected_answer",
    "old_score",
    "prior_score",
}


class GeminiImportError(ValueError):
    """Raised when a raw capture cannot be losslessly imported."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _read_json(path: str | Path, label: str) -> Any:
    file = Path(path)
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeminiImportError(f"{label} is not strict UTF-8 JSON: {file}: {exc}") from exc


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeminiImportError(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GeminiImportError(f"{label} must be a JSON array")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeminiImportError(f"{label} must be a non-empty string")
    return value


def _placeholder(value: Any) -> bool:
    return value is None or (isinstance(value, str) and (not value.strip() or PLACEHOLDER.fullmatch(value.strip()) is not None))


def _forbidden_metadata(value: Any, path: str = "envelope") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise GeminiImportError(f"forbidden private/secret field {path}.{key}")
            _forbidden_metadata(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_metadata(child, f"{path}[{index}]")


def _operator_field(raw: Mapping[str, Any], key: str, expected: str, label: str) -> str:
    supplied = raw.get(key)
    if _placeholder(supplied):
        return expected
    if not isinstance(supplied, str) or not supplied.strip():
        raise GeminiImportError(f"{label} must be a string or the prompt placeholder")
    if supplied != expected:
        raise GeminiImportError(f"{label} conflicts with operator/canonical value: {supplied!r} != {expected!r}")
    return expected


def _load_context(
    manifest_path: str | Path,
    packet_path: str | Path,
    instructions_path: str | Path,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    packet_file = Path(packet_path)
    instructions_file = Path(instructions_path)
    manifest = _object(_read_json(manifest_file, "Gemini v2 manifest"), "Gemini v2 manifest")
    packet = _object(_read_json(packet_file, "canonical packet"), "canonical packet")
    packet_id, packet_hash, instructions_hash = renderer._packet_basics(packet, packet_file, instructions_file)
    if manifest.get("schema_version") != renderer.MANIFEST_SCHEMA:
        raise GeminiImportError("unsupported Gemini v2 manifest schema")
    if manifest.get("packet_id") != packet_id or manifest.get("packet_sha256") != packet_hash:
        raise GeminiImportError("manifest packet identity/hash does not match canonical packet")
    if manifest.get("instructions_sha256") != instructions_hash:
        raise GeminiImportError("manifest instructions hash does not match frozen instructions")
    instructions_text = instructions_file.read_text(encoding="utf-8")
    try:
        renderer.validate_rendered_manifest(manifest_file.parent, manifest, instructions_text)
    except renderer.PromptRenderError as exc:
        raise GeminiImportError(f"rendered v2 prompt manifest validation failed: {exc}") from exc
    specs = renderer.batch_specs(packet)
    specs_by_id = {str(spec["batch_id"]): spec for spec in specs}
    manifest_batches = manifest.get("batches")
    if not isinstance(manifest_batches, list):
        raise GeminiImportError("manifest batches must be an array")
    manifest_by_id = {str(batch.get("batch_id")): batch for batch in manifest_batches if isinstance(batch, Mapping)}
    if set(manifest_by_id) != set(specs_by_id):
        raise GeminiImportError("manifest batch IDs do not match the frozen packet")
    for batch_id, spec in specs_by_id.items():
        entry = manifest_by_id[batch_id]
        if entry.get("track") != spec["track"] or entry.get("expected") != spec["expected"]:
            raise GeminiImportError(f"manifest expected schema mismatch for {batch_id}")
        if entry.get("task_data_sha256") != renderer.canonical_json_sha256(spec["task_payload"]):
            raise GeminiImportError(f"manifest task data hash mismatch for {batch_id}")
    return {
        "manifest_path": manifest_file.resolve(),
        "manifest_sha256": sha256_file(manifest_file),
        "manifest": manifest,
        "packet": packet,
        "packet_id": packet_id,
        "packet_sha256": packet_hash,
        "instructions_sha256": instructions_hash,
        "specs_by_id": specs_by_id,
        "manifest_by_id": manifest_by_id,
    }


def _canonical_source(spec: Mapping[str, Any], record: Mapping[str, Any], label: str) -> tuple[str, str]:
    payload = spec["task_payload"]
    if spec["track"] == "qualification":
        item_id = str(record.get("qualification_id"))
        item = next((item for item in payload["qualification_items"] if item.get("qualification_id") == item_id), None)
        if item is None:
            raise GeminiImportError(f"{label} has unknown qualification_id {item_id!r}")
        expected_source_id = str(item["source_id"])
        expected_hash = str(item["source_text_sha256"])
        source = str(item["source_en"])
    elif spec["track"] == "pointwise":
        item_id = str(record.get("item_id"))
        if item_id != str(payload["item_id"]):
            raise GeminiImportError(f"{label} has unexpected item_id {item_id!r}")
        expected_source_id = str(payload["source_id"])
        expected_hash = str(payload["source_text_sha256"])
        source = str(payload["source_en"])
    else:
        item_id = str(record.get("item_id"))
        task = next((task for task in payload["pairwise_tasks"] if task.get("pair_id") == record.get("pair_id")), None)
        if task is None:
            raise GeminiImportError(f"{label} has unknown pair_id {record.get('pair_id')!r}")
        if item_id != str(task["item_id"]):
            raise GeminiImportError(f"{label} item_id conflicts with pair task")
        expected_source_id = str(task["source_id"])
        expected_hash = str(task["source_text_sha256"])
        source = str(task["source_en"])
    if record.get("source_id") != expected_source_id:
        raise GeminiImportError(f"{label} source_id does not match canonical packet")
    if record.get("source_text_sha256") != expected_hash:
        raise GeminiImportError(f"{label} source_text_sha256 does not match canonical packet")
    return source, item_id


def _target_for_record(spec: Mapping[str, Any], record: Mapping[str, Any], variant: Mapping[str, Any] | None = None) -> str | None:
    if spec["track"] == "pointwise":
        item = spec["task_payload"]
        candidate_id = record.get("blind_id")
        candidate = next((candidate for candidate in item["candidates"] if candidate.get("blind_id") == candidate_id), None)
        return str(candidate["target_zh_tw"]) if isinstance(candidate, Mapping) else None
    if spec["track"] == "qualification":
        item_id = record.get("qualification_id")
        item = next((item for item in spec["task_payload"]["qualification_items"] if item.get("qualification_id") == item_id), None)
        if item is None or variant is None:
            return None
        target_id = variant.get("blind_id")
        candidate = next((candidate for candidate in item["variants"] if candidate.get("blind_id") == target_id), None)
        return str(candidate["target_zh_tw"]) if isinstance(candidate, Mapping) else None
    pair = next((task for task in spec["task_payload"]["pairwise_tasks"] if task.get("pair_id") == record.get("pair_id")), None)
    if pair is None:
        return None
    candidate_a = pair.get("candidate_a")
    candidate_b = pair.get("candidate_b")
    targets = []
    for candidate in (candidate_a, candidate_b):
        if isinstance(candidate, Mapping) and isinstance(candidate.get("target_zh_tw"), str):
            targets.append(candidate["target_zh_tw"])
    return "\n".join(targets) if targets else None


def _validate_errors(errors: Any, source: str, target: str | None, label: str) -> list[dict[str, Any]]:
    """Validate hard transport fields and return non-binding evidence audit.

    ``target_exact_quote`` is a hard binding because a non-literal quote can
    attach an error to the wrong blind target.  Source spans and target
    anchors are evidence diagnostics: the frozen protocol permits a missing
    or near-exact source explanation, so those conditions are recorded as
    unavailable/invalid evidence rather than quarantining an otherwise
    structurally valid judgment.  No input field is repaired.
    """
    if not isinstance(errors, list):
        raise GeminiImportError(f"{label}.errors must be an array of error objects")
    evidence: list[dict[str, Any]] = []
    for index, error in enumerate(errors):
        error_label = f"{label}.errors[{index}]"
        if not isinstance(error, Mapping):
            raise GeminiImportError(f"{error_label} must be an object; string error text cannot be losslessly imported")
        quote = error.get("target_exact_quote")
        if not isinstance(quote, str):
            raise GeminiImportError(f"{error_label}.target_exact_quote must be a string")
        if quote and (target is None or quote not in target):
            raise GeminiImportError(f"{error_label}.target_exact_quote is not an exact canonical target substring")
        source_span = error.get("source_span")
        anchor = error.get("target_anchor")
        source_span_status = "not_provided"
        if source_span is not None and not isinstance(source_span, str):
            source_span_status = "invalid_type"
        elif isinstance(source_span, str) and not source_span:
            source_span_status = "unavailable"
        elif isinstance(source_span, str) and source_span not in source:
            source_span_status = "invalid_nonliteral"
        elif isinstance(source_span, str):
            source_span_status = "valid_exact"
        anchor_status = "not_required" if quote else "unavailable"
        if anchor is not None and not isinstance(anchor, str):
            anchor_status = "invalid_type"
        elif isinstance(anchor, str) and not anchor:
            anchor_status = "unavailable" if not quote else "not_provided"
        elif isinstance(anchor, str) and (target is None or anchor not in target):
            anchor_status = "invalid_nonliteral"
        elif isinstance(anchor, str):
            anchor_status = "valid_exact"
        if not quote and source_span_status == "not_provided":
            source_span_status = "unavailable"
        evidence_status = "valid" if source_span_status in {"valid_exact", "not_required"} and anchor_status in {"valid_exact", "not_required"} else "unavailable"
        if source_span_status.startswith("invalid") or anchor_status.startswith("invalid"):
            evidence_status = "invalid"
        evidence.append({
            "error_index": index,
            "target_quote_status": "valid_exact" if quote else "omission",
            "source_span_status": source_span_status,
            "target_anchor_status": anchor_status,
            "status": evidence_status,
        })
    return evidence


def _validate_common_record(spec: Mapping[str, Any], record: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    source, _ = _canonical_source(spec, record, label)
    if "track" in record and record["track"] != spec["track"]:
        raise GeminiImportError(f"{label}.track conflicts with batch track")
    target: str | None = None
    evidence: list[dict[str, Any]] = []
    if spec["track"] == "pointwise":
        target = _target_for_record(spec, record)
        if target is None:
            raise GeminiImportError(f"{label} blind_id is not a canonical pointwise candidate")
        if "errors" not in record:
            raise GeminiImportError(f"{label}.errors is missing; missingness must remain explicit")
        evidence = _validate_errors(record["errors"], source, target, label)
    elif spec["track"] == "pairwise":
        if record.get("orientation") != spec["expected"]["orientation"]:
            raise GeminiImportError(f"{label}.orientation does not match batch orientation")
        task = next(task for task in spec["task_payload"]["pairwise_tasks"] if task["pair_id"] == record["pair_id"])
        for key in ("candidate_a", "candidate_b"):
            expected = task[key]
            if not isinstance(expected, Mapping) or record.get(key) != expected.get("blind_id"):
                raise GeminiImportError(f"{label}.{key} does not match canonical blind candidate ID")
        if "errors" in record:
            evidence = _validate_errors(record["errors"], source, target, label)
    return evidence


def _record_key(spec: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    if spec["track"] == "qualification":
        return str(record.get("qualification_id"))
    if spec["track"] == "pointwise":
        return f"{record.get('item_id')}/{record.get('blind_id')}"
    return str(record.get("pair_id"))


def _validate_qualification_record(spec: Mapping[str, Any], record: Mapping[str, Any], label: str) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    source, _ = _canonical_source(spec, record, label)
    if "track" in record and record["track"] != "qualification":
        raise GeminiImportError(f"{label}.track conflicts with batch track")
    item = next(item for item in spec["task_payload"]["qualification_items"] if item["qualification_id"] == record["qualification_id"])
    variants = record.get("judgments")
    if not isinstance(variants, list):
        raise GeminiImportError(f"{label}.judgments must be an array")
    expected = {str(variant["blind_id"]): variant for variant in item["variants"]}
    seen: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        variant_label = f"{label}.judgments[{index}]"
        variant_map = _object(variant, variant_label)
        blind_id = _nonempty_string(variant_map.get("blind_id"), f"{variant_label}.blind_id")
        if blind_id not in expected:
            raise GeminiImportError(f"{variant_label} has unknown blind_id {blind_id!r}")
        if blind_id in seen:
            raise GeminiImportError(f"duplicate qualification judgment {record['qualification_id']}/{blind_id}")
        seen.add(blind_id)
        target = _target_for_record(spec, record, variant_map)
        if target is None:
            raise GeminiImportError(f"{variant_label} does not map to a canonical target")
        if "errors" not in variant_map:
            raise GeminiImportError(f"{variant_label}.errors is missing; missingness must remain explicit")
        for entry in _validate_errors(variant_map["errors"], source, target, variant_label):
            evidence.append({"judgment": blind_id, **entry})
    missing = sorted(set(expected) - seen)
    return [str(record["qualification_id"])], [f"{record['qualification_id']}/{blind_id}" for blind_id in missing], evidence


def _validate_records(context: Mapping[str, Any], spec: Mapping[str, Any], records: list[Any], allow_partial: bool) -> dict[str, Any]:
    expected_keys = [str(key) for key in spec["expected"]["record_keys"]]
    expected_set = set(expected_keys)
    seen: set[str] = set()
    observed_judgments: set[str] = set()
    missing_judgments: set[str] = set()
    evidence_audit: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records):
        label = f"{spec['batch_id']}.records[{index}]"
        record = _object(raw_record, label)
        key = _record_key(spec, record)
        if key not in expected_set:
            raise GeminiImportError(f"{label} has unknown record key {key!r}")
        if key in seen:
            raise GeminiImportError(f"duplicate record key {key!r} in {spec['batch_id']}")
        seen.add(key)
        if spec["track"] == "qualification":
            _, missing, evidence = _validate_qualification_record(spec, record, label)
            evidence_audit.extend({"record_key": key, **entry} for entry in evidence)
            item_id = key
            variants = record.get("judgments")
            if isinstance(variants, list):
                observed_judgments.update(f"{item_id}/{variant.get('blind_id')}" for variant in variants if isinstance(variant, Mapping))
            missing_judgments.update(missing)
        else:
            for entry in _validate_common_record(spec, record, label):
                evidence_audit.append({"record_key": key, **entry})
    missing_records = [key for key in expected_keys if key not in seen]
    if (missing_records or missing_judgments) and not allow_partial:
        raise GeminiImportError(f"{spec['batch_id']} is incomplete; missing records={missing_records}, judgments={sorted(missing_judgments)}")
    if spec["track"] == "qualification":
        expected_judgments = set(spec["expected"]["judgment_keys"])
        missing_judgments.update(expected_judgments - observed_judgments)
    evidence_invalid = sum(1 for entry in evidence_audit if entry["status"] == "invalid")
    evidence_unavailable = sum(1 for entry in evidence_audit if entry["status"] == "unavailable")
    return {
        "status": "partial_exploratory" if missing_records or missing_judgments else "complete",
        "expected_record_count": len(expected_keys),
        "observed_record_count": len(seen),
        "missing_record_keys": missing_records,
        "expected_judgment_count": len(spec["expected"].get("judgment_keys", [])),
        "observed_judgment_count": len(observed_judgments),
        "missing_judgment_keys": sorted(missing_judgments),
        "evidence_audit": {
            "status": "not_applicable" if not evidence_audit else ("partial" if evidence_invalid or evidence_unavailable else "complete"),
            "n_errors": len(evidence_audit),
            "n_invalid": evidence_invalid,
            "n_unavailable": evidence_unavailable,
            "entries": evidence_audit,
        },
    }


def adapt_capture(
    manifest_path: str | Path,
    packet_path: str | Path,
    instructions_path: str | Path,
    raw_path: str | Path,
    batch_id: str,
    judge_id: str,
    model_label_from_ui: str,
    run_id: str,
    output_path: str | Path,
    *,
    family: str = "google-gemini",
    allow_partial: bool = False,
) -> dict[str, Any]:
    context = _load_context(manifest_path, packet_path, instructions_path)
    specs_by_id = context["specs_by_id"]
    if batch_id not in specs_by_id:
        raise GeminiImportError(f"unknown v2 batch ID {batch_id!r}")
    for value, label in ((judge_id, "judge_id"), (model_label_from_ui, "model_label_from_ui"), (run_id, "run_id"), (family, "family")):
        _nonempty_string(value, label)
    raw_file = Path(raw_path)
    raw = _object(_read_json(raw_file, "raw Gemini capture"), "raw Gemini capture")
    _forbidden_metadata(raw)
    spec = specs_by_id[batch_id]
    packet_id = _nonempty_string(context["packet_id"], "packet_id")
    packet_hash = context["packet_sha256"]
    if raw.get("packet_id") != packet_id:
        raise GeminiImportError("raw capture packet_id must match frozen packet; packet fields are not operator placeholders")
    if raw.get("packet_sha256") != packet_hash:
        raise GeminiImportError("raw capture packet_sha256 must match frozen packet; packet fields are not operator placeholders")
    if raw.get("track") != spec["track"]:
        raise GeminiImportError("raw capture track must match canonical batch")
    _operator_field(raw, "task_id", str(spec["task_id"]), "task_id")
    _operator_field(raw, "batch_id", batch_id, "batch_id")
    _operator_field(raw, "judge_id", judge_id, "judge_id")
    _operator_field(raw, "model_label_from_ui", model_label_from_ui, "model_label_from_ui")
    records = _list(raw.get("records"), "raw capture records")
    coverage = _validate_records(context, spec, records, allow_partial)
    if coverage["status"] == "partial_exploratory" and not allow_partial:
        raise GeminiImportError("partial capture requires --allow-partial")
    source_record_hashes = [canonical_sha256(record) for record in records]
    adapted = {
        "schema_version": ADAPTED_SCHEMA,
        "packet_id": packet_id,
        "packet_sha256": packet_hash,
        "judge_id": judge_id,
        "model_label_from_ui": model_label_from_ui,
        "task_id": str(spec["task_id"]),
        "track": spec["track"],
        "batch_id": batch_id,
        "run_id": run_id,
        "judge": {
            "judge_id": judge_id,
            "family": family,
            "provider": "gemini_ui",
            "model_label_from_ui": model_label_from_ui,
            "ui_model_version": model_label_from_ui,
        },
        "records": records,
        "coverage": coverage,
        "provenance": {
            "raw_capture_path": str(raw_file.resolve()),
            "raw_capture_sha256": sha256_file(raw_file),
            "raw_capture_bytes": raw_file.stat().st_size,
            "manifest_path": str(context["manifest_path"]),
            "manifest_sha256": context["manifest_sha256"],
            "prompt_filename": context["manifest_by_id"][batch_id]["filename"],
            "prompt_sha256": next(item["sha256"] for item in context["manifest"]["files"] if item["batch_id"] == batch_id),
            "render_version": context["manifest"]["render_version"],
            "instructions_sha256": context["instructions_sha256"],
            "normalization": ["filled operator envelope metadata only"],
            "judgment_payloads_unchanged": True,
            "record_canonical_sha256": source_record_hashes,
            "allow_partial": bool(allow_partial),
        },
    }
    output = Path(output_path)
    if output.exists():
        raise GeminiImportError(f"refusing to overwrite existing adapted output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(adapted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return adapted


def _analysis_source_alias(spec: Mapping[str, Any], record: Mapping[str, Any], label: str) -> tuple[str, dict[str, Any] | None]:
    """Return the analysis cluster ID without changing raw capture metadata.

    The frozen packet exposes pointwise source labels as ``source-01`` while
    the historical runner bundles use ``src-01`` (the pointwise item ID) as
    the analysis cluster.  Bind the item and source hash first (the adapter
    has already checked the raw source label), then apply only this explicit
    operator analysis alias in the assembled copy.  Qualification and
    pairwise rows already use the shared IDs and are left untouched.
    """

    raw_source_id = record.get("source_id")
    if not isinstance(raw_source_id, str) or not raw_source_id:
        raise GeminiImportError(f"{label}.source_id is missing before analysis aliasing")
    if spec["track"] != "pointwise":
        return raw_source_id, None
    item_id = record.get("item_id")
    payload = spec["task_payload"]
    canonical_analysis_id = str(item_id)
    if canonical_analysis_id != str(payload["item_id"]):
        raise GeminiImportError(f"{label} item/source alias does not match canonical pointwise item")
    if record.get("source_text_sha256") != payload["source_text_sha256"]:
        raise GeminiImportError(f"{label} source hash does not match before analysis aliasing")
    expected_packet_source_label = str(payload["source_id"])
    if raw_source_id != expected_packet_source_label and raw_source_id != canonical_analysis_id:
        raise GeminiImportError(
            f"{label} source_id {raw_source_id!r} is neither the frozen packet source label "
            f"{expected_packet_source_label!r} nor the known analysis ID {canonical_analysis_id!r}"
        )
    if raw_source_id == canonical_analysis_id:
        return canonical_analysis_id, None
    return canonical_analysis_id, {
        "kind": "operator_analysis_cluster_alias",
        "raw_source_id": raw_source_id,
        "analysis_source_id": canonical_analysis_id,
        "packet_source_label": expected_packet_source_label,
        "item_id": canonical_analysis_id,
        "source_text_sha256": payload["source_text_sha256"],
        "reason": "frozen packet source_label (source-NN) is aliased to shared historical analysis cluster src-NN=item_id",
    }


def _wrapped_record(record: Mapping[str, Any], batch: Mapping[str, Any], spec: Mapping[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    wrapped = dict(record)
    for key, value in (("track", batch["track"]), ("batch_id", batch["batch_id"]), ("task_id", batch["task_id"]), ("run_id", batch["run_id"])):
        if key in wrapped and wrapped[key] != value:
            raise GeminiImportError(f"assembled record {key} conflicts with adapted batch metadata")
        wrapped.setdefault(key, value)
    analysis_source_id, alias = _analysis_source_alias(spec, record, label)
    if alias is not None:
        if "raw_source_id" in wrapped and wrapped["raw_source_id"] != alias["raw_source_id"]:
            raise GeminiImportError(f"{label}.raw_source_id conflicts with operator alias provenance")
        if "source_id_alias" in wrapped and wrapped["source_id_alias"] != alias:
            raise GeminiImportError(f"{label}.source_id_alias conflicts with operator alias provenance")
        wrapped["raw_source_id"] = alias["raw_source_id"]
        wrapped["source_id"] = analysis_source_id
        wrapped["source_id_alias"] = alias
    elif spec["track"] == "pointwise" and "source_id_alias" in wrapped:
        raise GeminiImportError(f"{label}.source_id_alias is not allowed when no alias is required")
    return wrapped, alias


def _verify_raw_provenance(bundle: Mapping[str, Any], input_path: str | Path) -> None:
    """Verify that an adapted bundle still corresponds to its untouched raw capture."""

    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("judgment_payloads_unchanged") is not True:
        raise GeminiImportError(f"{input_path}.provenance does not attest unchanged judgment payloads")
    raw_path_value = provenance.get("raw_capture_path")
    raw_path = Path(raw_path_value) if isinstance(raw_path_value, str) else None
    if raw_path is None or not raw_path.is_file():
        raise GeminiImportError(f"{input_path} raw_capture_path is missing or unavailable")
    expected_raw_sha = provenance.get("raw_capture_sha256")
    if not isinstance(expected_raw_sha, str) or not HEX64.fullmatch(expected_raw_sha):
        raise GeminiImportError(f"{input_path} raw_capture_sha256 is missing or malformed")
    actual_raw_sha = sha256_file(raw_path)
    if actual_raw_sha.lower() != expected_raw_sha.lower():
        raise GeminiImportError(f"{input_path} raw capture SHA-256 changed after adaptation")
    raw = _object(_read_json(raw_path, f"raw capture referenced by {input_path}"), "raw capture referenced by adapted bundle")
    raw_records = _list(raw.get("records"), f"raw capture referenced by {input_path}.records")
    adapted_records = _list(bundle.get("records"), f"{input_path}.records")
    expected_record_hashes = provenance.get("record_canonical_sha256")
    if not isinstance(expected_record_hashes, list) or len(expected_record_hashes) != len(raw_records):
        raise GeminiImportError(f"{input_path} record provenance hash list is missing or has the wrong length")
    raw_hashes = [canonical_sha256(record) for record in raw_records]
    adapted_hashes = [canonical_sha256(record) for record in adapted_records]
    if raw_hashes != expected_record_hashes or adapted_hashes != expected_record_hashes:
        raise GeminiImportError(f"{input_path} adapted records no longer match the untouched raw capture")


def assemble_bundles(
    manifest_path: str | Path,
    packet_path: str | Path,
    instructions_path: str | Path,
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    context = _load_context(manifest_path, packet_path, instructions_path)
    if not input_paths:
        raise GeminiImportError("assemble requires at least one adapted batch")
    bundles: list[Mapping[str, Any]] = []
    seen_batches: set[str] = set()
    seen_run_ids: set[str] = set()
    for input_path in input_paths:
        bundle = _object(_read_json(input_path, f"adapted Gemini bundle {input_path}"), "adapted Gemini bundle")
        if bundle.get("schema_version") != ADAPTED_SCHEMA:
            raise GeminiImportError(f"{input_path} is not an adapted Gemini v2 bundle")
        if bundle.get("packet_id") != context["packet_id"] or bundle.get("packet_sha256") != context["packet_sha256"]:
            raise GeminiImportError(f"{input_path} packet identity/hash mismatch")
        _verify_raw_provenance(bundle, input_path)
        batch_id = _nonempty_string(bundle.get("batch_id"), f"{input_path}.batch_id")
        if batch_id not in context["specs_by_id"]:
            raise GeminiImportError(f"{input_path} has unknown batch_id {batch_id!r}")
        if batch_id in seen_batches:
            raise GeminiImportError(f"duplicate adapted batch {batch_id}")
        seen_batches.add(batch_id)
        spec = context["specs_by_id"][batch_id]
        if bundle.get("track") != spec["track"] or bundle.get("task_id") != spec["task_id"]:
            raise GeminiImportError(f"{input_path} track/task_id does not match canonical batch")
        run_id = _nonempty_string(bundle.get("run_id"), f"{input_path}.run_id")
        if run_id in seen_run_ids:
            raise GeminiImportError(f"duplicate run_id {run_id!r}; each fresh Gemini chat needs distinct provenance")
        seen_run_ids.add(run_id)
        _nonempty_string(bundle.get("judge_id"), f"{input_path}.judge_id")
        _nonempty_string(bundle.get("model_label_from_ui"), f"{input_path}.model_label_from_ui")
        _nonempty_string(bundle.get("family"), f"{input_path}.family") if "family" in bundle else None
        if not isinstance(bundle.get("judge"), Mapping):
            raise GeminiImportError(f"{input_path}.judge metadata is missing")
        if bundle["judge"].get("provider") != "gemini_ui":
            raise GeminiImportError(f"{input_path} provider is not gemini_ui")
        if bundle["judge"].get("judge_id") != bundle.get("judge_id"):
            raise GeminiImportError(f"{input_path} judge metadata mismatch")
        if bundle["judge"].get("model_label_from_ui") != bundle.get("model_label_from_ui"):
            raise GeminiImportError(f"{input_path} UI model metadata mismatch")
        if not isinstance(bundle["judge"].get("family"), str) or not bundle["judge"].get("family", "").strip():
            raise GeminiImportError(f"{input_path} judge family metadata is missing")
        provenance_entry = bundle.get("provenance")
        if not isinstance(provenance_entry, Mapping):
            raise GeminiImportError(f"{input_path}.provenance is missing")
        if provenance_entry.get("manifest_sha256") != context["manifest_sha256"] or provenance_entry.get("instructions_sha256") != context["instructions_sha256"]:
            raise GeminiImportError(f"{input_path} prompt provenance does not match frozen v2 manifest")
        raw_records = _list(bundle.get("records"), f"{input_path}.records")
        recalculated_coverage = _validate_records(context, spec, raw_records, allow_partial=True)
        supplied_coverage = bundle.get("coverage")
        if not isinstance(supplied_coverage, Mapping) or supplied_coverage.get("status") != recalculated_coverage["status"]:
            raise GeminiImportError(f"{input_path}.coverage does not match its records")
        bundles.append(bundle)
    reference = bundles[0]
    reference_judge = reference["judge"]
    for bundle in bundles[1:]:
        for key in ("judge_id", "model_label_from_ui"):
            if bundle.get(key) != reference.get(key):
                raise GeminiImportError(f"adapted batches disagree on {key}; do not assemble as one judge")
        if bundle["judge"].get("family") != reference_judge.get("family"):
            raise GeminiImportError("adapted batches disagree on judge family")
    expected_batches = set(context["specs_by_id"])
    missing_batches = sorted(expected_batches - seen_batches)
    if missing_batches and not allow_partial:
        raise GeminiImportError(f"assembled Gemini bundle is missing batches: {missing_batches}")
    records: list[dict[str, Any]] = []
    provenance: list[Mapping[str, Any]] = []
    source_alias_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        batch_id = str(bundle["batch_id"])
        spec = context["specs_by_id"][batch_id]
        batch = {
            "batch_id": batch_id,
            "task_id": bundle["task_id"],
            "track": bundle["track"],
            "run_id": bundle["run_id"],
        }
        raw_records = _list(bundle.get("records"), f"{batch_id}.records")
        for index, raw_record in enumerate(raw_records):
            wrapped, alias = _wrapped_record(_object(raw_record, f"{batch_id}.records[{index}]"), batch, spec, f"{batch_id}.records[{index}]")
            records.append(wrapped)
            if alias is not None:
                source_alias_rows.append({
                    "track": batch["track"],
                    "batch_id": batch_id,
                    "record_index": index,
                    **alias,
                })
        if isinstance(bundle.get("provenance"), Mapping):
            provenance.append(bundle["provenance"])
    assembled = {
        "schema_version": ASSEMBLED_SCHEMA,
        "packet_id": context["packet_id"],
        "packet_sha256": context["packet_sha256"],
        "judge_id": reference["judge_id"],
        "model_label_from_ui": reference["model_label_from_ui"],
        "task_id": "gemini-pilot-v2",
        "judge": dict(reference_judge),
        "batch_ids": sorted(seen_batches),
        "records": records,
        "coverage": {
            "status": "partial_exploratory" if missing_batches else "complete",
            "expected_batch_count": len(expected_batches),
            "observed_batch_count": len(seen_batches),
            "missing_batches": missing_batches,
            "source_batches": sorted(seen_batches),
            "note": "A partial bundle is exploratory only; omitted tracks/records are not score zero and do not count as extra judges.",
        },
        "provenance": {
            "manifest_path": str(context["manifest_path"]),
            "manifest_sha256": context["manifest_sha256"],
            "render_version": context["manifest"]["render_version"],
            "instructions_sha256": context["instructions_sha256"],
            "adapted_inputs": provenance,
            "record_wrappers_added": ["track", "batch_id", "task_id", "run_id", "raw_source_id", "source_id_alias"],
            "source_id_analysis_normalization": {
                "status": "applied" if source_alias_rows else "not_needed",
                "scope": "assembled Gemini analysis copy only",
                "rule": "pointwise packet source_label source-NN -> shared analysis cluster src-NN=item_id after source hash binding; qualification/pairwise unchanged",
                "per_record": source_alias_rows,
            },
            "raw_adapted_record_hashes_verified": True,
            "score_error_fields_unchanged": True,
            "allow_partial": bool(allow_partial),
        },
    }
    output = Path(output_path)
    if output.exists():
        raise GeminiImportError(f"refusing to overwrite existing assembled output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assembled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return assembled


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapt = subparsers.add_parser("adapt", help="validate one raw Gemini capture and add operator provenance")
    adapt.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    adapt.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    adapt.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    adapt.add_argument("--input", type=Path, required=True)
    adapt.add_argument("--batch-id", required=True)
    adapt.add_argument("--judge-id", required=True)
    adapt.add_argument("--model-label-from-ui", required=True)
    adapt.add_argument("--run-id", required=True)
    adapt.add_argument("--family", default="google-gemini")
    adapt.add_argument("--allow-partial", action="store_true")
    adapt.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble", help="assemble validated adapted v2 batches")
    assemble.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    assemble.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    assemble.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    assemble.add_argument("--input", type=Path, action="append", required=True)
    assemble.add_argument("--allow-partial", action="store_true")
    assemble.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "adapt":
            result = adapt_capture(
                args.manifest,
                args.packet,
                args.instructions,
                args.input,
                args.batch_id,
                args.judge_id,
                args.model_label_from_ui,
                args.run_id,
                args.output,
                family=args.family,
                allow_partial=args.allow_partial,
            )
        else:
            result = assemble_bundles(
                args.manifest,
                args.packet,
                args.instructions,
                args.input,
                args.output,
                allow_partial=args.allow_partial,
            )
    except GeminiImportError as exc:
        print(f"FAIL: {exc}")
        return 2
    print(json.dumps({"status": "PASS", "schema_version": result["schema_version"], "output": str(args.output), "coverage": result["coverage"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
