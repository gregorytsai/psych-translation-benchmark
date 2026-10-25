#!/usr/bin/env python3
"""Approval-gated fresh-context runner for the new-data cloud judges.

This runner intentionally owns only the ``judge_evaluation/cloud_runner`` and
``raw/cloud`` artifacts.  It reads public packet files only and never opens
the private blind map.  GPT requests are executed as one ephemeral Codex CLI
process per task/candidate, with a fresh scratch directory and an exact
structured-output schema.  Gemini is exposed through the same packet/checkpoint
interface; a native UI capture adapter can import one completed response at a
time without putting private mapping data in the UI prompt.

The runner is deliberately conservative:

* no request is sent unless ``packet_ready.json`` and an explicit approval
  marker are present and agree with the packet/manifest hashes;
* the approval marker is a file written by the root owner, never inferred from
  a timestamp or an arbitrary truthy field;
* one task/candidate is sent per fresh context (no candidate batch prompting);
* image tasks use the real public PNG attachment, never OCR text or image
  ground truth; and
* every raw response, CLI event stream, route identity and checkpoint is kept
  under ``raw/cloud`` with missing records represented as missing, not zero.

The public packet builder may evolve field names.  ``load_public_packet``
accepts the stable packet shapes used by the benchmark (a single packet JSON,
or pointwise/pairwise directories of JSON task files) while rejecting anything
that exposes model names or private paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parent
PUBLIC_DIR = EVAL_DIR / "public"
RAW_DIR = EVAL_DIR / "raw" / "cloud"
DEFAULT_MANIFEST = EVAL_DIR / "manifest.json"
DEFAULT_PACKET_READY = EVAL_DIR / "packet_ready.json"
DEFAULT_PROGRESS = EVAL_DIR / "progress_cloud.md"
GPT_MODEL = "gpt-6.1-sol"
GPT_JUDGE_ID = "gpt-6.1-sol-codex-new"
GEMINI_JUDGE_ID = "gemini-3.8-flash-high-antigravity-new"


class RunnerError(RuntimeError):
    pass


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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunnerError(f"required public file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def redact(value: Any) -> Any:
    """Redact credential-shaped metadata while retaining route evidence."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            low = str(k).lower()
            if low in {"token", "access_token", "refresh_token", "id_token"} or any(x in low for x in ("api_key", "apikey", "authorization", "bearer", "password", "secret")):
                out[str(k)] = "<REDACTED>"
            else:
                out[str(k)] = redact(v)
        return out
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <REDACTED>", value)
        value = re.sub(r"(?i)\bsk-[A-Za-z0-9._-]{12,}", "sk-<REDACTED>", value)
        return re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+", r"\1=<REDACTED>", value)
    return value


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def public_only_check(value: Any, *, where: str = "packet") -> None:
    """Reject obvious private/model/path leakage in public task material."""

    banned_keys = {"answer_key", "mapping", "model", "model_name", "model_alias", "private", "raw_file", "raw_path", "ground_truth", "scores", "prior_score"}
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in banned_keys:
                raise RunnerError(f"public packet contains private/provenance field {where}.{key}")
            public_only_check(child, where=f"{where}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            public_only_check(child, where=f"{where}[{i}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in ("candidate_mapping.private", "answer_key.private", "/private/", "/raw/", "models.ini", "litellm")):
            raise RunnerError(f"public packet contains private/path reference at {where}")


def find_packet_file(root: Path) -> Path | None:
    for name in ("packet.json", "judge_packet.json", "public_packet.json"):
        p = root / name
        if p.exists():
            return p
    return None


def _normalise_item(item: dict[str, Any], *, track: str, path: Path | None = None) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RunnerError(f"{track} task is not an object: {path or '<inline>'}")
    out = dict(item)
    out.setdefault("track", track)
    input_type = str(item.get("input_type", "text")).lower()
    if track == "pointwise":
        required = ("item_id", "candidates")
    elif track == "pairwise":
        required = ("pair_id", "item_id", "orientation", "candidate_a", "candidate_b")
    else:
        required = ("qualification_id", "variants")
    missing = [key for key in required if key not in out]
    if input_type != "image" and "source_en" not in out:
        missing.append("source_en")
    if missing:
        raise RunnerError(f"{track} task missing {missing}: {path or '<inline>'}")
    return out


def _canonical_job(job: dict[str, Any], *, track: str, path: Path | None = None) -> dict[str, Any]:
    """Adapt one public JSONL worker job to the runner's canonical shape."""

    if track == "pointwise":
        if "job_id" not in job or "blind_id" not in job or "target_zh_tw" not in job:
            raise RunnerError(f"pointwise job missing job_id/blind_id/target_zh_tw: {path or '<inline>'}")
        out = dict(job)
        out["item_id"] = str(job["job_id"])
        out["input_type"] = "image" if str(job.get("source_kind", "text")).lower() == "image" else "text"
        out["candidates"] = [{"blind_id": job["blind_id"], "target_zh_tw": job["target_zh_tw"], **({"image_alias": job["image_alias"]} if job.get("image_alias") else {})}]
        return _normalise_item(out, track=track, path=path)
    if track == "pairwise":
        if "pair_job_id" not in job or "candidate_a" not in job or "candidate_b" not in job:
            raise RunnerError(f"pairwise job missing pair_job_id/candidates: {path or '<inline>'}")
        out = dict(job)
        out["pair_id"] = str(job["pair_job_id"])
        out["item_id"] = str(job.get("pair_key", job["pair_job_id"]))
        out["input_type"] = "image" if str(job.get("source_kind", "text")).lower() == "image" else "text"
        return _normalise_item(out, track=track, path=path)
    return _normalise_item(job, track=track, path=path)


def _read_jsonl_jobs(path: Path, *, track: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise RunnerError(f"public {track} job file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(job, dict):
            raise RunnerError(f"public job is not an object at {path}:{line_no}")
        public_only_check(job, where=f"{path}:{line_no}")
        rows.append(_canonical_job(job, track=track, path=path))
    return rows


def _read_task_dir(path: Path, *, track: str, slot: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if slot:
        return _read_jsonl_jobs(path / f"{slot}.jsonl", track=track)
    rows: list[dict[str, Any]] = []
    for p in sorted(path.glob("*.json")):
        loaded = load_json(p)
        if isinstance(loaded, list):
            rows.extend(_normalise_item(v, track=track, path=p) for v in loaded)
        elif isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
            rows.extend(_normalise_item(v, track=track, path=p) for v in loaded["items"])
        elif isinstance(loaded, dict) and isinstance(loaded.get("tasks"), list):
            rows.extend(_normalise_item(v, track=track, path=p) for v in loaded["tasks"])
        elif isinstance(loaded, dict):
            rows.append(_normalise_item(loaded, track=track, path=p))
        else:
            raise RunnerError(f"unsupported {track} task file: {p}")
    return rows


def load_public_packet(public_dir: Path, manifest_path: Path, slot: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load packet + manifest strictly from public paths."""

    manifest = load_json(manifest_path)
    public_only_check(manifest, where="manifest")
    packet_file = find_packet_file(public_dir)
    packet: dict[str, Any]
    if packet_file:
        loaded = load_json(packet_file)
        if not isinstance(loaded, dict):
            raise RunnerError(f"public packet must be an object: {packet_file}")
        public_only_check(loaded, where="packet")
        packet = dict(loaded)
    else:
        packet = {"packet_id": manifest.get("packet_id")}
        for track, dirname, key in (("pointwise", "pointwise", "items"), ("pairwise", "pairwise", "tasks"), ("qualification", "qualification", "items")):
            rows = _read_task_dir(public_dir / dirname, track=track, slot=slot)
            if rows:
                packet[track] = {key: rows}
    for track, key in (("pointwise", "items"), ("pairwise", "tasks"), ("qualification", "items")):
        packet.setdefault(track, {})
        section = packet.get(track, {})
        if isinstance(section, list):
            rows = section
            packet[track] = {key: rows}
        elif isinstance(section, dict):
            rows = section.get(key)
            if rows is None:
                rows = section.get("tasks" if track == "pairwise" else "items", [])
            if not isinstance(rows, list):
                raise RunnerError(f"public {track} section is malformed")
            packet[track][key] = [_normalise_item(v, track=track) for v in rows]
        else:
            packet[track] = {key: []}
    if not packet.get("packet_id"):
        packet["packet_id"] = manifest.get("packet_id")
    if not packet.get("packet_id"):
        raise RunnerError("packet_id is missing")
    return packet, manifest


def verify_ready(manifest_path: Path, ready_path: Path, approval_path: Path) -> dict[str, Any]:
    """Require exact explicit root approval before any provider call."""

    manifest = load_json(manifest_path)
    ready = load_json(ready_path)
    if not isinstance(ready, dict):
        raise RunnerError("packet_ready.json must be an object")
    if ready.get("ready") is not True and ready.get("packet_ready") is not True and ready.get("status") not in {"READY", "ready", "PASS"}:
        raise RunnerError("packet_ready.json does not assert readiness")
    if not approval_path.exists():
        raise RunnerError(f"root approval marker is missing: {approval_path}")
    marker = approval_path.read_text(encoding="utf-8").strip()
    marker_ok = marker.lower() in {"root_approved", "root-approved", "approved", "true", "ready"}
    if not marker_ok:
        try:
            marker_obj = json.loads(marker)
        except json.JSONDecodeError:
            marker_obj = None
        marker_ok = isinstance(marker_obj, dict) and marker_obj.get("root_approved") is True
    if not marker_ok:
        raise RunnerError(f"root approval marker has unexpected content: {approval_path}")
    expected_packet = ready.get("packet_sha256") or ready.get("public_packet_sha256") or manifest.get("packet_sha256")
    if expected_packet and ready.get("packet_sha256") and expected_packet != ready.get("packet_sha256"):
        raise RunnerError("packet_ready packet hash fields disagree")
    expected_manifest = ready.get("manifest_sha256")
    actual_manifest = sha256_file(manifest_path)
    if expected_manifest and expected_manifest != actual_manifest:
        raise RunnerError("packet_ready manifest_sha256 does not match manifest.json")
    # Verify every frozen public payload hash from the manifest before the
    # first score call.  This never opens private/ entries.
    public_hashes = manifest.get("public_file_sha256", {})
    if not isinstance(public_hashes, dict) or not public_hashes:
        raise RunnerError("manifest has no public_file_sha256 inventory")
    checked_public = 0
    for relative, expected in public_hashes.items():
        relative = str(relative)
        if not relative.startswith("public/"):
            continue
        public_path = manifest_path.parent / relative
        if not public_path.is_file():
            raise RunnerError(f"frozen public payload is missing: {relative}")
        actual = sha256_file(public_path)
        if actual != expected:
            raise RunnerError(f"frozen public payload hash mismatch: {relative}")
        checked_public += 1
    if checked_public == 0:
        raise RunnerError("manifest public_file_sha256 contains no public entries")
    return {"ready": ready, "approval_marker": str(approval_path), "approval_text": marker, "manifest_sha256": actual_manifest, "public_files_verified": checked_public}


def resolve_instruction(public_dir: Path, packet: dict[str, Any], explicit: Path | None) -> tuple[str, str]:
    if explicit:
        text = explicit.read_text(encoding="utf-8")
        return text, sha256_bytes(text.encode("utf-8"))
    for name in ("judge_instructions.txt", "instructions.txt", "instructions.md"):
        path = public_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return text, sha256_bytes(text.encode("utf-8"))
    for key in ("instructions", "judge_instructions"):
        if isinstance(packet.get(key), str):
            text = packet[key]
            return text, sha256_bytes(text.encode("utf-8"))
    raise RunnerError("public judge instructions not found")


def extract_json(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value).strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for i, char in enumerate(value):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(value[i:])
                return parsed
            except json.JSONDecodeError:
                continue
    raise RunnerError("provider output contains no decodable JSON")


def load_schema(public_dir: Path, track: str) -> Path:
    """Return the frozen public JSON schema path.

    The public schema is the contract used for post-validation.  The schema
    passed to Codex is derived separately because the OpenAI strict-schema
    subset does not accept the pointwise schema's conditional ``allOf``.
    """

    candidates = [
        public_dir / "schemas" / f"{track}.json",
        public_dir / "schemas" / f"{track}_schema.json",
        public_dir / "schemas" / f"{track}_result.schema.json",
        public_dir / f"{track}.schema.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise RunnerError(f"missing public strict output schema for {track}; expected one of {', '.join(str(x) for x in candidates)}")


def derive_transport_schema(public_schema_path: Path, track: str, raw_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Derive a strict transport schema without modifying the public schema.

    Codex's strict structured-output transport needs every property required
    and does not accept the pointwise schema's conditional ``allOf``.  The
    derived pointwise schema therefore makes ``block_reason`` required and
    represents score/subscores as nullable fields.  The frozen public schema
    remains the authoritative contract and is checked again after parsing.
    """

    public_schema = load_json(public_schema_path)
    if not isinstance(public_schema, dict):
        raise RunnerError(f"public schema must be an object: {public_schema_path}")
    if track == "pairwise":
        # Pairwise is already a strict flat schema.  Copy only the supported
        # object/field contract so metadata such as $id/$schema cannot cause a
        # provider-side schema rejection.
        derived: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "TIE"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "critical_override": {"type": "boolean"},
                "reason_codes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
            },
            "required": ["winner", "confidence", "critical_override", "reason_codes"],
        }
    elif track == "pointwise":
        errors_schema = public_schema.get("properties", {}).get("errors", {}).get("items")
        if not isinstance(errors_schema, dict):
            raise RunnerError("frozen pointwise schema is missing its error-item schema")
        # Keep the frozen error-item constraints; this part of the public
        # schema is already strict and has no conditional branches.
        subscores = {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "terminology_locale": {"type": "integer", "minimum": 0, "maximum": 30},
                "semantic_safety": {"type": "integer", "minimum": 0, "maximum": 30},
                "fluency": {"type": "integer", "minimum": 0, "maximum": 20},
                "completeness_format": {"type": "integer", "minimum": 0, "maximum": 20},
            },
            "required": ["terminology_locale", "semantic_safety", "fluency", "completeness_format"],
        }
        derived = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "blind_id": {"type": "string", "pattern": "^b-[0-9a-f]{12}$"},
                "status": {"type": "string", "enum": ["ok", "blocked", "missing"]},
                "abstain": {"type": "boolean"},
                "score_0_100": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
                "subscores": subscores,
                "critical": {"type": "boolean"},
                "errors": {"type": "array", "items": errors_schema},
                # Required for strict transport.  It must be "" for status
                # ok and non-empty for blocked/missing (validated below).
                "block_reason": {"type": "string"},
            },
            "required": [
                "blind_id", "status", "abstain", "score_0_100", "subscores",
                "critical", "errors", "block_reason",
            ],
        }
    else:
        raise RunnerError(f"unsupported transport schema track: {track}")
    derived_sha = sha256_json(derived)
    public_sha = sha256_file(public_schema_path)
    out_path = raw_dir / "transport_schemas" / f"{track}.json"
    write_json(out_path, derived)
    metadata = {
        "track": track,
        "public_schema_path": str(public_schema_path),
        "public_schema_sha256": public_sha,
        "transport_schema_path": str(out_path),
        "transport_schema_sha256": derived_sha,
        "derivation": "flatten-pointwise-allOf; require-block_reason; nullable-score-subscores; preserve-error-items",
    }
    write_json(raw_dir / "transport_schemas" / f"{track}.metadata.json", metadata)
    return out_path, metadata


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_error(error: Any, target: str) -> tuple[list[str], list[str]]:
    structural: list[str] = []
    evidence: list[str] = []
    if not isinstance(error, dict):
        return ["error item is not an object"], []
    allowed = {"severity", "type", "target_exact_quote", "source_evidence", "source_span", "target_anchor", "confidence"}
    extra = set(error) - allowed
    missing = allowed - set(error)
    if extra:
        structural.append(f"error has extra fields: {sorted(extra)}")
    if missing:
        structural.append(f"error is missing fields: {sorted(missing)}")
    if error.get("severity") not in {"minor", "major", "critical"}:
        structural.append("error severity is invalid")
    for key in ("type", "target_exact_quote", "source_evidence", "source_span", "target_anchor"):
        if not isinstance(error.get(key), str):
            structural.append(f"error {key} is not a string")
    if isinstance(error.get("type"), str) and not error["type"]:
        structural.append("error type is empty")
    if isinstance(error.get("source_evidence"), str) and not error["source_evidence"]:
        structural.append("error source_evidence is empty")
    confidence = error.get("confidence")
    if not _is_number(confidence) or not 0 <= confidence <= 1:
        structural.append("error confidence is outside 0..1")
    quote = error.get("target_exact_quote")
    if isinstance(quote, str) and quote and quote not in target:
        evidence.append("target_exact_quote is not an exact target substring")
    if isinstance(quote, str) and not quote and not error.get("source_span"):
        evidence.append("omission error lacks source_span support")
    return structural, evidence


def validate_result_diagnostics(track: str, parsed: Any, task: dict[str, Any]) -> dict[str, list[str]]:
    """Validate structure/score fields and separately retain evidence diagnostics."""

    if not isinstance(parsed, dict):
        return {"structural_errors": ["result is not an object"], "evidence_diagnostics": []}
    if track == "pairwise":
        allowed = {"winner", "confidence", "critical_override", "reason_codes"}
        problems = []
        if set(parsed) != allowed:
            problems.append(f"pairwise keys mismatch: expected {sorted(allowed)}, got {sorted(parsed)}")
        if parsed.get("winner") not in {"A", "B", "TIE"}:
            problems.append("pairwise winner is invalid")
        if not _is_number(parsed.get("confidence")) or not 0 <= parsed.get("confidence", -1) <= 1:
            problems.append("pairwise confidence is outside 0..1")
        if not isinstance(parsed.get("critical_override"), bool):
            problems.append("pairwise critical_override is not boolean")
        codes = parsed.get("reason_codes")
        if not isinstance(codes, list) or not codes or any(not isinstance(x, str) or not x for x in codes):
            problems.append("pairwise reason_codes must be a non-empty string array")
        return {"structural_errors": problems, "evidence_diagnostics": []}
    if track != "pointwise":
        return {"structural_errors": [f"unsupported validation track: {track}"], "evidence_diagnostics": []}

    allowed = {"blind_id", "status", "abstain", "score_0_100", "subscores", "critical", "errors", "block_reason"}
    problems: list[str] = []
    evidence_diagnostics: list[str] = []
    if set(parsed) != allowed:
        problems.append(f"pointwise keys mismatch: expected {sorted(allowed)}, got {sorted(parsed)}")
    candidate = (task.get("candidates") or [{}])[0]
    expected_blind = candidate.get("blind_id")
    if parsed.get("blind_id") != expected_blind:
        problems.append(f"blind_id mismatch: expected {expected_blind!r}")
    status = parsed.get("status")
    if status not in {"ok", "blocked", "missing"}:
        problems.append("pointwise status is invalid")
    if not isinstance(parsed.get("abstain"), bool):
        problems.append("pointwise abstain is not boolean")
    if not isinstance(parsed.get("critical"), bool):
        problems.append("pointwise critical is not boolean")
    errors = parsed.get("errors")
    if not isinstance(errors, list):
        problems.append("pointwise errors is not an array")
        errors = []
    target = str(candidate.get("target_zh_tw", candidate.get("target", "")))
    for i, error in enumerate(errors):
        structural, evidence = _validate_error(error, target)
        for issue in structural:
            problems.append(f"errors[{i}]: {issue}")
        for issue in evidence:
            evidence_diagnostics.append(f"errors[{i}]: {issue}")
    block_reason = parsed.get("block_reason")
    if not isinstance(block_reason, str):
        problems.append("block_reason is not a string")
        block_reason = ""
    if status == "ok":
        if parsed.get("abstain") is not False:
            problems.append("ok result must have abstain=false")
        score = parsed.get("score_0_100")
        if not _is_integer(score) or not 0 <= score <= 100:
            problems.append("ok score_0_100 must be an integer in 0..100")
        subscores = parsed.get("subscores")
        fields = ("terminology_locale", "semantic_safety", "fluency", "completeness_format")
        if not isinstance(subscores, dict):
            problems.append("ok subscores must be an object")
        else:
            if set(subscores) != set(fields):
                problems.append("ok subscores keys mismatch")
            limits = {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 20}
            for field in fields:
                value = subscores.get(field)
                if not _is_integer(value) or not 0 <= value <= limits[field]:
                    problems.append(f"subscores.{field} outside allowed bounds")
            if all(_is_integer(subscores.get(field)) for field in fields) and _is_integer(score):
                if sum(subscores[field] for field in fields) != score:
                    problems.append("subscores do not sum exactly to score_0_100")
        if block_reason != "":
            problems.append("ok result must use empty block_reason")
    elif status in {"blocked", "missing"}:
        if parsed.get("abstain") is not True:
            problems.append("blocked/missing result must have abstain=true")
        if parsed.get("score_0_100") is not None:
            problems.append("blocked/missing score must be null")
        if parsed.get("subscores") is not None:
            problems.append("blocked/missing subscores must be null")
        if parsed.get("critical") is not False:
            problems.append("blocked/missing critical must be false")
        if errors:
            problems.append("blocked/missing errors must be empty")
        if not block_reason:
            problems.append("blocked/missing block_reason must be non-empty")
    if isinstance(errors, list) and isinstance(parsed.get("critical"), bool):
        has_critical_error = any(isinstance(error, dict) and error.get("severity") == "critical" for error in errors)
        if has_critical_error and parsed["critical"] is not True:
            evidence_diagnostics.append("critical=false disagrees with critical-severity error")
        if parsed["critical"] is True and not has_critical_error:
            evidence_diagnostics.append("critical=true has no critical-severity error")
    return {"structural_errors": problems, "evidence_diagnostics": evidence_diagnostics}


def validate_result(track: str, parsed: Any, task: dict[str, Any]) -> list[str]:
    """Compatibility helper returning only structural invalidation errors."""

    return validate_result_diagnostics(track, parsed, task)["structural_errors"]


def task_body(task: dict[str, Any], track: str, *, include_image: bool = False) -> str:
    """Render exactly one public task, with one candidate or pair per call."""

    input_type = str(task.get("input_type", "text")).lower()
    source_block = (
        "Source image: use the attached PNG pixels only. No OCR transcript or image ground truth is supplied."
        if input_type == "image" or include_image
        else f"English source:\n{task['source_en']}"
    )
    if track == "pointwise":
        candidates = task.get("candidates") or []
        if len(candidates) != 1:
            raise RunnerError("pointwise task must be reduced to exactly one candidate before calling the judge")
        c = candidates[0]
        return f"Task kind: pointwise\nItem ID: {task['item_id']}\nSource ID: {task.get('source_id', task.get('source_label', task['item_id']))}\n{source_block}\n\nAnonymous candidate {c['blind_id']}:\n{c.get('target_zh_tw', c.get('target', ''))}"
    if track == "pairwise":
        a, b = task["candidate_a"], task["candidate_b"]
        return f"Task kind: pairwise\nPair ID: {task['pair_id']}\nOrientation: {task['orientation']}\n{source_block}\n\nA ({a['blind_id']}):\n{a.get('target_zh_tw', a.get('target', ''))}\n\nB ({b['blind_id']}):\n{b.get('target_zh_tw', b.get('target', ''))}"
    variants = task.get("variants") or []
    return f"Task kind: qualification\nQualification ID: {task['qualification_id']}\n{source_block}\n\n" + "\n\n".join(f"{v['blind_id']}:\n{v.get('target_zh_tw', v.get('target', ''))}" for v in variants)


def attachment_for_task(task: dict[str, Any], public_dir: Path) -> Path | None:
    for key in ("image_path", "image_attachment", "attachment", "image_file", "image_alias", "image_filename"):
        value = task.get(key)
        if value:
            path = Path(str(value))
            if not path.is_absolute():
                path = public_dir / (Path("images") / path if key in {"image_alias", "image_filename"} else path)
            resolved = path.resolve()
            public_root = public_dir.resolve()
            if public_root not in resolved.parents and resolved != public_root:
                raise RunnerError(f"public image attachment escapes public directory: {resolved}")
            if resolved.exists() and resolved.is_file():
                return resolved
            raise RunnerError(f"public image attachment missing: {resolved}")
    # Some packet builders place the safe image alias on each candidate.
    for candidate in task.get("candidates", []):
        if isinstance(candidate, dict):
            for key in ("image_path", "image_attachment", "attachment", "image_file", "image_alias", "image_filename"):
                value = candidate.get(key)
                if value:
                    clone = dict(task)
                    clone[key] = value
                    return attachment_for_task(clone, public_dir)
    return None


def task_records(packet: dict[str, Any], tracks: Iterable[str]) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for track in tracks:
        section = packet.get(track, {})
        key = "tasks" if track == "pairwise" else "items"
        for task in section.get(key, []):
            if track == "pointwise":
                for candidate in task.get("candidates", []):
                    one = dict(task)
                    one["candidates"] = [candidate]
                    req_id = f"pointwise-{task.get('job_id', task['item_id'])}"
                    rows.append((track, req_id, one))
            elif track == "pairwise":
                req_id = f"pairwise-{task.get('pair_job_id', task['pair_id'])}"
                rows.append((track, req_id, task))
            else:
                req_id = f"qualification-{task['qualification_id']}"
                rows.append((track, req_id, task))
    return rows


def output_text_from_events(events: list[dict[str, Any]], last_message: Path | None) -> str:
    if last_message and last_message.exists():
        text = last_message.read_text(encoding="utf-8")
        if text.strip():
            return text
    for event in reversed(events):
        item = event.get("item") if isinstance(event, dict) else None
        if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
            return item["text"]
    return ""


def _find_model_evidence(events: list[Any]) -> str | None:
    """Find a provider-reported model name without guessing one."""

    keys = {"resolved_model", "model_name", "model_id", "model"}

    def walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            for child in value.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found:
                    return found
        return None

    return walk(events)


def run_codex_one(*, track: str, request_id: str, task: dict[str, Any], instruction: str, public_schema: Path, transport_schema: Path, schema_metadata: dict[str, Any], public_dir: Path, raw_dir: Path, model: str, timeout: float, judge_slot: str, provenance: dict[str, Any], raw_filename: str | None = None, attempt: int = 0) -> dict[str, Any]:
    """Run one exact GPT task in one fresh ephemeral Codex context."""

    attachment = attachment_for_task(task, public_dir)
    with tempfile.TemporaryDirectory(prefix="judge-gpt-", dir="/private/tmp") as scratch:
        scratch_path = Path(scratch)
        scratch_schema = scratch_path / "output.schema.json"
        shutil.copyfile(transport_schema, scratch_schema)
        prompt = (
            "You are one independent blind translation-quality judge. Use only the attached public task and the frozen evaluator instructions below. "
            "This is one fresh context; do not inspect files, repository state, prior runs, scores, or outside information. Do not browse or call tools. "
            "Return exactly one JSON value conforming to the supplied strict transport schema, with no Markdown, explanation, or rewrite. "
            "For pointwise status=ok, set block_reason to the empty string; for status=blocked or missing, set a concise non-empty block_reason.\n\n"
            "FROZEN EVALUATOR INSTRUCTIONS:\n" + instruction + "\n\nTASK:\n" + task_body(task, track, include_image=attachment is not None)
        )
        cmd = [
            "codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
            "--sandbox", "read-only", "--model", model, "-C", str(scratch_path),
            "-c", 'web_search="disabled"', "-c", 'model_reasoning_effort="max"',
            "--output-schema", str(scratch_schema), "--output-last-message", str(scratch_path / "last_message.txt"), "--json",
        ]
        # ``--image <FILE>...`` is a variadic option in the current Codex
        # CLI.  If the positional prompt follows it, the parser can consume
        # the prompt as another image path and then fall back to stdin (the
        # resulting diagnostic is ``No prompt provided via stdin``).  Put the
        # prompt before the variadic option so image jobs retain their exact
        # native-pixel attachment while still receiving the prompt argument.
        cmd.append(prompt)
        if attachment:
            cmd.extend(["--image", str(attachment)])
        started = time.monotonic()
        try:
            proc = subprocess.run(cmd, cwd=str(scratch_path), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            proc = None
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        else:
            stdout = proc.stdout
            stderr = proc.stderr
        elapsed = time.monotonic() - started
        events: list[Any] = []
        for line in stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        last_message = scratch_path / "last_message.txt"
        # Codex only writes this path when requested; keep this fallback for
        # future CLI versions that expose it in JSON events.
        content = output_text_from_events(events, last_message)
        if not content and stdout.strip() and not events:
            content = stdout.strip()
        resolved_model = _find_model_evidence(events)
        route_verification = "provider_reported_model" if resolved_model else "explicit_cli_model_flag;provider_model_not_reported"
        public_task_sha = sha256_json(task)
        image_sha = sha256_file(attachment) if attachment else None
        public_source_sha = sha256_bytes(str(task.get("source_en", "")).encode("utf-8")) if task.get("source_en") is not None else None
        raw = {
            "schema_version": "new-cloud-judge-raw-v1",
            "request_id": request_id,
            "attempt": attempt,
            "track": track,
            "judge_slot": judge_slot,
            "provenance": redact(provenance),
            "task_binding": {
                "public_task_sha256": public_task_sha,
                "job_id": task.get("job_id"),
                "pair_job_id": task.get("pair_job_id"),
                "blind_id": task.get("candidates", [{}])[0].get("blind_id") if track == "pointwise" else None,
                "candidate_a_blind_id": task.get("candidate_a", {}).get("blind_id") if track == "pairwise" else None,
                "candidate_b_blind_id": task.get("candidate_b", {}).get("blind_id") if track == "pairwise" else None,
                "source_sha256": public_source_sha,
                "image_sha256": image_sha,
            },
            "judge": {"judge_id": GPT_JUDGE_ID, "provider": "openai", "route": "codex_cli", "requested_model": model, "resolved_model": resolved_model, "reasoning_effort": "max", "auth": "ChatGPT-managed existing login", "route_verification": route_verification},
            "settings": {"fresh_context": True, "ephemeral": True, "sandbox": "read-only", "web_search": "disabled", "strict_output_schema": str(public_schema), "transport_schema": str(transport_schema), "transport_schema_sha256": schema_metadata.get("transport_schema_sha256"), "public_schema_sha256": schema_metadata.get("public_schema_sha256"), "image_attachment": str(attachment) if attachment else None},
            "elapsed_s": elapsed,
            "exit_code": None if proc is None else proc.returncode,
            "timed_out": timed_out,
            "stdout_sha256": sha256_bytes(stdout.encode("utf-8", errors="replace")),
            "stderr_sha256": sha256_bytes(stderr.encode("utf-8", errors="replace")),
            "events": redact(events),
            "stdout": redact(stdout),
            "stderr": redact(stderr),
            "assistant_content": redact(content),
            "assistant_content_sha256": sha256_bytes(content.encode("utf-8")),
            "status": "ok" if (not timed_out and proc is not None and proc.returncode == 0 and content.strip()) else "failed",
        }
        raw_path = raw_dir / (raw_filename or f"{request_id}.gpt.json")
        if raw_path.exists():
            raise RunnerError(f"refusing to overwrite immutable raw record: {raw_path}")
        if resolved_model and resolved_model != model:
            raw["status"] = "route_mismatch"
            raw["route_mismatch"] = {"requested_model": model, "reported_model": resolved_model}
        write_json(raw_path, raw)
        if raw["status"] == "route_mismatch":
            return {"status": "route_mismatch", "request_id": request_id, "raw_path": str(raw_path), "reason": f"provider reported {resolved_model!r}, requested {model!r}"}
        if raw["status"] != "ok":
            return {"status": "failed", "request_id": request_id, "raw_path": str(raw_path), "reason": "codex CLI failed or returned no final message"}
        try:
            parsed = extract_json(content)
        except RunnerError as exc:
            raw["parse_error"] = str(exc)
            write_json(raw_path, raw)
            return {"status": "failed", "request_id": request_id, "raw_path": str(raw_path), "reason": str(exc)}
        raw["parsed"] = redact(parsed)
        raw["parsed_sha256"] = sha256_json(parsed)
        diagnostics = validate_result_diagnostics(track, parsed, task)
        validation_errors = diagnostics["structural_errors"]
        raw["validation"] = {"status": "ok" if not validation_errors else "invalid", "errors": validation_errors, "evidence_diagnostics": diagnostics["evidence_diagnostics"]}
        if validation_errors:
            raw["status"] = "invalid"
        write_json(raw_path, raw)
        if validation_errors:
            return {"status": "failed", "request_id": request_id, "raw_path": str(raw_path), "reason": "post-validation failed", "validation_errors": validation_errors}
        result = {"status": "ok", "request_id": request_id, "raw_path": str(raw_path), "parsed": parsed}
        if diagnostics["evidence_diagnostics"]:
            result["evidence_diagnostics"] = diagnostics["evidence_diagnostics"]
        return result


def recover_existing_raw(*, track: str, request_id: str, task: dict[str, Any], raw_dir: Path, provenance: dict[str, Any], raw_path: Path | None = None) -> dict[str, Any] | None:
    """Reload an immutable raw record after a crash gap; never regrade it."""

    raw_path = raw_path or (raw_dir / f"{request_id}.gpt.json")
    if not raw_path.exists():
        return None
    raw = load_json(raw_path)
    if not isinstance(raw, dict) or raw.get("request_id") != request_id or raw.get("track") != track:
        raise RunnerError(f"existing raw record does not bind to request {request_id}: {raw_path}")
    stored_provenance = raw.get("provenance")
    if stored_provenance != provenance:
        raise RunnerError(f"existing raw provenance mismatch; refusing to regrade or overwrite: {raw_path}")
    if raw.get("status") == "route_mismatch":
        return {"status": "route_mismatch", "request_id": request_id, "raw_path": str(raw_path), "reason": "existing raw record reports a route mismatch"}
    parsed = raw.get("parsed")
    if parsed is None:
        return {"status": "failed", "request_id": request_id, "raw_path": str(raw_path), "reason": "existing raw crash-gap record has no parsed result; preserved without retry"}
    diagnostics = validate_result_diagnostics(track, parsed, task)
    if diagnostics["structural_errors"]:
        return {"status": "failed", "request_id": request_id, "raw_path": str(raw_path), "reason": "existing raw result remains structurally invalid", "validation_errors": diagnostics["structural_errors"]}
    result = {"status": "ok", "request_id": request_id, "raw_path": str(raw_path), "parsed": parsed, "recovered_existing_raw": True}
    if diagnostics["evidence_diagnostics"]:
        result["evidence_diagnostics"] = diagnostics["evidence_diagnostics"]
    return result


def retryable_infrastructure_failure(result: dict[str, Any], raw_dir: Path) -> bool:
    """Allow one same-prompt retry for a preserved no-response infrastructure failure."""

    if result.get("status") != "failed":
        return False
    raw_path = Path(str(result.get("raw_path", "")))
    if not raw_path.is_absolute():
        raw_path = raw_dir / raw_path
    if not raw_path.exists():
        return False
    raw = load_json(raw_path)
    if not isinstance(raw, dict) or raw.get("parsed") is not None or raw.get("assistant_content"):
        return False
    stderr = str(raw.get("stderr", ""))
    return raw.get("exit_code") not in (None, 0) and any(token in stderr for token in ("failed to initialize in-process app-server client", "readonly database", "Operation not permitted", "No prompt provided via stdin"))


def verify_gpt_route() -> dict[str, Any]:
    codex = shutil.which("codex")
    if not codex:
        return {"status": "unavailable", "reason": "codex executable not found"}
    status = subprocess.run([codex, "login", "status"], capture_output=True, text=True, timeout=30, check=False)
    help_run = subprocess.run([codex, "exec", "--help"], capture_output=True, text=True, timeout=30, check=False)
    help_text = help_run.stdout + help_run.stderr
    required = ["--model", "--image", "--ephemeral", "--ignore-user-config", "--output-schema", "--json", "--sandbox"]
    missing = [flag for flag in required if flag not in help_text]
    if status.returncode != 0 or "logged in" not in (status.stdout + status.stderr).lower():
        return {"status": "unavailable", "reason": "Codex CLI is not logged in", "login_status": redact(status.stdout + status.stderr), "missing_flags": missing}
    if missing:
        return {"status": "unavailable", "reason": f"Codex CLI missing required flags: {missing}", "login_status": redact(status.stdout + status.stderr), "missing_flags": missing}
    return {"status": "available", "executable": codex, "cli_version": subprocess.run([codex, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip(), "login_status": redact(status.stdout + status.stderr), "requested_model": GPT_MODEL, "requested_reasoning_effort": "max", "required_flags": required}


def write_progress(progress_path: Path, *, status: str, route: dict[str, Any], counts: dict[str, int], notes: list[str]) -> None:
    lines = [
        "# New-data cloudjudge progress",
        "",
        f"- Status: `{status}`",
        f"- Updated: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`",
        f"- GPT route: `{route.get('status')}`; requested model `{route.get('requested_model', GPT_MODEL)}`; requested reasoning `{route.get('requested_reasoning_effort', 'max')}`",
        f"- Counts: `{json.dumps(counts, sort_keys=True)}`",
        "",
        "## Notes",
        "",
    ] + [f"- {note}" for note in notes]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=("gpt", "gemini", "both"), default="gpt")
    parser.add_argument("--start", action="store_true", help="send score calls; without this, only inspect/validate routes and packet")
    parser.add_argument("--root", type=Path, default=EVAL_DIR)
    parser.add_argument("--public-dir", type=Path, default=PUBLIC_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--packet-ready", type=Path, default=DEFAULT_PACKET_READY)
    parser.add_argument("--approval-marker", type=Path, default=None)
    parser.add_argument("--instructions", type=Path, default=None)
    parser.add_argument("--slot", default=None, help="generic public judge slot, e.g. judge-04; required to select exactly one 238+72 job stream")
    parser.add_argument("--tracks", default="pointwise,pairwise")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-infra", action="store_true", help="retry once only for a preserved no-response Codex initialization failure")
    args = parser.parse_args()
    root = args.root.resolve()
    public_dir = args.public_dir.resolve()
    manifest_path = args.manifest.resolve()
    ready_path = args.packet_ready.resolve()
    if args.approval_marker:
        approval_path = args.approval_marker.resolve()
        if approval_path.name != "ROOT_APPROVED":
            raise RunnerError("approval marker basename must be exactly ROOT_APPROVED")
    else:
        approval_path = root / "ROOT_APPROVED"
    raw_dir = root / "raw" / "cloud"
    progress_path = root / "progress_cloud.md"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tracks = tuple(x.strip() for x in args.tracks.split(",") if x.strip())
    route = verify_gpt_route()
    counts = {"planned": 0, "ok": 0, "failed": 0, "skipped": 0}
    notes = ["Official OpenAI docs were consulted for codex exec, --model, --image, --ephemeral, --output-schema, and ChatGPT-managed auth.", "No format-only repair is implemented in this runner; malformed outputs are preserved as failed with no quality retry.", "Codex CLI help exposes no project_doc_max_bytes setting; each call uses a fresh /private/tmp scratch cwd, --ignore-user-config, --skip-git-repo-check, and read-only sandbox.", "Gemini native UI scoring is paused after audit found the sent prompt did not embed the selected public JSON schema; full prompt/hash/conversation audit is preserved in raw/cloud/gemini_prompt_audit.jsonl."]
    gemini_route = {
        "status": "available_native_ui",
        "provider": "google",
        "route": "antigravity_native_ui",
        "visible_model_label": "Gemini 3.8 Flash High",
        "cli": "no verified batch/API route",
        "automation": "manual_cua_operator_required",
        "reason": "native UI route is available and model label is visible; this GPT runner will not fabricate or silently substitute Gemini responses",
    }
    notes.append(f"Gemini route inspection: {json.dumps(gemini_route, ensure_ascii=False, sort_keys=True)}")
    if args.retry_infra:
        notes.append("One infrastructure-only retry was explicitly requested for a preserved no-response Codex initialization failure; no score/evidence retry is permitted.")
    if args.judge in {"gpt", "both"}:
        notes.append(f"GPT route inspection: {json.dumps(route, ensure_ascii=False, sort_keys=True)}")
    write_progress(progress_path, status="ROUTE_INSPECTED", route=route, counts=counts, notes=notes)
    try:
        packet, manifest = load_public_packet(public_dir, manifest_path, slot=args.slot)
        tasks_for_count = task_records(packet, tracks)
        counts["planned"] = len(tasks_for_count)
        if args.start and not args.slot:
            raise RunnerError("--slot is required before score calls so exactly one public judge stream is selected")
    except RunnerError as exc:
        write_progress(progress_path, status="BLOCKED_PUBLIC_PACKET", route=route, counts=counts, notes=notes + [str(exc)])
        print(json.dumps({"status": "blocked", "reason": str(exc), "route": route}, ensure_ascii=False, indent=2))
        return 2
    if not args.start:
        write_progress(progress_path, status="READY_WAITING_FOR_ROOT_APPROVAL", route=route, counts=counts, notes=notes + ["No score calls made; pass --start only after the root approval marker is present."])
        print(json.dumps({"status": "ready_waiting_for_approval", "route": route, "gemini_route": gemini_route, "planned": counts["planned"], "packet_id": packet.get("packet_id")}, ensure_ascii=False, indent=2))
        return 0
    if args.judge in {"gemini", "both"}:
        reason = "Gemini native UI route is available but this CLI runner does not issue UI score calls; continue with the designated CUA operator and do not treat this as a Gemini score bundle."
        write_progress(progress_path, status="DEFERRED_GEMINI_NATIVE_UI", route=route, counts=counts, notes=notes + [reason])
        print(json.dumps({"status": "deferred_to_native_ui", "reason": reason, "gpt_route": route, "gemini_route": gemini_route}, ensure_ascii=False, indent=2))
        return 6
    try:
        ready_evidence = verify_ready(manifest_path, ready_path, approval_path)
    except RunnerError as exc:
        write_progress(progress_path, status="BLOCKED_MISSING_ROOT_APPROVAL", route=route, counts=counts, notes=notes + [str(exc)])
        print(json.dumps({"status": "blocked", "reason": str(exc), "route": route}, ensure_ascii=False, indent=2))
        return 3
    if args.judge in {"gpt", "both"} and route.get("status") != "available":
        write_progress(progress_path, status="BLOCKED_GPT_ROUTE", route=route, counts=counts, notes=notes + ["GPT score calls were not attempted because exact Codex CLI route/auth is unavailable."])
        print(json.dumps({"status": "blocked", "reason": "GPT route unavailable", "route": route, "approval": ready_evidence}, ensure_ascii=False, indent=2))
        return 4
    instruction, instruction_sha = resolve_instruction(public_dir, packet, args.instructions.resolve() if args.instructions else None)
    packet_sha = sha256_json(packet)
    manifest_sha = sha256_file(manifest_path)
    schema_cache: dict[str, tuple[Path, Path, dict[str, Any]]] = {}
    for track in tracks:
        public_schema = load_schema(public_dir, track)
        transport_schema, schema_metadata = derive_transport_schema(public_schema, track, raw_dir)
        schema_cache[track] = (public_schema, transport_schema, schema_metadata)
    schema_derivations = {key: value[2] for key, value in schema_cache.items()}
    checkpoint_path = raw_dir / "gpt.checkpoint.json"
    previous: dict[str, Any] = {}
    if args.resume and checkpoint_path.exists():
        previous = load_json(checkpoint_path)
        checkpoint_checks = {
            "packet_id": packet.get("packet_id"),
            "packet_sha256": packet_sha,
            "manifest_sha256": manifest_sha,
            "instruction_sha256": instruction_sha,
        }
        if any(previous.get(key) != value for key, value in checkpoint_checks.items()):
            raise RunnerError("resume checkpoint does not match current public packet/instructions/hashes")
        if previous.get("judge", {}).get("judge_slot") != args.slot or previous.get("judge", {}).get("requested_model") != GPT_MODEL or previous.get("judge", {}).get("requested_reasoning_effort") != "max":
            raise RunnerError("resume checkpoint route/mode does not match exact requested GPT route")
        if previous.get("schema_derivations") != schema_derivations:
            raise RunnerError("resume checkpoint schema derivation does not match frozen public schemas")
    elif checkpoint_path.exists():
        raise RunnerError("checkpoint exists; pass --resume to continue without regrading")
    results = list(previous.get("results", [])) if isinstance(previous.get("results"), list) else []
    attempt_history = list(previous.get("attempt_history", [])) if isinstance(previous.get("attempt_history"), list) else []
    retry_request_ids: set[str] = set()
    retry_attempt_paths: dict[str, Path] = {}
    if args.retry_infra:
        if not args.resume or not previous:
            raise RunnerError("--retry-infra requires --resume with an existing checkpoint")
        retained_results: list[Any] = []
        for prior in results:
            if isinstance(prior, dict) and retryable_infrastructure_failure(prior, raw_dir):
                logical_id = str(prior.get("request_id"))
                retry_path = raw_dir / f"{logical_id}.gpt.retry1.json"
                if retry_path.exists():
                    # A retry attempt is immutable.  Do not issue a second retry or
                    # overwrite the preserved retry raw, even if it also failed.
                    retained_results.append(prior)
                else:
                    retry_request_ids.add(logical_id)
                    retry_attempt_paths[logical_id] = retry_path
                    attempt_history.append({"kind": "infrastructure_retry_original", "result": prior})
            else:
                retained_results.append(prior)
        results = retained_results
        if not retry_request_ids:
            raise RunnerError("--retry-infra found no eligible preserved initialization failure")
    for prior in results:
        if isinstance(prior, dict):
            if prior.get("status") == "ok":
                counts["ok"] += 1
            elif prior.get("status") in {"failed", "route_mismatch"}:
                counts["failed"] += 1
    completed = {str(x.get("request_id")) for x in results if isinstance(x, dict) and x.get("status") in {"ok", "failed"}}
    tasks = task_records(packet, tracks)
    if args.max_requests is not None:
        tasks = tasks[: args.max_requests]
    for index, (track, request_id, task) in enumerate(tasks, 1):
        if request_id in completed and request_id not in retry_request_ids:
            counts["skipped"] += 1
            continue
        public_schema, transport_schema, schema_metadata = schema_cache[track]
        provenance = {
            "packet_sha256": packet_sha,
            "manifest_sha256": manifest_sha,
            "instruction_sha256": instruction_sha,
            "mode": track,
            "judge_slot": args.slot,
            "requested_model": GPT_MODEL,
            "requested_reasoning_effort": "max",
            "public_schema_sha256": schema_metadata["public_schema_sha256"],
            "transport_schema_sha256": schema_metadata["transport_schema_sha256"],
        }
        retry_path = retry_attempt_paths.get(request_id)
        result = None if retry_path else recover_existing_raw(track=track, request_id=request_id, task=task, raw_dir=raw_dir, provenance=provenance)
        if result is None:
            result = run_codex_one(track=track, request_id=request_id, task=task, instruction=instruction, public_schema=public_schema, transport_schema=transport_schema, schema_metadata=schema_metadata, public_dir=public_dir, raw_dir=raw_dir, model=GPT_MODEL, timeout=args.timeout, judge_slot=args.slot, provenance=provenance, raw_filename=retry_path.name if retry_path else None, attempt=1 if retry_path else 0)
        results.append(result)
        if result.get("status") == "ok":
            counts["ok"] += 1
        else:
            counts["failed"] += 1
        checkpoint = {"schema_version": "new-cloud-judge-checkpoint-v1", "status": "CHECKPOINT", "packet_id": packet.get("packet_id"), "packet_sha256": packet_sha, "manifest_sha256": manifest_sha, "instruction_sha256": instruction_sha, "approval": ready_evidence, "judge": {"judge_id": GPT_JUDGE_ID, "judge_slot": args.slot, "requested_model": GPT_MODEL, "requested_reasoning_effort": "max", "route": route}, "schema_derivations": schema_derivations, "attempt_history": redact(attempt_history), "infra_retry_used": bool(args.retry_infra), "results": redact(results), "counts": counts, "last_request_id": request_id, "last_index": index}
        write_json(checkpoint_path, checkpoint)
        write_progress(progress_path, status="RUNNING_GPT", route=route, counts=counts, notes=notes + [f"Real first checkpoint written after `{request_id}`; raw evidence is under `{raw_dir}`."])
        if result.get("status") == "route_mismatch":
            write_progress(progress_path, status="BLOCKED_MODEL_ROUTE_MISMATCH", route=route, counts=counts, notes=notes + [f"Stopped immediately at `{request_id}` because provider-reported model differed from exact requested model."])
            print(json.dumps({"status": "blocked_model_route_mismatch", "request_id": request_id, "result": result}, ensure_ascii=False, indent=2))
            return 7
    final = {"schema_version": "new-cloud-judge-bundle-v1", "status": "COMPLETE" if counts["failed"] == 0 and counts["ok"] == counts["planned"] else "PARTIAL", "packet_id": packet.get("packet_id"), "packet_sha256": packet_sha, "manifest_sha256": manifest_sha, "instruction_sha256": instruction_sha, "approval": ready_evidence, "judge": {"judge_id": GPT_JUDGE_ID, "judge_slot": args.slot, "requested_model": GPT_MODEL, "requested_reasoning_effort": "max", "route": route}, "schema_derivations": schema_derivations, "attempt_history": redact(attempt_history), "infra_retry_used": bool(args.retry_infra), "results": redact(results), "counts": counts}
    write_json(raw_dir / "gpt.bundle.json", final)
    write_progress(progress_path, status=final["status"], route=route, counts=counts, notes=notes + [f"Bundle written to `{raw_dir / 'gpt.bundle.json'}`."])
    print(json.dumps({"status": final["status"], "counts": counts, "bundle": str(raw_dir / "gpt.bundle.json")}, ensure_ascii=False, indent=2))
    return 0 if final["status"] == "COMPLETE" else 5


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"cloud runner blocked: {exc}", file=sys.stderr)
        raise SystemExit(2)
