#!/usr/bin/env python3
"""Render a provenance-checked Gemini prompt-format v2.

This renderer is deliberately separate from the original v1 renderer.  It
does not modify the frozen packet, frozen shared instructions, v1 prompts, or
any raw capture.  Every v2 prompt embeds the exact frozen
``judge_instructions.txt`` text and then appends only a batch-specific JSON
envelope contract and the corresponding public blind task data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKET = ROOT / "judge_packet.json"
DEFAULT_INSTRUCTIONS = ROOT / "judge_instructions.txt"
RENDER_VERSION = "gemini-prompt-format-v2"
MANIFEST_SCHEMA = "gemini-prompt-format-v2-manifest-v1"


class PromptRenderError(ValueError):
    """Raised when frozen input or a rendered prompt is not self-consistent."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _source_hash(source: str) -> str:
    return sha256_text(source)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptRenderError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise PromptRenderError(f"{label} must be a non-empty array")
    return value


def _packet_basics(packet: Mapping[str, Any], packet_path: Path, instructions_path: Path) -> tuple[str, str, str]:
    packet_id = packet.get("packet_id")
    if not isinstance(packet_id, str) or not packet_id:
        raise PromptRenderError("frozen packet must contain a non-empty packet_id")
    packet_hash = sha256_file(packet_path)
    instructions_hash = sha256_file(instructions_path)
    declared_instructions_hash = packet.get("judge_instructions_sha256")
    if not isinstance(declared_instructions_hash, str) or declared_instructions_hash.lower() != instructions_hash:
        raise PromptRenderError("frozen packet judge_instructions_sha256 does not match frozen instructions")
    return packet_id, packet_hash, instructions_hash


def _check_public_task_object(value: Mapping[str, Any], label: str) -> None:
    """Reject obvious owner-only material before it can reach a prompt."""

    forbidden = {"answer_key", "candidate_mapping", "model_mapping", "expected_answer", "old_score", "prior_score"}

    def visit(child: Any, path: str) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                if str(key).lower() in forbidden:
                    raise PromptRenderError(f"{label} contains forbidden private field at {path}.{key}")
                visit(nested, f"{path}.{key}")
        elif isinstance(child, list):
            for index, nested in enumerate(child):
                visit(nested, f"{path}[{index}]")

    visit(value, label)


def _qualification_spec(packet: Mapping[str, Any]) -> dict[str, Any]:
    qualification = _require_mapping(packet.get("qualification"), "packet.qualification")
    items = _require_list(qualification.get("items"), "packet.qualification.items")
    payload: list[dict[str, Any]] = []
    record_keys: list[str] = []
    judgment_keys: list[str] = []
    seen_items: set[str] = set()
    for item in items:
        item_map = _require_mapping(item, "qualification item")
        item_id = item_map.get("qualification_id", item_map.get("item_id"))
        source = item_map.get("source_en")
        variants = item_map.get("variants")
        if not isinstance(item_id, str) or not item_id:
            raise PromptRenderError("qualification item missing qualification_id")
        if item_id in seen_items:
            raise PromptRenderError(f"duplicate qualification item {item_id}")
        seen_items.add(item_id)
        if not isinstance(source, str) or not source:
            raise PromptRenderError(f"qualification item {item_id} missing source_en")
        if not isinstance(variants, list) or not variants:
            raise PromptRenderError(f"qualification item {item_id} missing variants")
        variant_ids: list[str] = []
        for variant in variants:
            variant_map = _require_mapping(variant, f"qualification {item_id} variant")
            blind_id = variant_map.get("blind_id")
            if not isinstance(blind_id, str) or not blind_id:
                raise PromptRenderError(f"qualification {item_id} variant missing blind_id")
            if blind_id in variant_ids:
                raise PromptRenderError(f"duplicate qualification variant {item_id}/{blind_id}")
            variant_ids.append(blind_id)
            judgment_keys.append(f"{item_id}/{blind_id}")
        record_keys.append(item_id)
        payload.append({
            "qualification_id": item_id,
            "source_id": item_id,
            "source_en": source,
            "source_text_sha256": _source_hash(source),
            "target_locale": item_map.get("target_locale"),
            "variants": variants,
        })
    result = {
        "batch_id": "qualification-all",
        "task_id": "qualification-all",
        "track": "qualification",
        "task_payload": {"qualification_items": payload},
        "expected": {
            "record_count": len(record_keys),
            "judgment_count": len(judgment_keys),
            "record_keys": record_keys,
            "judgment_keys": judgment_keys,
        },
    }
    _check_public_task_object(result["task_payload"], "qualification task payload")
    return result


def _pointwise_specs(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    pointwise = _require_mapping(packet.get("pointwise"), "packet.pointwise")
    items = _require_list(pointwise.get("items"), "packet.pointwise.items")
    specs: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    for item in items:
        item_map = _require_mapping(item, "pointwise item")
        item_id = item_map.get("item_id")
        source = item_map.get("source_en")
        candidates = item_map.get("candidates")
        if not isinstance(item_id, str) or not item_id:
            raise PromptRenderError("pointwise item missing item_id")
        if item_id in seen_items:
            raise PromptRenderError(f"duplicate pointwise item {item_id}")
        seen_items.add(item_id)
        if not isinstance(source, str) or not source:
            raise PromptRenderError(f"pointwise item {item_id} missing source_en")
        if not isinstance(candidates, list) or not candidates:
            raise PromptRenderError(f"pointwise item {item_id} missing candidates")
        candidate_ids: list[str] = []
        for candidate in candidates:
            candidate_map = _require_mapping(candidate, f"pointwise {item_id} candidate")
            blind_id = candidate_map.get("blind_id", candidate_map.get("candidate_id"))
            if not isinstance(blind_id, str) or not blind_id:
                raise PromptRenderError(f"pointwise {item_id} candidate missing blind_id")
            if blind_id in candidate_ids:
                raise PromptRenderError(f"duplicate pointwise candidate {item_id}/{blind_id}")
            candidate_ids.append(blind_id)
        source_id = item_map.get("source_label", item_id)
        task_payload = {
            "item_id": item_id,
            "source_id": source_id,
            "source_text_sha256": _source_hash(source),
            "domain": item_map.get("domain"),
            "input_type": item_map.get("input_type"),
            "source_en": source,
            "target_locale": item_map.get("target_locale"),
            "candidates": candidates,
        }
        spec = {
            "batch_id": f"pointwise-{item_id}",
            "task_id": f"pointwise-{item_id}",
            "track": "pointwise",
            "task_payload": task_payload,
            "expected": {
                "record_count": len(candidate_ids),
                "record_keys": [f"{item_id}/{candidate_id}" for candidate_id in candidate_ids],
                "candidate_ids": candidate_ids,
                "item_ids": [item_id],
            },
        }
        _check_public_task_object(task_payload, f"pointwise task payload {item_id}")
        specs.append(spec)
    return specs


def _pairwise_specs(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairwise = _require_mapping(packet.get("pairwise"), "packet.pairwise")
    tasks = _require_list(pairwise.get("tasks"), "packet.pairwise.tasks")
    specs: list[dict[str, Any]] = []
    for orientation in ("AB", "BA"):
        selected = [task for task in tasks if isinstance(task, Mapping) and str(task.get("orientation", "")).upper() == orientation]
        if not selected:
            raise PromptRenderError(f"packet has no pairwise {orientation} tasks")
        payload: list[dict[str, Any]] = []
        record_keys: list[str] = []
        seen_pairs: set[str] = set()
        for task in selected:
            task_map = _require_mapping(task, f"pairwise {orientation} task")
            pair_id = task_map.get("pair_id")
            item_id = task_map.get("item_id")
            source = task_map.get("source_en")
            candidate_a = task_map.get("candidate_a")
            candidate_b = task_map.get("candidate_b")
            if not all(isinstance(value, str) and value for value in (pair_id, item_id, source)):
                raise PromptRenderError(f"pairwise {orientation} task missing required fields")
            if not isinstance(candidate_a, Mapping) or not isinstance(candidate_b, Mapping):
                raise PromptRenderError(f"pairwise {orientation} task {pair_id} candidates must be objects")
            candidate_a_id = candidate_a.get("blind_id", candidate_a.get("candidate_id"))
            candidate_b_id = candidate_b.get("blind_id", candidate_b.get("candidate_id"))
            if not all(isinstance(value, str) and value for value in (candidate_a_id, candidate_b_id)):
                raise PromptRenderError(f"pairwise {orientation} task {pair_id} candidates missing blind_id")
            if pair_id in seen_pairs:
                raise PromptRenderError(f"duplicate pairwise {orientation} pair {pair_id}")
            seen_pairs.add(pair_id)
            if candidate_a_id == candidate_b_id:
                raise PromptRenderError(f"pairwise {orientation} task {pair_id} has identical candidates")
            task_payload = {
                "pair_id": pair_id,
                "item_id": item_id,
                "source_id": item_id,
                "source_text_sha256": _source_hash(source),
                "orientation": orientation,
                "source_en": source,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
            }
            payload.append(task_payload)
            record_keys.append(pair_id)
        spec = {
            "batch_id": f"pairwise-{orientation}",
            "task_id": f"pairwise-{orientation}",
            "track": "pairwise",
            "task_payload": {"pairwise_tasks": payload},
            "expected": {"record_count": len(record_keys), "record_keys": record_keys, "orientation": orientation},
        }
        _check_public_task_object(spec["task_payload"], f"pairwise {orientation} task payload")
        specs.append(spec)
    return specs


def batch_specs(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = [_qualification_spec(packet), *_pointwise_specs(packet), *_pairwise_specs(packet)]
    if len(specs) != 6:
        raise PromptRenderError(f"expected six Gemini batches, got {len(specs)}")
    return specs


def _filename_for_spec(spec: Mapping[str, Any]) -> str:
    batch_id = str(spec["batch_id"])
    if batch_id == "qualification-all":
        return "qualification_all.md"
    if batch_id.startswith("pointwise-"):
        # Keep the public v1 naming convention for source batches.  The
        # hyphen is part of the batch ID and makes the three source files
        # unambiguous when copied between chats.
        return f"{batch_id}.md"
    if batch_id.startswith("pairwise-"):
        orientation = batch_id.removeprefix("pairwise-")
        return f"pairwise_{orientation}.md"
    raise PromptRenderError(f"unknown batch id {batch_id}")


def _operator_contract(packet_id: str, packet_hash: str, spec: Mapping[str, Any]) -> str:
    track = str(spec["track"])
    batch_id = str(spec["batch_id"])
    if track == "qualification":
        record = {
            "qualification_id": "<canonical qualification item id>",
            "source_id": "<canonical source id>",
            "source_text_sha256": "<canonical source sha256>",
            "judgments": [{
                "blind_id": "<qual-A or qual-B>",
                "abstain": False,
                "score_0_100": "<integer 0-100 or null>",
                "subscores": "<four integer fields or null>",
                "critical": False,
                "errors": "<error objects supported by this source, or []>",
            }],
        }
    elif track == "pointwise":
        record = {
            "item_id": "<canonical item id>",
            "source_id": "<canonical source id>",
            "source_text_sha256": "<canonical source sha256>",
            "blind_id": "<cand-##>",
            "abstain": False,
            "score_0_100": "<integer 0-100 or null>",
            "subscores": "<four integer fields or null>",
            "critical": False,
            "errors": "<error objects supported by this source, or []>",
        }
    else:
        record = {
            "pair_id": "<canonical pair id>",
            "item_id": "<canonical item id>",
            "source_id": "<canonical source id>",
            "source_text_sha256": "<canonical source sha256>",
            "orientation": str(spec["expected"].get("orientation")),
            "candidate_a": "<canonical candidate_a>",
            "candidate_b": "<canonical candidate_b>",
            "winner": "<A, B, or TIE>",
            "confidence": "<number 0-1>",
            "critical_override": False,
            "reason_codes": "<reason-code array>",
        }
    envelope = {
        "packet_id": packet_id,
        "packet_sha256": packet_hash,
        "judge_id": "<operator fills>",
        "model_label_from_ui": "<operator fills exact visible Gemini UI label>",
        "task_id": "<operator fills exact task id>",
        "track": track,
        "batch_id": "<operator fills exact batch id>",
        "records": [record],
    }
    return (
        "## Batch response contract\n\n"
        "Return exactly one JSON object with this envelope. Do not fill or invent "
        "operator-only fields (`judge_id`, `model_label_from_ui`, `task_id`, or "
        "`batch_id`); leave their placeholders unchanged. The operator fills "
        "those fields from the chat provenance and visible Gemini UI. Do not "
        "self-report a model version. The fixed packet ID/hash and track must "
        "remain unchanged. Return one record for every supplied task/candidate; "
        "for qualification return every item and both anonymous judgments. If a "
        "supplied target is genuinely unjudgeable, still return its judgment "
        "with `abstain: true`, null score/subscores, and an explanatory note; "
        "never omit the record or fabricate a judgment.\n\n"
        "```json\n" + _json_block(envelope) + "\n```\n\n"
    )


def _render_prompt(packet_id: str, packet_hash: str, instructions_hash: str, instructions_text: str, spec: Mapping[str, Any]) -> str:
    batch_id = str(spec["batch_id"])
    track = str(spec["track"])
    payload = spec["task_payload"]
    return (
        f"# Gemini blind judge pilot — {batch_id} (prompt-format-v2)\n\n"
        "This is a provisional, non-clinical blind translation assessment. Use "
        "only the task data in this document. Do not browse, call tools, or use "
        "outside/generated reference material. Do not infer model names, model "
        "families, model size, speed, prior scores, expected answers, or a "
        "reference translation. Locale-appropriate terminology variants or "
        "synonyms are not automatic errors.\n\n"
        f"- Packet: `{packet_id}`\n"
        f"- Packet SHA-256 (raw `judge_packet.json`): `{packet_hash}`\n"
        f"- Instructions SHA-256: `{instructions_hash}`\n"
        f"- Track: `{track}`\n"
        f"- Batch: `{batch_id}`\n"
        f"- Prompt rendering version: `{RENDER_VERSION}`\n\n"
        "## Frozen shared judge instructions (verbatim; authoritative for every track)\n\n"
        + instructions_text
        + "\n## V2 operational rules\n\n"
        "The frozen shared instructions above apply without modification. The "
        "batch contract below only specifies the transport envelope and does not "
        "change rubric weights or scoring rules. Return UTF-8 JSON only, with no "
        "Markdown fences or reasoning. Keep operator-only metadata placeholders "
        "unchanged; the operator fills them after the chat.\n\n"
        + _operator_contract(packet_id, packet_hash, spec)
        + "## Canonical public task data\n\n"
        "The following JSON is copied from the frozen blind packet. Do not alter "
        "source text, target text, anonymous IDs, source hashes, candidate order, "
        "pair orientation, or task count.\n\n"
        "```json\n"
        + _json_block(payload)
        + "\n```\n\n"
        "Return the response envelope only after judging this batch."
        "\n"
    )


def _manifest_for(packet_id: str, packet_hash: str, instructions_hash: str, specs: Sequence[Mapping[str, Any]], files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "render_version": RENDER_VERSION,
        "packet_id": packet_id,
        "packet_sha256": packet_hash,
        "instructions_sha256": instructions_hash,
        "shared_instructions_embedded_verbatim": True,
        "batch_count": len(specs),
        "batches": [
            {
                "batch_id": spec["batch_id"],
                "task_id": spec["task_id"],
                "track": spec["track"],
                "filename": _filename_for_spec(spec),
                "task_data_sha256": canonical_json_sha256(spec["task_payload"]),
                "expected": spec["expected"],
            }
            for spec in specs
        ],
        "files": list(files),
        "note": "All task data are public blind packet data. No private answer key, candidate mapping, model name, or prior score is rendered.",
    }


def validate_rendered_manifest(output_dir: str | Path, manifest: Mapping[str, Any], instructions_text: str) -> dict[str, Any]:
    """Validate every v2 prompt file and its embedded frozen/task material."""

    directory = Path(output_dir)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise PromptRenderError("unsupported Gemini v2 manifest schema")
    failures: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 6:
        failures.append("manifest must list exactly six rendered files")
    for entry in files if isinstance(files, list) else []:
        if not isinstance(entry, Mapping):
            failures.append("manifest file entry is not an object")
            continue
        filename = entry.get("filename")
        if not isinstance(filename, str):
            failures.append("manifest file entry missing filename")
            continue
        path = directory / filename
        if not path.is_file():
            failures.append(f"missing rendered prompt {filename}")
            continue
        content = path.read_text(encoding="utf-8")
        if instructions_text not in content:
            failures.append(f"{filename} does not embed frozen instructions verbatim")
        if entry.get("sha256") != sha256_file(path):
            failures.append(f"{filename} SHA-256 mismatch")
        if entry.get("bytes") != path.stat().st_size:
            failures.append(f"{filename} byte count mismatch")
        batch_id = entry.get("batch_id")
        batch_specs_by_id = {str(batch.get("batch_id")): batch for batch in manifest.get("batches", []) if isinstance(batch, Mapping)}
        batch = batch_specs_by_id.get(str(batch_id))
        if batch is None:
            failures.append(f"{filename} has no batch manifest entry")
        else:
            if f"- Batch: `{batch_id}`" not in content:
                failures.append(f"{filename} batch ID is not rendered")
            if not isinstance(batch.get("task_data_sha256"), str) or f"Prompt rendering version: `{RENDER_VERSION}`" not in content:
                failures.append(f"{filename} missing render provenance")
    if failures:
        raise PromptRenderError("; ".join(failures))
    return {"status": "PASS", "files": len(files), "shared_instructions_embedded_verbatim": True}


def render(packet_path: str | Path, instructions_path: str | Path, output_dir: str | Path, *, replace_existing: bool = False) -> dict[str, Any]:
    packet_file = Path(packet_path)
    instructions_file = Path(instructions_path)
    directory = Path(output_dir)
    packet = json.loads(packet_file.read_text(encoding="utf-8"))
    if not isinstance(packet, Mapping):
        raise PromptRenderError("frozen packet must be a JSON object")
    instructions_text = instructions_file.read_text(encoding="utf-8")
    packet_id, packet_hash, instructions_hash = _packet_basics(packet, packet_file, instructions_file)
    specs = batch_specs(packet)
    if directory.exists() and any(directory.iterdir()):
        expected_names = {_filename_for_spec(spec) for spec in specs} | {"manifest.json"}
        actual_names = {child.name for child in directory.iterdir()}
        unexpected = sorted(actual_names - expected_names)
        if not replace_existing or unexpected:
            suffix = f"; unexpected entries: {unexpected}" if unexpected else ""
            raise PromptRenderError(f"refusing to overwrite non-empty output directory {directory}{suffix}")
    directory.mkdir(parents=True, exist_ok=True)
    file_entries: list[dict[str, Any]] = []
    for spec in specs:
        filename = _filename_for_spec(spec)
        content = _render_prompt(packet_id, packet_hash, instructions_hash, instructions_text, spec)
        if instructions_text not in content:
            raise PromptRenderError(f"renderer failed to embed instructions in {filename}")
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        file_entries.append({
            "filename": filename,
            "batch_id": spec["batch_id"],
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    manifest = _manifest_for(packet_id, packet_hash, instructions_hash, specs, file_entries)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_rendered_manifest(directory, manifest, instructions_text)
    return {"manifest": manifest_path, "files": [directory / entry["filename"] for entry in file_entries], "manifest_data": manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace only the renderer's six known prompts and manifest in an existing v2 directory",
    )
    args = parser.parse_args(argv)
    result = render(args.packet, args.instructions, args.output_dir, replace_existing=args.replace_existing)
    print(json.dumps({"manifest": str(result["manifest"]), "files": [str(path) for path in result["files"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
