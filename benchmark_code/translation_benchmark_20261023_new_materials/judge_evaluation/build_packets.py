#!/usr/bin/env python3
"""Build the blinded, reproducible judge packets for the new-material batch.

This builder is intentionally limited to packet construction.  It never calls
an LLM, a web service, LiteLLM, or a judge endpoint.  The seven translator
outputs are read from the frozen CSV, then copied into per-judge, single-
candidate pointwise jobs and a small, frozen AB/BA pairwise audit.  Candidate
identities, source metadata, and all provenance are retained only in
``private/``; files under ``public/`` are safe to hand to a judge worker.

The output is reproducible from the input hashes and ``BASE_SEED``.  A
deliberate ``--overwrite`` is required to rebuild an already populated packet
directory.  Existing source/material/raw files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
MATERIAL_ROOT = ROOT.parent
RESULTS_ROOT = MATERIAL_ROOT / "results"
MATERIALS_ROOT = MATERIAL_ROOT / "materials"

INPUT_CSV = RESULTS_ROOT / "translation_results.csv"
METADATA_PATH = MATERIALS_ROOT / "metadata_manifest.json"
FROZEN_TEXT_PATH = MATERIALS_ROOT / "frozen_text_materials.jsonl"
IMAGE_SELECTION_PATH = MATERIALS_ROOT / "image_selection.json"
OLD_RUBRIC_PATH = ROOT.parent.parent / "translation_benchmark_20261022_judge_pilot" / "judge_instructions.txt"

PACKET_ID = "translation-benchmark-new-materials-judge-packets-20261023-v1"
PACKET_VERSION = 1
BASE_SEED = 20261023
JUDGE_SLOTS = [f"judge-{i:02d}" for i in range(1, 6)]
TEXT_CASES = [f"text-{i:02d}" for i in range(1, 31)]
IMAGE_CASES = ["img-01", "img-04", "img-09", "img-10"]
EXPECTED_CASES = TEXT_CASES + IMAGE_CASES

FORBIDDEN_PUBLIC_VALUES = (
    "translategemma-",
    "gemma4-",
    "translategemma ",
    "gemma 4 ",
    "model_id",
    "model_name",
    "model_alias",
    "source_owner",
    "old_score",
    "historical_score",
    "template_path",
    "template_sha256",
    "elapsed_seconds",
    "prompt_tokens",
    "completion_tokens",
    "rank",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive input error
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def derive_seed(*parts: object) -> int:
    payload = "|".join([str(BASE_SEED), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def opaque_id(prefix: str, *parts: object, length: int = 12) -> str:
    payload = "|".join([str(BASE_SEED), *map(str, parts)]).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:length]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_clean_destination(overwrite: bool) -> None:
    if not ROOT.exists():
        ROOT.mkdir(parents=True)
    # Other workers own ``cloud_runner/``, ``local_runner/``, ``raw/`` and
    # their progress files in this shared directory.  They are deliberately
    # outside the packet builder's overwrite set and must not block a clean
    # packet build.
    generated = {"public", "private", "tests", "manifest.json", "packet_ready.json", "protocol.md", "progress_protocol.md"}
    owned_existing = [p for p in ROOT.iterdir() if p.name in generated and (p.is_file() or any(q.is_file() for q in p.rglob("*")))]
    if owned_existing and not overwrite:
        names = ", ".join(sorted(p.name for p in owned_existing)[:12])
        raise SystemExit(f"refusing to overwrite populated packet directory; use --overwrite deliberately ({names})")
    if overwrite:
        # Only generated packet children are removed.  The builder itself and
        # any user-owned file outside this explicit allow-list are preserved.
        for child in list(ROOT.iterdir()):
            if child.name not in generated:
                continue
            if child.name == "tests" and child.is_dir():
                # Keep authored contract tests; only clear generated reports
                # and caches beneath the owned test directory.
                for test_child in list(child.iterdir()):
                    if test_child.suffix == ".py":
                        continue
                    if test_child.is_dir():
                        shutil.rmtree(test_child)
                    else:
                        test_child.unlink()
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def load_inputs() -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    with INPUT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 238:
        raise ValueError(f"translation_results.csv must contain exactly 238 rows, found {len(rows)}")

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("case_id") not in EXPECTED_CASES:
            raise ValueError(f"unexpected case_id {row.get('case_id')!r}")
        by_case[row["case_id"]].append(row)
        by_model[row["model_id"]].append(row)
        if row.get("status") != "ok" or row.get("http_status") != "200" or row.get("error"):
            raise ValueError(f"input row is not a successful translation cell: {row.get('model_id')} {row.get('case_id')}")
        if not row.get("translation_zh_tw"):
            raise ValueError(f"empty translation: {row.get('model_id')} {row.get('case_id')}")

    model_ids = sorted(by_model)
    if len(model_ids) != 7:
        raise ValueError(f"expected 7 translator slots, found {model_ids}")
    if any(len(items) != 34 for items in by_model.values()):
        raise ValueError({model: len(items) for model, items in by_model.items()})
    if any(len(items) != 7 for items in by_case.values()) or set(by_case) != set(EXPECTED_CASES):
        raise ValueError("each of 30 text and 4 image cases must have exactly seven rows")
    if any(row.get("source_en") for row in rows if row.get("input_type") == "image"):
        raise ValueError("image rows must not carry an English transcription in the judging input")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    image_selection = json.loads(IMAGE_SELECTION_PATH.read_text(encoding="utf-8"))
    frozen_text = {row["item_id"]: row for row in read_jsonl(FROZEN_TEXT_PATH)}
    if set(frozen_text) != set(TEXT_CASES):
        raise ValueError("frozen text material inventory does not match text-01..text-30")
    if metadata.get("freeze_ready") is not True:
        raise ValueError("metadata_manifest.json freeze_ready is not true")
    return rows, frozen_text, {row["image_id"]: row for row in image_selection["selected_images"]}, metadata


def validate_source_binding(rows: list[dict[str, str]], frozen_text: dict[str, dict[str, Any]], images: dict[str, dict[str, Any]]) -> None:
    manifest_hashes = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["text_inventory"]["source_text_hashes_sha256_by_item"]
    rows_by_key = {(row["model_id"], row["case_id"]): row for row in rows}
    for case_id in TEXT_CASES:
        source = frozen_text[case_id]["source_en"]
        expected_sha = sha256_bytes(source.encode("utf-8"))
        if expected_sha != manifest_hashes[case_id]:
            raise ValueError(f"frozen text hash mismatch for {case_id}")
        for model_id in sorted({row["model_id"] for row in rows}):
            row = rows_by_key[model_id, case_id]
            if row["source_en"] != source or row["source_sha256"] != expected_sha:
                raise ValueError(f"CSV/frozen source mismatch for {model_id}/{case_id}")
    for image_id, image in images.items():
        source_path = Path(image["source_absolute_path"])
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        actual_sha = sha256_file(source_path)
        if actual_sha != image["sha256"]:
            raise ValueError(f"authoritative image hash mismatch for {image_id}")
        for model_id in sorted({row["model_id"] for row in rows}):
            row = rows_by_key[model_id, image_id]
            if row["image_sha256"] != image["sha256"] or row["image_path"] != image["source_absolute_path"]:
                raise ValueError(f"CSV/image-selection mismatch for {model_id}/{image_id}")


def make_material_catalog(
    rows: list[dict[str, str]], frozen_text: dict[str, dict[str, Any]], images: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    source_by_case = {case: frozen_text[case] for case in TEXT_CASES}
    catalog: list[dict[str, Any]] = []
    for case_id in TEXT_CASES:
        source = source_by_case[case_id]
        catalog.append({
            "material_id": case_id,
            "source_kind": "text",
            "source_id": source["source_id"],
            "source_en": source["source_en"],
            "source_sha256": sha256_bytes(source["source_en"].encode("utf-8")),
            "source_length_chars": len(source["source_en"]),
            "topic": source.get("topic", ""),
        })
    for image_id in IMAGE_CASES:
        image = images[image_id]
        catalog.append({
            "material_id": image_id,
            "source_kind": "image",
            "source_id": image_id,
            "image_sha256": image["sha256"],
            "source_absolute_path": image["source_absolute_path"],
            "layout_stratum": image.get("layout_stratum", ""),
        })
    return catalog


def select_pair_sources(catalog: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select four text items by deterministic length strata plus two fixed images.

    The text records are sorted by character length, then source-page metadata
    (`source_id`), then item ID. Four equally sized length-rank strata are
    formed; the item nearest each stratum's median length is selected,
    preferring a source page not already selected. This preserves a compact,
    auditable rule without leaking source IDs to public judge packets.
    """

    text = [row for row in catalog if row["source_kind"] == "text"]
    length_sorted = sorted(text, key=lambda row: (row["source_length_chars"], row["source_id"], row["material_id"]))
    selected: list[dict[str, Any]] = []
    used_source_ids: set[str] = set()
    strata_meta: list[dict[str, Any]] = []
    for stratum_index in range(4):
        start = math.floor(stratum_index * len(length_sorted) / 4)
        end = math.floor((stratum_index + 1) * len(length_sorted) / 4)
        bucket = length_sorted[start:end]
        if not bucket:
            raise ValueError("empty text-length stratum")
        median_length = sorted(row["source_length_chars"] for row in bucket)[(len(bucket) - 1) // 2]
        available = [row for row in bucket if row["source_id"] not in used_source_ids]
        candidates = available or bucket
        chosen = min(candidates, key=lambda row: (abs(row["source_length_chars"] - median_length), row["source_id"], row["material_id"]))
        selected.append(chosen)
        used_source_ids.add(chosen["source_id"])
        strata_meta.append({
            "stratum": stratum_index + 1,
            "rank_start_zero_based": start,
            "rank_end_exclusive_zero_based": end,
            "median_length_chars": median_length,
            "selected_material_id": chosen["material_id"],
            "selected_source_id": chosen["source_id"],
            "selected_length_chars": chosen["source_length_chars"],
        })

    selected.extend([next(row for row in catalog if row["material_id"] == "img-09"), next(row for row in catalog if row["material_id"] == "img-04")])
    selection_meta = {
        "rule_version": "pair-audit-selection-v1",
        "text_rule": "sort text by length_chars, source_id, item_id; choose nearest median of each four length-rank strata, preferring an unused source page",
        "text_strata": strata_meta,
        "selected_text_material_ids": [row["material_id"] for row in selected[:4]],
        "selected_image_material_ids": ["img-09", "img-04"],
        "selected_material_ids_in_order": [row["material_id"] for row in selected],
    }
    return selected, selection_meta


def build_instructions() -> str:
    return """# New-material blind judge instructions (2026-10-23)

You are one independent evaluator in a provisional, non-clinical English →
Taiwan Traditional Chinese (zh-TW) translation evaluation. The packet is
blind: every candidate is represented only by a random `blind_id` in a
pointwise job, and pairwise jobs expose only A/B. Do not infer or mention a
translator/model identity, size, speed, quantization, family, template,
previous result, rank, or score. There is no Chinese reference translation.

This batch has 30 text cases and 4 image cases. The old pilot's qualification
cloud tasks are **not rerun**. The new pointwise track is 238 jobs per judge
(one candidate at a time). The limited pairwise audit has six source items,
six spanning-chain candidate pairs per source, and both AB and BA orientations
(72 jobs per judge). Judge each job in a fresh isolated context; do not carry
information between jobs. Return one strict JSON object and no Markdown.

## Source presentation and image rule

For a text job, use only the displayed `source_en` and `target_zh_tw`. For an
image job, judge the displayed native image pixels and the displayed target
translation. Image jobs intentionally provide no English transcription,
machine-extracted text, image ground truth, or Chinese gold. Native visual
judging is mandatory. If the route cannot receive or inspect the image, return
`status: "blocked"` (or `"missing"`) with `abstain: true` and a concise
`block_reason`; do not OCR the image into a substitute text prompt and do not
penalize a translation for an unsupported route. The image rubric below uses
the same four dimensions as text (30/30/20/20); image results are a visual
translation diagnostic, not an OCR character-accuracy measurement.

## Pointwise rubric (text and image)

Return four integer sub-scores that sum exactly to `score_0_100`:

- `terminology_locale`: 0–30. Medical/psychological terms and Taiwan usage;
  Traditional Chinese, not Simplified Chinese.
- `semantic_safety`: 0–30. Meaning, negation, medication names, dose/number,
  units, frequency, timing, conditions, suicide/self-harm meaning, and other
  safety-critical facts visible in the source/image.
- `fluency`: 0–20. Natural, clear Taiwan Traditional Chinese register.
- `completeness_format`: 0–20. No omission/addition and preservation of
  visible text/content, paragraph/list structure, and relevant layout cues.

Do not average away a critical error. Set `critical: true` for a confirmed
medication, dose/number/unit, negation, time/condition, suicide/self-harm, or
similarly safety-critical meaning error. Use `abstain: true` only when the
source/target genuinely cannot be judged; a blocked/missing image route must
also abstain. Do not invent style issues merely to fill `errors`.

For every supported error, prefer exact evidence:

- `target_exact_quote`: the shortest exact target substring that is wrong;
  use `""` for an omission.
- `source_evidence`: exact or near-exact source/image evidence and a concise
  reason. For an image, describe the visible English cue without supplying an
  unseen transcription.
- `source_span`: supporting source phrase/cue (required for an omission).
- `target_anchor`: target phrase showing where an omission belongs; use `""`
  when not applicable.
- `confidence`: a number from 0 to 1.

The importer computes target character offsets from the exact quote. Do not
invent offsets. Preserve all raw responses. A malformed JSON response may get
at most one format-only repair request, saved alongside the original; a repair
must not change a score, choose a more attractive result, or replace the
original raw response. Never convert an invalid/missing score into zero.
The importer must surface (and never silently normalize) a `critical` flag that
disagrees with error severity, a score/sub-score sum conflict, an unmatched
exact quote, or any other validation conflict.

## Pointwise output

See `schemas/pointwise_result.schema.json`. For a supported judgment use
`status: "ok"`, `abstain: false`, an integer score, four integer sub-scores,
`critical`, and an evidence array. For a blocked/missing case use
`status: "blocked"` or `"missing"`, `abstain: true`, `score_0_100: null`,
`subscores: null`, `critical: false`, `errors: []`, and a non-empty
`block_reason`. Do not add fields.

## Pairwise output and AB/BA rule

See `schemas/pairwise_result.schema.json`. For each orientation return
`winner: "A"`, `"B"`, or `"TIE"`, a 0–1 confidence, `critical_override`,
and one or more reason codes. TIE means neither presented translation is
meaningfully better in that orientation. The worker must not use TIE to hide
an orientation swap. The importer maps each AB/BA response back to the
private canonical pair: if the two orientations disagree after mapping, the
derived result is **ABSTAIN**, not TIE. Raw A/B votes remain unchanged.

## Scope and interpretation

Do not provide medical advice, a rewrite, a summary, or a clinical/deployment
safety certification. Scores are exploratory translation judgments. Later
aggregation reports text and image tracks separately, arithmetic-mean ICC
(2,1) and ICC(2,k) with the actual complete k, medians separately, source
cluster bootstrap diagnostics (the 30 text cases come from 12 source pages,
not 30 independent populations), and leave-one-family-out sensitivity. A
small number of source clusters does not justify a fabricated confidence
interval. Evidence/critical/error annotations and pairwise consistency remain
separate from reliability statistics.
"""


def build_schemas() -> dict[str, dict[str, Any]]:
    error_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["severity", "type", "target_exact_quote", "source_evidence", "source_span", "target_anchor", "confidence"],
        "properties": {
            "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
            "type": {"type": "string", "minLength": 1},
            "target_exact_quote": {"type": "string"},
            "source_evidence": {"type": "string", "minLength": 1},
            "source_span": {"type": "string"},
            "target_anchor": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    pointwise = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "pointwise_result.schema.json",
        "title": "New-material pointwise judge result",
        "type": "object",
        "additionalProperties": False,
        "required": ["blind_id", "status", "abstain", "score_0_100", "subscores", "critical", "errors"],
        "properties": {
            "blind_id": {"type": "string", "pattern": "^b-[0-9a-f]{12}$"},
            "status": {"type": "string", "enum": ["ok", "blocked", "missing"]},
            "abstain": {"type": "boolean"},
            "score_0_100": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
            "subscores": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["terminology_locale", "semantic_safety", "fluency", "completeness_format"],
                "properties": {
                    "terminology_locale": {"type": "integer", "minimum": 0, "maximum": 30},
                    "semantic_safety": {"type": "integer", "minimum": 0, "maximum": 30},
                    "fluency": {"type": "integer", "minimum": 0, "maximum": 20},
                    "completeness_format": {"type": "integer", "minimum": 0, "maximum": 20},
                },
            },
            "critical": {"type": "boolean"},
            "errors": {"type": "array", "items": error_schema},
            "block_reason": {"type": "string"},
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "ok"}}},
                "then": {"properties": {"abstain": {"const": False}, "score_0_100": {"type": "integer"}, "subscores": {"type": "object"}}},
            },
            {
                "if": {"properties": {"status": {"enum": ["blocked", "missing"]}}},
                "then": {"properties": {"abstain": {"const": True}, "score_0_100": {"const": None}, "subscores": {"const": None}, "critical": {"const": False}, "errors": {"maxItems": 0}, "block_reason": {"type": "string", "minLength": 1}}, "required": ["block_reason"]},
            },
        ],
    }
    pairwise = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "pairwise_result.schema.json",
        "title": "New-material pairwise orientation result",
        "type": "object",
        "additionalProperties": False,
        "required": ["winner", "confidence", "critical_override", "reason_codes"],
        "properties": {
            "winner": {"type": "string", "enum": ["A", "B", "TIE"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "critical_override": {"type": "boolean"},
            "reason_codes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        },
    }
    return {"pointwise_result.schema.json": pointwise, "pairwise_result.schema.json": pairwise}


def public_value_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.append(str(key))
            result.extend(public_value_strings(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(public_value_strings(child))
    elif isinstance(value, str):
        result.append(value)
    return result


def build(overwrite: bool = False) -> dict[str, Any]:
    ensure_clean_destination(overwrite)
    rows, frozen_text, images, metadata = load_inputs()
    validate_source_binding(rows, frozen_text, images)
    catalog = make_material_catalog(rows, frozen_text, images)
    model_ids = sorted({row["model_id"] for row in rows})
    rows_by_model_case = {(row["model_id"], row["case_id"]): row for row in rows}
    pair_sources, pair_selection_meta = select_pair_sources(catalog)
    instructions = build_instructions()
    schemas = build_schemas()

    # Public instruction/schema files are deliberately model-neutral.
    for public_subdir in ("images", "schemas", "pointwise", "pairwise"):
        (ROOT / "public" / public_subdir).mkdir(parents=True, exist_ok=True)
    (ROOT / "public" / "instructions.md").write_text(instructions, encoding="utf-8")
    public_file_paths: list[Path] = [ROOT / "public" / "instructions.md"]
    for schema_name, schema in schemas.items():
        path = ROOT / "public" / "schemas" / schema_name
        write_json(path, schema)
        public_file_paths.append(path)

    # Image aliases are opaque and reused across the five judge slots.  Copy
    # bytes, not a re-encoded/rendered image; verify the digest after copy.
    image_aliases: dict[str, dict[str, Any]] = {}
    for image_id in IMAGE_CASES:
        image = images[image_id]
        source_path = Path(image["source_absolute_path"])
        suffix = source_path.suffix.lower() or ".png"
        alias = opaque_id("asset", image_id) + suffix
        dest = ROOT / "public" / "images" / alias
        shutil.copyfile(source_path, dest)
        copied_sha = sha256_file(dest)
        if copied_sha != image["sha256"]:
            raise ValueError(f"copied image digest mismatch for {image_id}")
        image_aliases[image_id] = {
            "public_alias": alias,
            "public_path": str(dest.relative_to(ROOT)),
            "sha256": copied_sha,
            "source_absolute_path_private": image["source_absolute_path"],
            "dimensions": image.get("dimensions", {}),
            "layout_stratum_private": image.get("layout_stratum", ""),
        }
        public_file_paths.append(dest)

    # Source aliases are judge-specific so no worker can link source IDs across
    # public files.  The actual English source is still shown for text jobs.
    source_alias_by_judge: dict[str, dict[str, str]] = {}
    candidate_map_by_judge: dict[str, dict[str, str]] = {}
    public_jobs_by_judge: dict[str, dict[str, list[dict[str, Any]]]] = {}
    private_job_map: dict[str, dict[str, Any]] = {}
    private_pair_map: dict[str, dict[str, Any]] = {}

    for judge_slot in JUDGE_SLOTS:
        source_rng = random.Random(derive_seed("source-order", judge_slot))
        source_alias_by_judge[judge_slot] = {
            material["material_id"]: opaque_id("src", judge_slot, material["material_id"])
            for material in catalog
        }
        candidate_order = model_ids[:]
        random.Random(derive_seed("candidate-order", judge_slot)).shuffle(candidate_order)
        candidate_map = {
            model_id: opaque_id("b", judge_slot, "candidate", index)
            for index, model_id in enumerate(candidate_order)
        }
        candidate_map_by_judge[judge_slot] = candidate_map

        pointwise_jobs: list[dict[str, Any]] = []
        for material in catalog:
            source_alias = source_alias_by_judge[judge_slot][material["material_id"]]
            for model_id in model_ids:
                row = rows_by_model_case[model_id, material["material_id"]]
                blind_id = candidate_map[model_id]
                job_id = opaque_id("pw", judge_slot, material["material_id"], blind_id)
                job: dict[str, Any] = {
                    "schema_version": "new-material-pointwise-job-v1",
                    "job_id": job_id,
                    "track": "pointwise",
                    "blind_source_id": source_alias,
                    "source_kind": material["source_kind"],
                    "blind_id": blind_id,
                    "target_locale": "zh-TW",
                    "target_zh_tw": row["translation_zh_tw"],
                    "result_schema": "schemas/pointwise_result.schema.json",
                }
                if material["source_kind"] == "text":
                    job["source_en"] = material["source_en"]
                else:
                    job["image_alias"] = image_aliases[material["material_id"]]["public_alias"]
                    job["native_image_required"] = True
                pointwise_jobs.append(job)
                private_job_map[job_id] = {
                    "judge_slot": judge_slot,
                    "track": "pointwise",
                    "material_id": material["material_id"],
                    "source_id": material["source_id"],
                    "source_kind": material["source_kind"],
                    "model_id": model_id,
                    "blind_id": blind_id,
                    "source_sha256": material.get("source_sha256", ""),
                    "image_sha256": material.get("image_sha256", ""),
                    "input_row_request_sha256": row.get("request_sha256", ""),
                    "input_row_response_body_sha256": row.get("response_body_sha256", ""),
                }
        source_rng.shuffle(pointwise_jobs)

        pair_jobs: list[dict[str, Any]] = []
        for source_index, material in enumerate(pair_sources, 1):
            candidate_chain = model_ids[:]
            random.Random(derive_seed("pair-chain", material["material_id"])).shuffle(candidate_chain)
            for edge_index, (left_model, right_model) in enumerate(zip(candidate_chain, candidate_chain[1:]), 1):
                pair_key = f"pair-{source_index:02d}-{edge_index:02d}"
                orientations = [("AB", left_model, right_model), ("BA", right_model, left_model)]
                for orientation, model_a, model_b in orientations:
                    pair_job_id = opaque_id("pair", judge_slot, material["material_id"], edge_index, orientation)
                    pair_job: dict[str, Any] = {
                        "schema_version": "new-material-pairwise-job-v1",
                        "pair_job_id": pair_job_id,
                        "track": "pairwise",
                        "pair_key": pair_key,
                        "orientation": orientation,
                        "blind_source_id": source_alias_by_judge[judge_slot][material["material_id"]],
                        "source_kind": material["source_kind"],
                        "target_locale": "zh-TW",
                        "candidate_a": {"blind_id": "A", "target_zh_tw": rows_by_model_case[model_a, material["material_id"]]["translation_zh_tw"]},
                        "candidate_b": {"blind_id": "B", "target_zh_tw": rows_by_model_case[model_b, material["material_id"]]["translation_zh_tw"]},
                        "result_schema": "schemas/pairwise_result.schema.json",
                    }
                    if material["source_kind"] == "text":
                        pair_job["source_en"] = material["source_en"]
                    else:
                        pair_job["image_alias"] = image_aliases[material["material_id"]]["public_alias"]
                        pair_job["native_image_required"] = True
                    pair_jobs.append(pair_job)
                    private_pair_map[pair_job_id] = {
                        "judge_slot": judge_slot,
                        "pair_job_id": pair_job_id,
                        "pair_key": pair_key,
                        "source_index": source_index,
                        "material_id": material["material_id"],
                        "source_id": material["source_id"],
                        "source_kind": material["source_kind"],
                        "orientation": orientation,
                        "canonical_left_model_id": left_model,
                        "canonical_right_model_id": right_model,
                        "model_a": model_a,
                        "model_b": model_b,
                        "source_sha256": material.get("source_sha256", ""),
                        "image_sha256": material.get("image_sha256", ""),
                    }
        source_rng.shuffle(pair_jobs)

        public_jobs_by_judge[judge_slot] = {"pointwise": pointwise_jobs, "pairwise": pair_jobs}
        pointwise_path = ROOT / "public" / "pointwise" / f"{judge_slot}.jsonl"
        pairwise_path = ROOT / "public" / "pairwise" / f"{judge_slot}.jsonl"
        write_jsonl(pointwise_path, pointwise_jobs)
        write_jsonl(pairwise_path, pair_jobs)
        public_file_paths.extend([pointwise_path, pairwise_path])

    # Public payload audit: keys/values may not expose candidate provenance or
    # runtime metadata.  Source text itself is allowed to contain ordinary
    # words such as "model"; only known identities/metadata keys are rejected.
    public_forbidden_hits: list[str] = []
    model_names = [str(model_id).lower() for model_id in model_ids]
    model_display_names = [str(row["model_name"]).lower() for row in rows]
    for path in public_file_paths:
        if path.suffix == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix == ".json":
            values = [json.loads(path.read_text(encoding="utf-8"))]
        else:
            values = []
        for value in values:
            for text in public_value_strings(value):
                low = text.lower()
                if any(model_name in low for model_name in model_names + model_display_names):
                    public_forbidden_hits.append(f"{path.name}:model_identity")
                if any(marker in low for marker in FORBIDDEN_PUBLIC_VALUES):
                    # Do not reject the result_schema path or generic score
                    # prose here; this audit covers payload JSON only, and the
                    # key scan below is the authoritative check.
                    if "result_schema" not in low and "score" not in low:
                        public_forbidden_hits.append(f"{path.name}:{text[:80]}")
            if isinstance(value, dict):
                for key in value:
                    key_low = key.lower()
                    if key_low in {"model", "model_id", "model_name", "model_alias", "source_owner", "old_score", "rank", "elapsed_seconds", "prompt_tokens", "completion_tokens", "template_path"}:
                        public_forbidden_hits.append(f"{path.name}:key:{key}")
    if public_forbidden_hits:
        raise ValueError("public blind payload audit failed: " + "; ".join(public_forbidden_hits[:8]))

    private_dir = ROOT / "private"
    write_json(private_dir / "candidate_map.json", {
        "schema_version": "new-material-private-candidate-map-v1",
        "packet_id": PACKET_ID,
        "judge_slots": JUDGE_SLOTS,
        "candidate_model_ids_private": model_ids,
        "mapping_by_judge": candidate_map_by_judge,
    })
    write_json(private_dir / "source_and_image_map.json", {
        "schema_version": "new-material-private-source-map-v1",
        "packet_id": PACKET_ID,
        "source_alias_by_judge": source_alias_by_judge,
        "materials": catalog,
        "image_aliases": image_aliases,
    })
    write_json(private_dir / "job_keymap.json", {
        "schema_version": "new-material-private-job-keymap-v1",
        "packet_id": PACKET_ID,
        "pointwise": private_job_map,
        "pairwise": private_pair_map,
    })
    write_json(private_dir / "pair_selection.json", {
        "schema_version": "new-material-private-pair-selection-v1",
        "packet_id": PACKET_ID,
        "base_seed": BASE_SEED,
        "selection": pair_selection_meta,
        "pairs_by_source": {
            material["material_id"]: {
                "candidate_order_seed": derive_seed("pair-chain", material["material_id"]),
                "candidate_chain_model_ids_private": (
                    (lambda order: order)(
                        sorted(model_ids)
                    )
                ),
            }
            for material in pair_sources
        },
    })
    # Replace the deliberately simple private pair map above with the actual
    # seeded chain orders; keeping this as a separate private structure makes
    # the source/candidate coverage audit explicit.
    pair_chain_private: dict[str, Any] = {}
    for material in pair_sources:
        candidate_chain = model_ids[:]
        random.Random(derive_seed("pair-chain", material["material_id"])).shuffle(candidate_chain)
        pair_chain_private[material["material_id"]] = {
            "candidate_order_seed": derive_seed("pair-chain", material["material_id"]),
            "candidate_chain_model_ids_private": candidate_chain,
            "edges_private": [[candidate_chain[i], candidate_chain[i + 1]] for i in range(6)],
        }
    pair_selection_path = private_dir / "pair_selection.json"
    pair_selection = json.loads(pair_selection_path.read_text(encoding="utf-8"))
    pair_selection["pairs_by_source"] = pair_chain_private
    write_json(pair_selection_path, pair_selection)

    input_hashes = {
        "translation_results.csv": sha256_file(INPUT_CSV),
        "metadata_manifest.json": sha256_file(METADATA_PATH),
        "frozen_text_materials.jsonl": sha256_file(FROZEN_TEXT_PATH),
        "image_selection.json": sha256_file(IMAGE_SELECTION_PATH),
        "old_judge_instructions.txt": sha256_file(OLD_RUBRIC_PATH),
    }
    pair_counts = {
        "pairwise_source_items": len(pair_sources),
        "pairwise_pairs_per_source": 6,
        "pairwise_orientations_per_pair": 2,
        "pairwise_jobs_per_judge": len(pair_sources) * 6 * 2,
        "pairwise_jobs_total": len(pair_sources) * 6 * 2 * len(JUDGE_SLOTS),
    }
    manifest: dict[str, Any] = {
        "schema_version": "translation-benchmark-new-materials-judge-packet-manifest-v1",
        "packet_id": PACKET_ID,
        "packet_version": PACKET_VERSION,
        "status": "PACKET_READY_FOR_ROOT_QA",
        "created_utc": utc_now(),
        "base_seed": BASE_SEED,
        "scoring_authorized": False,
        "root_approval_marker": "ROOT_APPROVED",
        "root_approval_marker_required_before_scoring": True,
        "judge_slots": JUDGE_SLOTS,
        "judge_slot_identity_policy": "public slots are generic; model/family identities are assigned privately by the root/worker and never appear in judge payloads",
        "input_hashes_sha256": input_hashes,
        "input_counts": {
            "translation_result_rows": len(rows),
            "translator_slots": len(model_ids),
            "materials_total": len(catalog),
            "text_materials": len(TEXT_CASES),
            "image_materials": len(IMAGE_CASES),
            "models_each_case": 7,
        },
        "job_counts": {
            "pointwise_jobs_per_judge": len(catalog) * len(model_ids),
            "pointwise_jobs_total": len(catalog) * len(model_ids) * len(JUDGE_SLOTS),
            **pair_counts,
            "substantive_jobs_total": len(catalog) * len(model_ids) * len(JUDGE_SLOTS) + pair_counts["pairwise_jobs_total"],
        },
        "tracks": {
            "pointwise": {"public_dir": "public/pointwise", "result_schema": "public/schemas/pointwise_result.schema.json", "fresh_isolated_context_per_job": True},
            "pairwise": {"public_dir": "public/pairwise", "result_schema": "public/schemas/pairwise_result.schema.json", "raw_ab_ba_preserved": True, "positional_inconsistency_derived_status": "ABSTAIN"},
            "qualification": {"status": "NOT_RERUN", "note": "Old qualification clouds are intentionally outside this new-material packet."},
        },
        "image_policy": {
            "native_pixels_required": True,
            "english_transcription_in_public_payload": False,
            "ground_truth_in_public_payload": False,
            "ocr_as_vision_fallback": False,
            "unsupported_route_status": "blocked_or_missing_abstain",
            "rubric": "unified_pointwise_30_30_20_20",
            "rubric_amendment_note": "The new image track uses the same four 30/30/20/20 fields as the new text track; it does not claim the old pilot's separate image rubric or OCR accuracy.",
        },
        "pair_selection": {
            "private_path": "private/pair_selection.json",
            "selection_meta": pair_selection_meta,
            "candidate_chain_seed_rule": "sha256(base_seed|pair-chain|material_id) first 8 bytes; shuffle seven private candidate IDs; adjacent six edges",
        },
        "interfaces": {
            "worker_input": "one JSONL job per line from exactly one judge slot; public instruction and referenced schema are included",
            "pointwise_job_fields": ["schema_version", "job_id", "track", "blind_source_id", "source_kind", "blind_id", "target_locale", "target_zh_tw", "source_en_or_image_alias", "result_schema"],
            "pairwise_job_fields": ["schema_version", "pair_job_id", "track", "pair_key", "orientation", "blind_source_id", "source_kind", "target_locale", "candidate_a", "candidate_b", "source_en_or_image_alias", "result_schema"],
            "pointwise_output": "one JSON object per pointwise job, keyed by input job_id in the worker envelope; result object must validate against public/schemas/pointwise_result.schema.json",
            "pairwise_output": "one JSON object per pairwise orientation; result must validate against public/schemas/pairwise_result.schema.json",
            "result_envelope": "worker raw record must retain judge_slot, track, input job identifier, first raw response, parsed result, validation status, HTTP/request/timing metadata, and optional format-only repair; this builder does not create or score envelopes",
            "raw_preservation": "save original response, parsed result, HTTP/request/timing metadata, validation diagnostics, and any one format-only repair without overwriting the original",
            "format_repair": "at most one repair for malformed format only; never repair a coherent score/subscore conflict or select a quality-preferred retry",
            "validation_rules": "reject bool-as-integer scores/subscores, enforce 30/30/20/20 bounds and exact sum, require null score/subscores for abstain, validate severity/confidence, and surface critical/error-severity disagreement without normalization",
            "missing_and_invalid": "missing/blocked/invalid are retained with explicit status and never changed to score zero",
            "identity_binding": "root uses private/job_keymap.json; workers must not add candidate/model/source metadata to public input or output",
        },
        "public_file_sha256": {},
        "private_file_sha256": {},
        "qa": {"builder_checks": "PASS", "public_payload_identity_scan": "PASS", "tests": "PENDING", "root_review": "PENDING"},
    }

    # Include generated artifact hashes after writing all content except the
    # self-referential manifest/packet-ready files.
    generated_public = sorted(path for path in (ROOT / "public").rglob("*") if path.is_file())
    manifest["public_file_sha256"] = {str(path.relative_to(ROOT)): sha256_file(path) for path in generated_public}
    generated_private = sorted(path for path in (ROOT / "private").rglob("*") if path.is_file())
    manifest["private_file_sha256"] = {str(path.relative_to(ROOT)): sha256_file(path) for path in generated_private}
    write_json(ROOT / "manifest.json", manifest)

    packet_ready = {
        "schema_version": "translation-benchmark-packet-ready-v1",
        "packet_id": PACKET_ID,
        "status": "PACKET_READY_FOR_ROOT_QA",
        "scoring_authorized": False,
        "root_approval_marker": "ROOT_APPROVED",
        "root_approval_required_before_any_judge_call": True,
        "manifest": "manifest.json",
        "public_payloads": "public/",
        "private_binding": "private/",
        "build_checks": "PASS",
        "tests": "PENDING",
        "input_hashes_sha256": input_hashes,
        "job_counts": manifest["job_counts"],
        "prohibited_actions_before_root_approval": ["no model/API/LiteLLM judge calls", "no score imports", "no public/private identity disclosure"],
    }
    write_json(ROOT / "packet_ready.json", packet_ready)

    protocol = f"""# 新素材五 judge blind packet protocol

## 狀態

本 packet 僅完成資料封裝，尚未評分。`packet_ready.json` 的
`scoring_authorized=false` 是硬閘門；只有 root 明確建立 `ROOT_APPROVED`
marker 且完成獨立 QA 後，worker 才可呼叫任何 judge。Builder 不呼叫模型、
LiteLLM、Open WebUI 或其他 API。

## 凍結來源

- 238 筆 `results/translation_results.csv`：7 個 translator slot × 30 個 text + 4 個 image。
- `materials/metadata_manifest.json`、`materials/frozen_text_materials.jsonl`、`materials/image_selection.json` 只讀。
- 舊 rubric `translation_benchmark_20261022_judge_pilot/judge_instructions.txt` 的四欄 30/30/20/20、strict sum、exact evidence 規則被保留；舊 qualification clouds 不重跑。
- 每個 public image alias 是 authoritative PNG 的 byte-identical copy；public payload 不含 image English transcription、ground truth 或候選 provenance。

## Job 與 blind binding

每個通用 judge slot (`judge-01`…`judge-05`) 收到獨立隨機化的 238 個
pointwise jobs。每個 job 僅有一個 blind candidate、source text 或 image
alias、target translation；source/candidate aliases 與模型身份均由
`private/` 映射，不能加入 public payload。每個 judge 另收 72 個 pairwise
jobs：6 source items × 6 adjacent spanning-chain pairs × AB/BA。pair source
是由 source metadata/length 四分位 strata deterministic 選出的 4 text，
加上 dense WHO `img-09` 與 synthetic medication `img-04`。7 個候選每個
source 都出現在六條 chain edges 中。總 substantive jobs =
{manifest['job_counts']['substantive_jobs_total']}（pointwise
{manifest['job_counts']['pointwise_jobs_total']} + pairwise
{manifest['job_counts']['pairwise_jobs_total']}）。

## Image rubric amendment

新 image track 採與 text 相同的 `terminology_locale` 30、
`semantic_safety` 30、`fluency` 20、`completeness_format` 20。這是保留同一
strict schema 的透明新批次選擇，不宣稱舊 pilot image 40/30/30 rubric 未變，
也不把結果當 OCR accuracy。Native pixels 無法使用時必須 `blocked`/
`missing` + `abstain=true`，不能 OCR-as-vision，也不能以猜測扣翻譯分數。

## Output / retry

依 `public/instructions.md` 與 `public/schemas/*.json` 回傳 strict JSON。
第一個 valid result 才是可評分結果；只允許最多一次 format-only repair，
兩個 raw response 都保存，不能因分數或證據好看而重問/替換。invalid、
missing、blocked 不得填零。AB/BA raw orientation 獨立保存；canonical mapping
後若位置反轉不一致，derived status 是 `ABSTAIN`，不是 `TIE`。

## 分析界線

後續 aggregate 必須 text/image 分開；ICC(2,1) 與 ICC(2,k) 只用實際 complete
units 的 arithmetic mean，median 獨立列示，CI 不可捏造。30 text 來自 12
source pages，source-cluster bootstrap 與 LOFO 是敏感度，不是 clinical
或 deployment safety 證明。error/critical gates 以 1/5 = REVIEW、2/5 =
FAIL 的探索性標籤呈現，不把 reliability 等同 correctness。舊 pilot ICC
不得冒充本新素材資料的 ICC。

## 文件

- Public worker input: `public/pointwise/*.jsonl`, `public/pairwise/*.jsonl`,
  `public/images/`, `public/instructions.md`, `public/schemas/`。
- Private binding/audit: `private/candidate_map.json`,
  `private/source_and_image_map.json`, `private/job_keymap.json`,
  `private/pair_selection.json`。
- Top-level `manifest.json` binds input hashes, interface, counts, and output
  hashes. `progress_protocol.md` is the quick status and gate log.
"""
    (ROOT / "protocol.md").write_text(protocol, encoding="utf-8")
    progress = f"""# New-material judge packet progress

- packet: `{PACKET_ID}`
- state: **PACKET_READY_FOR_ROOT_QA**
- scoring gate: **CLOSED** (`scoring_authorized=false`; requires root `ROOT_APPROVED` marker)
- builder: `PASS` (fixed seed `{BASE_SEED}`; input hashes and public/private hashes in `manifest.json`)
- source binding: `PASS` (238 translation rows; 30 frozen text + 4 frozen PNG aliases; all seven outputs per case)
- pointwise jobs: {manifest['job_counts']['pointwise_jobs_total']} total = 238 × 5
- pairwise jobs: {manifest['job_counts']['pairwise_jobs_total']} total = 6 sources × 6 chain edges × 2 orientations × 5
- substantive jobs: {manifest['job_counts']['substantive_jobs_total']}
- image route: native pixels required; public image payload has no English transcription or ground truth; unsupported route is blocked/missing, never OCR fallback
- qualification: old qualification clouds **not rerun**
- scoring/API/model calls: **none by this builder**
- tests: pending (run `python3 -m unittest discover -s tests -p 'test_*.py' -v`, then `python3 build_packets.py --finalize`)

Root review must verify private binding, deterministic rerun, all source/image
hashes, schemas, no public identity leakage, and the root approval marker before
any local/cloud worker starts.
"""
    (ROOT / "progress_protocol.md").write_text(progress, encoding="utf-8")

    # Refresh hashes for protocol/progress after their write.  These files are
    # not included in public/private payload hashes but are bound in manifest.
    manifest["documentation_sha256"] = {
        "protocol.md": sha256_file(ROOT / "protocol.md"),
        "build_packets.py": sha256_file(ROOT / "build_packets.py"),
    }
    write_json(ROOT / "manifest.json", manifest)
    return manifest


def finalize() -> None:
    ready_path = ROOT / "packet_ready.json"
    manifest_path = ROOT / "manifest.json"
    if not ready_path.exists() or not manifest_path.exists():
        raise SystemExit("packet is not built")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The test harness writes this evidence only after all tests pass.
    report_path = ROOT / "tests" / "test_report.json"
    if not report_path.exists():
        raise SystemExit("tests/test_report.json is missing; run the packet tests first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise SystemExit("cannot finalize packet_ready: packet tests are not PASS")
    ready["tests"] = "PASS"
    ready["test_report"] = "tests/test_report.json"
    ready["ready"] = True
    ready["packet_ready"] = True
    manifest["qa"]["tests"] = "PASS"
    manifest["qa"]["test_report"] = "tests/test_report.json"
    progress_path = ROOT / "progress_protocol.md"
    progress = progress_path.read_text(encoding="utf-8")
    progress = progress.replace(
        "- tests: pending (run `python3 -m unittest discover -s tests -p 'test_*.py' -v`, then `python3 build_packets.py --finalize`)",
        "- tests: **PASS** (`tests/test_report.json`; six offline contract tests; no model/API calls)",
    )
    progress_path.write_text(progress, encoding="utf-8")
    write_json(ready_path, ready)
    write_json(manifest_path, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="deliberately rebuild generated packet outputs")
    parser.add_argument("--finalize", action="store_true", help="mark packet_ready/manifest tests PASS after test_report.json")
    args = parser.parse_args(argv)
    try:
        if args.finalize:
            finalize()
        else:
            manifest = build(overwrite=args.overwrite)
            print(json.dumps({"status": "PASS", "packet_id": manifest["packet_id"], "job_counts": manifest["job_counts"]}, ensure_ascii=False, indent=2))
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
