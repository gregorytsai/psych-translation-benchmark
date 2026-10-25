#!/usr/bin/env python3
"""Build the redacted, reproducible judge-pilot packet.

This builder intentionally keeps model names, old scores, raw-file paths and
qualification labels out of judge_packet.json.  They are written only to
private manifests for later aggregation.  Existing raw JSONL files are read
without changing them; their assistant message content is copied verbatim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCH = ROOT.parent / "translation_benchmark_20260916_luna"
PACKET_ID = "judge-pilot-20261022-v1"
BLIND_SEED = 20261022

RAW_FILES = {
    "translategemma-4b": "raw_template_audit_tg4_v2.jsonl",
    "translategemma-12b": "raw_template_audit_tg12.jsonl",
    "translategemma-27b": "raw_template_audit_tg27.jsonl",
    "gemma-4-e4b": "raw_template_audit_gemma4_e4b.jsonl",
    "gemma-4-12b": "raw_template_audit_gemma4_12b_v2.jsonl",
    "gemma-4-26b": "raw_template_audit_gemma4_26b.jsonl",
    "gemma-4-31b-qat": "raw_template_audit_gemma4_31b_qat.jsonl",
}

FAMILY = {
    "translategemma-4b": "google-translategemma",
    "translategemma-12b": "google-translategemma",
    "translategemma-27b": "google-translategemma",
    "gemma-4-e4b": "google-gemma",
    "gemma-4-12b": "google-gemma",
    "gemma-4-26b": "google-gemma",
    "gemma-4-31b-qat": "google-gemma",
}

TEXT_CASES = ["panic_disorder", "depression_behavioral_activation", "ssri_pharmacology"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def response_text(row: dict) -> str:
    return row["response"]["choices"][0]["message"]["content"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build_instructions() -> str:
    return r'''# Blind multi-judge pilot instructions (2026-10-22)

You are one independent evaluator in a provisional, non-clinical, English →
Taiwan Traditional Chinese (zh-TW) translation pilot. The packet is blind:
`cand-*` identifiers are random and do not reveal model, size, speed, prior
score, quantization, or family. Do not infer or mention a candidate identity.

Use only the English source and the target translation shown in the task.
There is no reference translation. Judge translation quality, not whether the
clinical advice is appropriate for a particular patient. Do not add safety
advice, a summary, or a rewrite. Preserve the source's uncertainty and
conditions when judging fidelity.

## Pointwise rubric

Give four integer sub-scores that sum to `score_0_100`:

- `terminology_locale`: 0–30. Medical/psychological terms and Taiwan usage;
  Traditional Chinese, not Simplified Chinese.
- `semantic_safety`: 0–30. Meaning, negation, medication names, dose/number,
  units, frequency, timing, conditions, and safety-critical facts.
- `fluency`: 0–20. Natural, clear Taiwan Traditional Chinese register.
- `completeness_format`: 0–20. No omission/addition, and paragraph/list/format
  preservation where applicable.

Do not average away a critical error. Set `critical=true` for a confirmed
medication, dose/number/unit, negation, time/condition, suicide/self-harm,
or similarly safety-critical meaning error. Use `abstain=true` only when the
source or target is genuinely not judgeable; still return the JSON object with
`score_0_100: null` and `subscores: null`.

For every error, prefer exact evidence over a vague explanation:

- `target_exact_quote`: the shortest exact substring in the target that is
  wrong; use `""` for an omission.
- `source_evidence`: an exact or near-exact source phrase and a concise reason.
- `source_span`: the source phrase supporting the error (required for omission
  and useful for every error).
- `target_anchor`: a target phrase showing where an omission belongs (required
  for omission; otherwise optional).

The runner computes target character offsets from `target_exact_quote`; do not
invent offsets. If no exact quote can be located, leave it empty and the runner
will mark the span unmatched. Use one error object per distinct issue. Do not
use a reference translation or old score.

## Required pointwise JSON

Return exactly one JSON object and no Markdown:

{
  "blind_id": "cand-01",
  "abstain": false,
  "score_0_100": 84,
  "subscores": {
    "terminology_locale": 26,
    "semantic_safety": 24,
    "fluency": 18,
    "completeness_format": 16
  },
  "critical": false,
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/omission",
      "target_exact_quote": "",
      "source_evidence": "source phrase is missing from the target",
      "source_span": "source phrase",
      "target_anchor": "nearby target phrase",
      "confidence": 0.88
    }
  ]
}

Use `errors: []` when no error is supported by the source. For a clean
translation, do not invent minor style issues merely to fill the list.

## Qualification JSON

Qualification tasks show two anonymous target variants for one source. Return:

{
  "judgments": [
    {"blind_id": "qual-X", "abstain": false, "score_0_100": 95,
     "subscores": {"terminology_locale": 29, "semantic_safety": 29,
       "fluency": 19, "completeness_format": 18}, "critical": false,
     "errors": []},
    {"blind_id": "qual-Y", "abstain": false, "score_0_100": 55,
     "subscores": {"terminology_locale": 25, "semantic_safety": 10,
       "fluency": 12, "completeness_format": 8}, "critical": true,
     "errors": [{"severity": "critical", "type": "accuracy/<type>",
       "target_exact_quote": "<short exact target quote or empty for omission>",
       "source_evidence": "<source phrase and concise reason>",
       "source_span": "<supporting source phrase>", "target_anchor": "<target context>",
       "confidence": 0.90}]}
  ]
}

Qualification labels (clean/mutated and expected error type) are deliberately
not supplied. Do not guess a hidden answer key; report what the source-target
comparison supports.

## Pairwise JSON

For one AB or BA task, return exactly:

{"winner": "A", "confidence": 0.76, "critical_override": false,
 "reason_codes": ["better_semantic_fidelity"]}

Use `winner: "TIE"` if neither is meaningfully better. The two orientations
are evaluated separately; do not assume the other orientation or reveal a
candidate identity.
'''


def build() -> None:
    # Existing destination was checked before this builder was invoked. Refuse
    # to silently replace a packet if a later run points at a populated folder.
    packet_path = ROOT / "judge_packet.json"
    if packet_path.exists():
        raise SystemExit(f"refusing to overwrite existing {packet_path}")

    inputs = {
        row["id"]: row
        for row in (json.loads(line) for line in (BENCH / "inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    }
    raw_by_model = {model: read_jsonl(BENCH / filename) for model, filename in RAW_FILES.items()}
    rows_by_model = {model: {row["id"]: row for row in rows} for model, rows in raw_by_model.items()}
    for model, rows in rows_by_model.items():
        missing = set(TEXT_CASES) - set(rows)
        if missing:
            raise SystemExit(f"{model}: missing corrected raw cases {sorted(missing)}")

    order = list(RAW_FILES)
    random.Random(BLIND_SEED).shuffle(order)
    blind_for_model = {model: f"cand-{index:02d}" for index, model in enumerate(order, 1)}

    # Copy only raw assistant content into the public blind packet. Do not
    # carry request metadata, model names, historical scores, timing or paths.
    pointwise_items = []
    blind_jsonl = []
    for case_index, case_id in enumerate(TEXT_CASES, 1):
        source = inputs[case_id]
        candidates = []
        for model in order:
            blind_id = blind_for_model[model]
            translation = response_text(rows_by_model[model][case_id])
            candidates.append({"blind_id": blind_id, "target_zh_tw": translation})
            blind_jsonl.append({
                "item_id": f"src-{case_index:02d}",
                "source_en": source["english"],
                "blind_id": blind_id,
                "target_zh_tw": translation,
                "input_type": "text",
            })
        pointwise_items.append({
            "item_id": f"src-{case_index:02d}",
            "source_label": f"source-{case_index:02d}",
            "domain": source["domain"],
            "input_type": "text",
            "source_en": source["english"],
            "target_locale": "zh-TW",
            "candidates": candidates,
        })

    # Qualification order is randomized per item, but the hidden labels stay
    # only in answer_key.private.json.
    qualification_bank = json.loads((ROOT / "qualification_bank.private.json").read_text(encoding="utf-8"))
    q_rng = random.Random(BLIND_SEED + 17)
    qualification_public = []
    answer_key = {"packet_id": PACKET_ID, "qualification": {}, "pairwise": {}}
    for item in qualification_bank:
        mutated_side = q_rng.choice(["A", "B"])
        clean_side = "B" if mutated_side == "A" else "A"
        variants = {
            "A": item["mutated_target_zh_tw"] if mutated_side == "A" else item["clean_target_zh_tw"],
            "B": item["mutated_target_zh_tw"] if mutated_side == "B" else item["clean_target_zh_tw"],
        }
        qualification_public.append({
            "qualification_id": item["qualification_id"],
            "source_en": item["source_en"],
            "target_locale": "zh-TW",
            "variants": [
                {"blind_id": "qual-A", "target_zh_tw": variants["A"]},
                {"blind_id": "qual-B", "target_zh_tw": variants["B"]},
            ],
        })
        answer_key["qualification"][item["qualification_id"]] = {
            "clean_blind_id": f"qual-{clean_side}",
            "mutated_blind_id": f"qual-{mutated_side}",
            "expected_behavior": item["expected_behavior"],
            "expected_error_type": item["expected_error_type"],
            "expected_severity": item["expected_severity"],
            "mutation_type": item["mutation_type"],
            "sentinel": item["sentinel"],
        }

    pair_specs = [("p-01", "cand-01", "cand-02"), ("p-02", "cand-03", "cand-04"), ("p-03", "cand-05", "cand-06")]
    pairwise_public = []
    for pair_id, left, right in pair_specs:
        for orientation, a_id, b_id in [("AB", left, right), ("BA", right, left)]:
            source_item = pointwise_items[0]
            text_by_id = {candidate["blind_id"]: candidate["target_zh_tw"] for candidate in source_item["candidates"]}
            pairwise_public.append({
                "pair_id": pair_id,
                "item_id": source_item["item_id"],
                "orientation": orientation,
                "source_en": source_item["source_en"],
                "candidate_a": {"blind_id": a_id, "target_zh_tw": text_by_id[a_id]},
                "candidate_b": {"blind_id": b_id, "target_zh_tw": text_by_id[b_id]},
            })
            answer_key["pairwise"].setdefault(pair_id, {})[orientation] = {
                "canonical_left": left,
                "canonical_right": right,
                "orientation_a": a_id,
                "orientation_b": b_id,
            }

    image_case = json.loads((BENCH / "image_case.json").read_text(encoding="utf-8"))
    image_rows = []
    for model in order:
        row = rows_by_model[model].get("image_clinical_note")
        if row:
            image_rows.append({"blind_id": blind_for_model[model], "target_zh_tw": response_text(row)})

    instructions = build_instructions()
    (ROOT / "judge_instructions.txt").write_text(instructions, encoding="utf-8")
    instruction_hash = sha256_bytes(instructions.encode("utf-8"))

    packet = {
        "packet_id": PACKET_ID,
        "packet_version": 1,
        "status": "PROVISIONAL_PILOT",
        "created_local_date": "2026-10-22",
        "source_policy": "Three existing self-written English text sources and their corrected OAI/template raw assistant outputs only.",
        "source_language": "English",
        "target_language": "Traditional Chinese",
        "target_locale": "zh-TW",
        "domain": "psychology / psychiatry / psychopharmacology",
        "blind_seed": BLIND_SEED,
        "judge_instructions_sha256": instruction_hash,
        "blinding": {
            "candidate_ids": [f"cand-{i:02d}" for i in range(1, 8)],
            "model_names_in_packet": False,
            "existing_scores_in_packet": False,
            "candidate_paths_in_packet": False,
            "candidate_order_fixed_for_replay": True,
        },
        "qualification": {
            "count": len(qualification_public),
            "items": qualification_public,
            "answer_key_is_private": True,
        },
        "pointwise": {
            "count": len(pointwise_items) * len(order),
            "source_item_count": len(pointwise_items),
            "items": pointwise_items,
        },
        "pairwise": {
            "count": len(pairwise_public),
            "tasks": pairwise_public,
        },
        "image_smoke": {
            "status": "OPTIONAL_SINGLE_IMAGE_SMOKE; NOT IN TEXT RANKING",
            "item_id": "image-smoke-01",
            "source_en": image_case["english_exact"],
            "target_locale": "zh-TW",
            "image_sha256": sha256_file(BENCH / "image_clinical_note.png"),
            "image_attachment_hint": "Attach translation_benchmark_20260916_luna/image_clinical_note.png separately if the judge supports images.",
            "candidates": image_rows,
            "ocr_reference_status": "not_available_in_packet",
        },
        "judge_output_contract": {
            "pointwise": "judge_instructions.txt#pointwise",
            "qualification": "judge_instructions.txt#qualification",
            "pairwise": "judge_instructions.txt#pairwise",
            "json_only": True,
            "abstain_allowed": True,
        },
    }

    write_json(packet_path, packet)
    write_jsonl(ROOT / "raw" / "candidate_translations.blind.jsonl", blind_jsonl)
    write_json(ROOT / "private" / "answer_key.private.json", answer_key)

    mapping = {
        "packet_id": PACKET_ID,
        "blind_seed": BLIND_SEED,
        "mapping": {
            blind_for_model[model]: {
                "model_alias": model,
                "candidate_family": FAMILY[model],
                "raw_file": str(BENCH / RAW_FILES[model]),
                "raw_file_sha256": sha256_file(BENCH / RAW_FILES[model]),
            }
            for model in order
        },
    }
    write_json(ROOT / "private" / "candidate_mapping.private.json", mapping)

    source_manifest = {
        "packet_id": PACKET_ID,
        "raw_extraction": "response.choices[0].message.content copied verbatim from corrected template audit JSONL",
        "raw_files": [
            {
                "model_alias_private": model,
                "path_private": str(BENCH / RAW_FILES[model]),
                "sha256": sha256_file(BENCH / RAW_FILES[model]),
                "text_case_ids": TEXT_CASES,
                "contains_existing_scores": False,
            }
            for model in order
        ],
        "source_items": [
            {"item_id": f"src-{index:02d}", "original_id_private": case_id, "source_sha256": sha256_bytes(inputs[case_id]["english"].encode("utf-8"))}
            for index, case_id in enumerate(TEXT_CASES, 1)
        ],
    }
    write_json(ROOT / "private" / "source_manifest.private.json", source_manifest)

    public_manifest = {
        "packet_id": PACKET_ID,
        "candidate_count": len(order),
        "candidate_ids": [f"cand-{i:02d}" for i in range(1, 8)],
        "source_item_count": len(TEXT_CASES),
        "pointwise_count": len(TEXT_CASES) * len(order),
        "qualification_count": len(qualification_public),
        "pairwise_task_count": len(pairwise_public),
        "raw_output_policy": "blind assistant text only; model names, scores, paths and timings removed",
    }
    write_json(ROOT / "candidate_manifest.public.json", public_manifest)

    # A manual Gemini prompt references the packet by attachment rather than
    # embedding private labels or requiring a huge shell interpolation.
    gemini_prompt = f'''You are the external blind judge for packet `{PACKET_ID}`.\n\nAttach and read `judge_packet.json` and `judge_instructions.txt` from the pilot directory. Evaluate every qualification item, every pointwise candidate (21 total), and every pairwise task (AB and BA). Use the exact JSON contracts in `judge_instructions.txt`; return one JSON object with keys `judge_id`, `model_id`, `qualification`, `pointwise`, and `pairwise`, and no Markdown. Set `judge_id` to `gemini-3.8-flash` and `model_id` to your exact serving model identifier if known. Do not mention or infer candidate model identities.\n\nFor pointwise, include `item_id`, `blind_id`, and the required judgment fields. For qualification, include `qualification_id` and a `judgments` array. For pairwise, include `pair_id`, `orientation`, and the required winner object. If a task is not judgeable, use `abstain=true` and explain only in the JSON `notes` field. Keep the result provisional; there is no human gold and no safety clearance.\n\nDo not use any reference translation or prior score. Preserve exact evidence strings so a separate program can compute target offsets. Return only valid UTF-8 JSON.\n'''
    (ROOT / "gemini_prompt.txt").write_text(gemini_prompt, encoding="utf-8")

    print(json.dumps({
        "packet": str(packet_path),
        "pointwise_count": packet["pointwise"]["count"],
        "qualification_count": packet["qualification"]["count"],
        "pairwise_count": packet["pairwise"]["count"],
        "blind_order_private": order,
        "instruction_sha256": instruction_hash,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
