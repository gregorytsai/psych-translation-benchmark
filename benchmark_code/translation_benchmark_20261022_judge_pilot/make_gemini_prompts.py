#!/usr/bin/env python3
"""Render blind Gemini copy-paste batches from the public judge packet.

The renderer deliberately reads only ``judge_packet.json`` and never opens
the private mapping or qualification answer key.  It writes qualification,
source-split pointwise, and independent AB/BA pairwise Markdown prompts.  A
human operator records the model version visible in the Gemini UI separately;
the prompt does not ask Gemini to self-report or authenticate its model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
PACKET = ROOT / "judge_packet.json"
INSTRUCTIONS = ROOT / "judge_instructions.txt"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def common_intro(packet_hash: str, batch_id: str, track: str, instructions_hash: str) -> str:
    return f"""# Gemini blind judge pilot — {batch_id}

Copy this whole document into a fresh Gemini chat. This is a provisional,
non-clinical English → Taiwan Traditional Chinese translation assessment.

- Packet: `judge-pilot-20261022-v1`
- Packet SHA-256 (raw `judge_packet.json`): `{packet_hash}`
- Instructions SHA-256: `{instructions_hash}`
- Track: `{track}`
- Batch: `{batch_id}`

Use only the source and anonymous `cand-*`/`qual-*` text below. Do not infer or
mention model names, model families, model size, speed, prior scores, expected
answers, or a reference translation. Do not add clinical advice, a summary, or
a rewrite. Keep the output provisional; it is not human gold or a safety
clearance. Do not browse, call tools, or consult outside/generated reference
material. Locale-appropriate terminology variants or synonyms are not an
automatic error; judge whether the source meaning and Taiwan locale are
preserved.

Return only the requested UTF-8 JSON object, with no Markdown fences and no
reasoning. If a task is not judgeable, return `abstain: true`, null score and
subscores, and a short `notes` value. Do not fabricate a judgment when the
chat is truncated. The operator—not the prompt or a model self-report—must
record the exact model/version shown in the Gemini UI after the chat. Do not
guess or fill the operator-only metadata fields: `judge_id`,
`model_label_from_ui`, and `task_id`; the operator adds them after the chat.

"""


POINTWISE_RUBRIC = """## Pointwise rubric

For each candidate, return integer subscores summing to `score_0_100`:

- `terminology_locale`: 0–30 (medical/psychological terminology and Taiwan usage)
- `semantic_safety`: 0–30 (meaning, negation, medication, dose/number/unit, frequency, time, conditions, safety facts)
- `fluency`: 0–20 (natural and clear Taiwan Traditional Chinese)
- `completeness_format`: 0–20 (no omission/addition; preserve paragraphs/lists/format)

Set `critical: true` for a confirmed medication, dose/number/unit, negation,
time/condition, suicide/self-harm, or similarly safety-critical meaning error.
Use one error object per distinct issue. `target_exact_quote` must be an exact
target substring or `""` for an omission; use `source_evidence`, `source_span`,
and (for omissions) `target_anchor` to make the evidence auditable. Do not
invent minor style errors to fill the list.

Each pointwise record must have this shape:

```json
{"item_id":"src-01","source_id":"source-01","source_text_sha256":"...","blind_id":"cand-01","abstain":false,"score_0_100":84,"subscores":{"terminology_locale":26,"semantic_safety":24,"fluency":18,"completeness_format":16},"critical":false,"errors":[{"severity":"major","type":"accuracy/omission","target_exact_quote":"","source_evidence":"...","source_span":"...","target_anchor":"...","confidence":0.88}]}
```

"""


def qualification_prompt(packet: dict[str, Any], packet_hash: str, instructions_hash: str) -> str:
    records: list[dict[str, Any]] = []
    for item in packet["qualification"]["items"]:
        digest = source_hash(item["source_en"])
        records.append({"qualification_id": item["qualification_id"], "source_id": item["qualification_id"], "source_en": item["source_en"], "source_text_sha256": digest, "target_locale": item["target_locale"], "variants": item["variants"]})
    return common_intro(packet_hash, "qualification-all", "qualification", instructions_hash) + """## Qualification task

For every item, judge both anonymous variants independently. The clean/mutated
labels and expected error type are private; do not guess them. Return exactly:

```json
{"packet_id":"judge-pilot-20261022-v1","packet_sha256":"...","judge_id":"<operator fills>","model_label_from_ui":"<operator fills exact visible UI label>","task_id":"qualification-all","track":"qualification","batch_id":"qualification-all","records":[{"qualification_id":"q-01","source_id":"q-01","source_text_sha256":"...","judgments":[{"blind_id":"qual-A","abstain":false,"score_0_100":95,"subscores":{"terminology_locale":29,"semantic_safety":29,"fluency":19,"completeness_format":18},"critical":false,"errors":[]}]}]}
```

""" + json_block({"qualification_items": records}) + "\n"


def pointwise_prompt(packet: dict[str, Any], packet_hash: str, instructions_hash: str, item: dict[str, Any]) -> str:
    source_digest = source_hash(item["source_en"])
    tasks = {
        "item_id": item["item_id"],
        "source_id": item.get("source_label", item["item_id"]),
        "source_text_sha256": source_digest,
        "domain": item["domain"],
        "input_type": item["input_type"],
        "source_en": item["source_en"],
        "target_locale": item["target_locale"],
        "candidates": item["candidates"],
    }
    return common_intro(packet_hash, f"pointwise-{item['item_id']}", "pointwise", instructions_hash) + POINTWISE_RUBRIC + "## Task\n\n" + json_block(tasks) + "\n\nReturn one `records` entry for every candidate in this item. Do not omit a candidate.\n\n" + "```json\n{\"packet_id\":\"judge-pilot-20261022-v1\",\"packet_sha256\":\"...\",\"judge_id\":\"<operator fills>\",\"model_label_from_ui\":\"<operator fills exact visible UI label>\",\"task_id\":\"pointwise-" + item["item_id"] + "\",\"track\":\"pointwise\",\"batch_id\":\"pointwise-" + item["item_id"] + "\",\"records\":[...]}\n```\n"


def pairwise_prompt(packet: dict[str, Any], packet_hash: str, instructions_hash: str, orientation: str) -> str:
    tasks = [task for task in packet["pairwise"]["tasks"] if task["orientation"] == orientation]
    compact = []
    for task in tasks:
        compact.append({
            "pair_id": task["pair_id"],
            "item_id": task["item_id"],
            "source_id": task["item_id"],
            "source_text_sha256": source_hash(task["source_en"]),
            "orientation": task["orientation"],
            "source_en": task["source_en"],
            "candidate_a": task["candidate_a"],
            "candidate_b": task["candidate_b"],
        })
    return common_intro(packet_hash, f"pairwise-{orientation}", "pairwise", instructions_hash) + f"""## Pairwise task

Judge only translation quality. Return one independent record per task:

```json
{{"packet_id":"judge-pilot-20261022-v1","packet_sha256":"...","judge_id":"<operator fills>","model_label_from_ui":"<operator fills exact visible UI label>","task_id":"pairwise-{orientation}","track":"pairwise","batch_id":"pairwise-{orientation}","records":[{{"pair_id":"p-01","item_id":"src-01","source_id":"src-01","source_text_sha256":"...","orientation":"{orientation}","winner":"A","confidence":0.76,"critical_override":false,"reason_codes":["better_semantic_fidelity"]}}]}}
```

Use `winner: "TIE"` only when neither is meaningfully better. Do not assume
that the opposite orientation was or will be shown; this chat contains only
one orientation. Preserve the displayed `candidate_a`/`candidate_b` IDs.

""" + json_block({"pairwise_tasks": compact}) + "\n"


def render(packet_path: Path, output_dir: Path) -> list[Path]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_hash = file_sha256(packet_path)
    instructions_hash = file_sha256(INSTRUCTIONS)
    declared_instructions_hash = packet.get("judge_instructions_sha256")
    if isinstance(declared_instructions_hash, str) and declared_instructions_hash.lower() != instructions_hash:
        raise ValueError("shared judge instructions hash does not match the frozen packet declaration")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "qualification_all.md": qualification_prompt(packet, packet_hash, instructions_hash),
        "pairwise_AB.md": pairwise_prompt(packet, packet_hash, instructions_hash, "AB"),
        "pairwise_BA.md": pairwise_prompt(packet, packet_hash, instructions_hash, "BA"),
    }
    for item in packet["pointwise"]["items"]:
        outputs[f"pointwise_{item['item_id']}.md"] = pointwise_prompt(packet, packet_hash, instructions_hash, item)
    paths = []
    for name, content in outputs.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.packet, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
