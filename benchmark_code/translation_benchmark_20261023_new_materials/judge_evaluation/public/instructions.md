# New-material blind judge instructions (2026-10-23)

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
