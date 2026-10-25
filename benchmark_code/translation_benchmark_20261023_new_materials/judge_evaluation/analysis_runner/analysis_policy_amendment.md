# Analysis policy amendment (derived aggregation only)

This note is an analysis-layer pre-specification. It does not amend the frozen
public packet, private binding map, manifest, pointwise rubric, or raw records.

## Queue-state gate

While the requested primary jobs are still arriving, reports are descriptive
only. No model ranking, candidate mean comparison, ICC, Kendall W, LOFO order,
or bootstrap interval is published from an early/random queue snapshot. The
refresh command may be called after each atomic raw batch; it distinguishes
`not_attempted`, malformed, invalid, blocked, and valid records and never
changes a missing score to zero.

Once every requested primary job has a terminal raw attempt (valid, explicitly
blocked/unsupported, or preserved invalid/failure), the importer may compute a
common-case exploratory analysis even if some cells remain unusable. The
terminal-attempt status and exclusions must be shown separately from the
complete fixed-panel gate.

## Common-case exploratory analysis after terminal coverage

Candidate comparisons use only source items for which all seven candidates and
the requested fixed panel are valid, with identical case weights. Excluded
case IDs and reasons are reported. A preliminary text analysis requires at
least 12 common text items spanning at least 6 source pages; it is labeled
limited and does not imply that 30 text items were observed. If that threshold
is not met, text remains descriptive only.

The image fixed panel is Qwen, Nemotron, GPT, and Gemini (`k=4`); Mistral's
known missing native-image route is not imputed. A descriptive common image
comparison requires at least 2 images with all four judges. Image cluster CIs
remain unavailable/unreliable here because the four images form only three
shared source groups. No 238-row independence claim is made.

ICC(2,1) and ICC(2,k) are computed only on an explicitly named balanced
candidate-by-item matrix within the stated common-case subset, using the
arithmetic panel mean interpretation; median is reported separately. The
fixed-panel complete gate remains the preferred/full analysis. ICC is
reliability/consistency evidence, not correctness proof, and is not used to
discard a judge or select a better-looking score.

Kendall W uses tie correction. The principal W is a system-level 7-candidate
rank concordance after per-judge candidate means; candidate×item unit-level W
and per-item W distribution are reported separately. If tie correction makes
the denominator zero (for example all ranks constant), W is NA rather than a
fabricated value.

## Retry and evidence policy

The first schema-valid attempt is selected by attempt order. A same-prompt
infrastructure retry is allowed only when the earlier attempt contains no
parsed candidate judgment and is labeled operational recovery. A format-only
repair must be explicitly marked in raw provenance; a parsed sum/schema error
is not silently converted into either category. No quality-selected retry is
permitted. All invalid raw attempts remain in `invalid_annotations.csv`.
When execution metadata is present, a parsed JSON object is eligible only if
its raw status/HTTP/finish/exit, model identity, and applicable native
template/image gates pass; a failed transport or model route cannot be
rescued by a structurally valid score.

Image visible-English matches use the frozen `provisional_reference_text`
privately for diagnostic purposes only, preserving its machine-extracted / AI-
visual-checked / not-human-gold status. They are flags, not proof of an error;
the target text and raw judge outputs remain immutable.

The image CSV preserves attribution: `img-01` and `img-04` are authored
synthetic renders with complete visible labels, whereas `img-09` and `img-10`
are WHO PDF renders with provisional machine/AI reference text. This analysis
does not claim OCR character-accuracy ground truth; see the frozen
`materials_audit.md` for source attribution and rights notes.
