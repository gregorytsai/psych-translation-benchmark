# Public release boundary / 公開範圍

This release reports the frozen 2026-10-23 translation candidates and the final
2026-10-25 post-recovery evaluation. Earlier pilot scores are not pooled into the
headline results.

本專案公開固定的 238 份翻譯、最新逐筆評分、彙整表、可公開素材及完整封存程式。
原始結果中的本機絕對路徑欄位移除；程式中的原工作站根路徑改為通用的
`/benchmark`。原始雜湊及公開副本雜湊見 `docs/data/source_file_provenance.json`。
模型輸出的翻譯內容與數值不因這個處理而更改。

Included: all 238 measured translations; all final score/status tables, including
the invalid score and unsupported-by-design cells; code, templates, public source
attribution, 30 frozen text items and the two explicitly identified synthetic
images (img-01 and img-04). The two WHO source PNGs (img-09 and img-10) are not
mirrored in this release: official page-one PDF links and frozen SHA-256
references are provided instead. The 14 WHO-derived model translations remain
in the measured results as unofficial adaptations under CC BY-NC-SA 3.0
Unported; the original WHO works retain CC BY-NC-SA 3.0 IGO.

Excluded: credentials, authentication stores, browser profiles, machine-specific
logs, process identifiers, model weights, the unsanitized full archive and older
translation/evaluation outputs. The code archive contains historical tools as
well as the current pipeline; historical tools do not supply the latest ranking.

The code includes explicit synthetic test credentials and `example.invalid`
fixtures. These are not live credentials. Never enter real credentials into the
public repository. The original experiment's private blind-ID map is an
experimental randomization map, not a credential; the final public CSVs already
identify the candidate model for every measured translation.

Public tables support offline recomputation. Running the original experiment
requires appropriate model weights, a compatible runtime, adapted local paths,
and cloud accounts where applicable. The public archive is not a turnkey
deployment and has not been validated as a live LiteLLM/Open WebUI service.
