# 心理健康英中翻譯實測 / Mental-health translation benchmark

這是 GitHub Pages 靜態網站的可重建來源。網站預設讀取已提交的 `docs/data/` frozen CSV；只有在明確傳入 `--results` 時才會讀取外部 review 目錄。`build_site.py` 只輸出明確允許公開的欄位。

```powershell
python .\build_site.py
```

輸出寫到 `docs/`：雙語首頁、七個雙語模型頁、方法、可搜尋的 238 筆翻譯、完整結果下載及授權。首頁用總覽圖比較文字分數、圖片分數與模型載入後的回應時間。所有實際翻譯保留原樣。

若存在 `docs/data/editorial_sections.json`，其以頁面檔名為鍵、以已審核 HTML 片段為值；一般頁面會在 `</main>` 前插入，下載與授權頁則以對應片段取代佔位內容。檔案不存在時不影響重建。

公開站刻意不包含 raw attempt、絕對路徑、PID、私有 binding、模型權重或未整理的 logs。分數是 common-case panel 的描述性觀察；generation 暖請求秒數也只適用於這次固定環境。

[繁體中文](https://gregorytsai.github.io/psych-translation-benchmark/) · [English](https://gregorytsai.github.io/psych-translation-benchmark/index-en.html)

## Scope / 範圍

7 models, 30 English text items and 4 images, 238 frozen translations. The
candidate outputs were generated on 2026-10-23; final evaluation recovery ended
on 2026-10-25. The text comparison uses the same 29 items and 5 AI judges.
One invalid Qwen score excludes text-15 from every model's balanced comparison;
all translations remain downloadable. Images use 4 items and 4 eligible judges.
Mistral image scores are unsupported by design, never filled with zero.

Scores are AI-panel ratings out of 100, not accuracy percentages. No human gold
standard or clinical safety validation is claimed. Warm request times are
specific to this configuration. The top text means (26B 98.614; 31B 98.531) differ
by only 0.083 points. High averages can coexist with local errors; judges can
also make questionable criticisms. All original scores are preserved.

## Offline reproduction / 離線重現

Python 3.10+ standard library, without calling any model:

```sh
python scripts/verify_analysis.py --data-dir docs/data --output-dir verification
python build_site.py
python scripts/package_release.py
python scripts/validate_release.py
python -m http.server 8766 --directory docs
```

`benchmark_code/` contains all 114 archived Python, Jinja, Swift, shell and MJS
files, including historical tools. Latest translation entry:
`translation_benchmark_20261023_new_materials/runner/run_translation.py`.
Latest evaluation aggregation entry:
`translation_benchmark_20261023_new_materials/judge_evaluation/recovery_20261025_failed_jobs/analysis/post_recovery_runner.py`.

Original runners require model weights, compatible macOS/Metal runtime where
applicable, adapted local paths and authenticated cloud accounts. They are not
a turnkey hosted service. Generic `/benchmark` replaces original workstation
roots; hashes and modification records are in `docs/data/source_file_provenance.json`.

Translator settings: context 8192, max output 4096, thinking off. The original
general note's 32768/16384/8192 refers to local judges, not translators. Runtime:
llama.cpp 0.7.0-dev build 524, commit `620d2225c`, Darwin arm64. Exact hardware is
not asserted without evidence.

## Downloads and rights / 下載與授權

- `docs/downloads/complete-results.zip`: complete final public tables and notices.
- `docs/downloads/benchmark-code.zip`: all archived code, website builder,
  offline verifier and licenses.
- `docs/downloads/materials.zip`: 30 frozen texts, 4 exact PNGs and attribution.
  Original WHO PDFs are linked, not redistributed as PDF binaries.

ZIP paths are repository-relative. A Git clone obtains the complete project.
Read [material rights](docs/materials/RIGHTS.md), [license scope](LICENSE) and
[public boundary](PUBLICATION_BOUNDARY.md). WHO-derived content and translations
are CC BY-NC-SA 3.0 IGO. The site's MIT license does not override source material,
archived code or third-party templates. This non-commercial benchmark is not
an official agency translation or medical recommendation.
