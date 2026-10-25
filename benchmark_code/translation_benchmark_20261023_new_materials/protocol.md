# 2026-10-23 七模型新素材翻譯執行協定

這個目錄只保存本輪 translation execution。`materials/` 由材料 agent 管理，禁止覆寫；`models.ini`、launchd 與 formal Open WebUI/LiteLLM 設定只讀。

## Gate

正式 generation 前，runner 必須驗證 `materials/metadata_manifest.json` 的 `freeze_ready=true`、30 筆 frozen text、4 個選定 image、238 個 planned model-case cells，以及 frozen text SHA-256、每筆 text source SHA-256、4 個 PNG SHA-256 與 authoritative bundle byte identity。任何門檻失敗都停止，不以舊素材替代。

本輪使用 `/benchmark/llama-v0.7.0/build-release/bin/llama-server`，每模型一個 isolated sequential server，連接埠 18101–18107。每 batch 只載入一個模型，記錄精確 PID、port、command、server version、`/health`、`/props`、`/v1/models`、stdout/stderr，結束時 guarded cleanup 並確認 port 已關閉。啟動前的 RAM gate 為 conservative available >= (model+mmproj file-size estimate + 32 GiB)，載入後再記錄 available 並要求 >=32 GiB；兩個 threshold/actual 都保存於 server preflight。runtime command version 必須 match config 宣告的 build 524 / commit 620d2225c，且 `/props.build_info` 需同樣 match。正式 8080 只做 read-only GET，不重啟或修改。

## Fixed generation

`ctx-size=8192`, `temperature=0`, `top_p=1`, `seed=20261023`, `stream=false`, `parallel=1`, `batch-size=2048`, `ubatch-size=4096`, `max_tokens=4096`, `reasoning=off`, `reasoning_effort=none`, `--no-cache-prompt`/`cache_prompt=false` where supported, `--no-cont-batching`, `--spec-type none`, no draft/MTP flags, `n_gpu_layers=all`, `flash-attn=on`, one request at a time. The newer llama.cpp flags and resulting `/props`/verbose render are evidence; old b10964 results are not reused as a PASS claim.

## Template owner and requests

llama.cpp is the only template owner. Every request sends ordinary OpenAI `messages`; no caller pre-renders control tokens. Gemma 4 uses the GGUF embedded native Gemma 4 template with `--reasoning off`, `--reasoning-format deepseek`, and `--chat-template-kwargs '{"enable_thinking":false}'`. Its text content is the shared instruction followed by two newlines and the English source. The kwargs are retained for native-template compatibility; a deprecation warning is expected and recorded, while the gate uses observed `reasoning_in_content`/`reasoning_content` and requires empty reasoning.

TranslateGemma 4B/12B use the previously validated fixed English→zh-TW custom adapter because OAI normalization drops upstream typed language fields. TranslateGemma 27B has no embedded template and must use that same adapter. The adapter owns the professional translation instruction and fixed `en`/`zh-TW` fields: text requests therefore send **source-only** user content, so the shared instruction is not accidentally translated as source. Image requests use the standard OAI content array with the shared semantic instruction and one `image_url` data URL; the TG image adapter uses its native image branch and ignores the text part when rendering. No English ground truth is put in an image prompt.

Each real image case is one request with one local PNG encoded as a `data:image/png;base64,...` URL. The redacted request stores the path and image SHA-256 rather than the pixel payload. Image ground truth is retained only in case metadata/CSV for future judging. Image results are end-to-end image translation evidence; they are not OCR CER or OCR exactness measurements.

## Evidence and status

For every case, raw JSONL stores the redacted request, request/response hashes, full parsed response (including separated content/reasoning, finish reason, usage, verbose non-generation prompt, generation settings, timings, cache fields), HTTP envelope, elapsed time, model/mmproj identity, template/config hashes, source/image hash and render/content checks. Transport/format/render failures are retained as-is. Only finite, provenance-linked technical retries are allowed for transport/HTTP failures; retries never replace the original record and are not quality retries.

Before real materials, each route receives a separate technical-marker text and image smoke request. The immutable first smoke JSONL/v1 summaries are retained; the corrected v2 validator recognizes Gemma 4's native `<|image>` + `<image|>` pair and exact one-user/one-model turn structure. A separate `template_probe.py` batch calls documented `/apply-template` for text and image only (no generation) and stores the HTTP/request/time/body/prompt hashes and render envelope; `template_probe.summary.json` must be 7/7 for both before generation. Smoke only establishes endpoint/template/image-marker capability and must not be counted in translation CSVs or quality results. A route that cannot pass native image rendering is recorded `N/A_not_supported`; no OCR-to-text fallback is allowed. Any non-empty reasoning content while reasoning is off, non-stop/invalid finish reason, duplicated source/turn/image marker, or control-token leak in output is retained as a technical failure.

CSV output is CSV-only (`results/translation_results.csv`) with one row for each model×case cell, including failures and unsupported cells. No XLSX/PDF/judging is produced in this step. The bundled ArtifactTool `Workbook.fromCSV` QA is run after CSV creation to verify headers, row count, UTF-8 and sample rows.
