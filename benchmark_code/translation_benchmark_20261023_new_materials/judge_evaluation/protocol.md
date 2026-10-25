# 新素材五 judge blind packet protocol

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
1550（pointwise
1190 + pairwise
360）。

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
