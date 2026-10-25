# 七模型英語→臺灣正體中文翻譯的多 LLM Judge 評測協議

**適用範圍：** TranslateGemma 4B/12B/27B、Gemma 4 E4B/12B/26B/31B QAT 七個本機 GGUF 的心理學、精神醫學文字與圖片翻譯。

**本版狀態：** 目前完全不加入人工評分；本文件定義一個可重跑的自動評測版，以及未來接入人工 gold 的作業版。不得把本版的 judge 一致性誤稱為臨床正確性或人工驗證。

**重要分工：** 七個 GGUF 是「候選翻譯系統」，不是七個獨立 judge。它們都屬 Gemma/TranslateGemma lineage，若互相評分會產生同家族相關偏差與 self-preference；主排名應使用另外選定、通過資格考、至少三個不同模型家族的 judge。這一點與多家族 jury 的研究設計一致。[PoLL 多模型 panel](https://arxiv.org/abs/2404.18796)、[LLM-as-a-Jury for Machine Translation Publishability Assessment](https://aclanthology.org/2026.eamt-1.7/)、[Beyond the Surface: Measuring Self-Preference in LLM Judgments](https://aclanthology.org/2025.emnlp-main.86/)

## 1. 執行摘要與預先註冊決策

### 1.1 建議主設定

| 項目 | 建議 | 理由 |
|---|---|---|
| primary judge panel | **5 個 judge、至少 4 個家族**；每家族最多 2 個 | 5 為奇數，可做多數決；比 3 有較好的錯誤覆蓋，比 7 成本低。若只能找到 3 家族，5 個仍可跑，但須標記「相關 panel」，不得宣稱 5 個獨立觀測者。 |
| judge 輸出 | 結構化 JSON：pointwise MQM spans + 分數 + critical flag；另做 A/B/TIE | Scalar 不能解釋錯誤，也可能讓藥名、數字、否定等 critical error 被平均分數掩蓋。GEMBA-MQM V2 也以結構化 MQM 標註和多次聚合降低變異。[GEMBA V2](https://aclanthology.org/2025.wmt-1.67/) |
| 主要 ranking | pointwise 分數作品質軸；A/B/TIE 作相對 ranking 交叉驗證；兩者分歧則 `uncertain` | Pairwise 對中型開放模型常優於絕對打分，但有 position bias，必須雙向交換。[LLM Comparative Assessment](https://aclanthology.org/2024.eacl-long.8/)、[Large Language Models are not Fair Evaluators](https://aclanthology.org/2024.acl-long.511/) |
| critical safety gate | deterministic violation 或至少 2/5 judges 標記 critical → `FAIL/REVIEW`；只有 1/5 → `REVIEW`，不可自動通過 | 沒有人類真值時採 fail-closed；一票不是自動定罪，但未解決的 critical 分歧也不能標為安全。 |
| benchmark 規模 | 40 個 source item：32 文字、8 圖片；另有 30 個 judge qualification item | 20 項只適合 smoke test；40 項才適合初步排名與分層 CI。所有結論仍應標為本地小樣本結果。 |
| speed | 與品質完全分開報告；以 warm API elapsed、prompt tok/s、generation tok/s、p50/p95、timeout rate 建 Pareto front | 快不代表正確，品質分數不可用速度加權。 |

### 1.2 3/5/7 judges 的取捨

| panel | 最低家族條件 | 用途 | 限制 |
|---|---:|---|---|
| 3 | 3 家族、每家族 1 個 | 最低可用 screening；2/3 majority | 對一個錯誤判斷很敏感；不適合高風險 critical gate 的唯一依據。 |
| **5（推薦）** | **至少 4 家族；最多一個家族 2 席** | primary pointwise、critical gate、top-candidate A/B | 成本約為 3 judge 的 1.67 倍，但多數與 leave-one-out 敏感度分析較穩。 |
| 7 | 至少 5 家族；最多兩家族各 2 席 | 需要較穩定的研究 jury、要估 panel dependence 或做 leave-one-out | 比 5 多 40% 呼叫；若新增 judge 和既有 judge 高度相關，實際有效樣本幾乎沒增加。 |

**5 個不是文獻證明的普遍最適值，而是本案在成本、奇數投票與 leave-one-out 診斷之間的工程選擇。** 已有研究發現，即使 9 個 judge 橫跨 7 個家族，也可能只提供約 2 個獨立票的資訊；所以是否由 5 升到 7，必須看 qualification residual correlation、`N_eff` 與 leave-one-judge-out 排名，而不是只看名目數量。[Nine Judges, Two Effective Votes](https://arxiv.org/abs/2605.29800)

GEMBA-MQM V2 對同一 segment 做十次判斷，再移除 2σ outlier 並以 reciprocal-rank weighted average 聚合；這是「同一 judge 的 stochastic repeatability」證據，不等於十個獨立家族。[GEMBA V2](https://aclanthology.org/2025.wmt-1.67/)。本協議採 5 個 judge、至少 4 個家族為主，qualification 和 disagreement item 另做 2–3 次 repeat。

## 2. Judge 入選條件與資格考

### 2.1 必要資格

每個 judge 必須在正式 benchmark 前通過以下六類測試。分數門檻是本專案預先註冊的工程 gate，不是文獻宣稱的通用標準；所有原始輸出、模型 digest、prompt hash、版本與 seed 都要保存。

| 資格 | 測試設計 | 建議 pass gate |
|---|---|---:|
| 英語→zh-TW 語言/locale | 來源含臺灣用語、繁簡字、醫學常用譯名、量表名、英文殘留；judge 必須辨識 `zh-Hant-TW` 而非僅「中文」 | 20 個 locale checks 至少 19 正確；不得把明確簡體輸出標成臺灣 locale pass。 |
| 精神醫學/藥理能力 | 來源與候選含 diagnosis、症狀、藥名、劑量、單位、頻率、禁忌、副作用、危機/自殺安全語意 | 30 個 known-error item 中，critical mutation recall ≥ 0.90；否定、劑量與自殺風險三類 sentinel 不得漏判；critical false-negative 不得被平均分數掩蓋。 |
| MQM 錯誤分類 | 對 omission、addition、mistranslation、untranslated、terminology、number/dose、negation、time、fluency、locale/format 輸出 type/severity | 已知 mutation 的 type macro-F1 ≥ 0.85；severity 至少 90% 落在正確或相鄰級別。 |
| structured output | 僅輸出 JSON；error span 是 target-side character offsets，start/end 合法；無額外 markdown/commentary | 100 次資格呼叫中 schema parse ≥ 99%；offset 合法 ≥ 98%；失敗最多重試一次，仍失敗即剔除該 judge。 |
| repeatability | 同一 item、同一 prompt、同一參數重跑 3 次；另測空白、換行、不可見字元的 invariance | categorical agreement ≥ 0.90；0–100 score 的 item-level SD ≤ 5；否則只能做診斷 judge。 |
| position/self/style bias | A/B 交換順序；候選匿名；改變長度、標點、格式但不改語意；若 judge 是候選同家族，加入 self-output identity test | A/B 順序造成的 win-rate 差 ≤ 0.05；無模型名稱可辨識輸入；若 self-preference 無法在 known-quality anchor 上校正，剔除 primary panel。 |

### 2.2 Qualification item 組成

30 個資格 item 建議包含：10 個 clean translation、10 個單一 critical mutation、10 個 major/minor 或 locale/format mutation。critical mutation 至少涵蓋：

- 否定翻轉：`does not` ↔ `does`、`no suicidal thoughts` ↔ `suicidal thoughts`。
- 數字/劑量/單位：`5 mg` ↔ `50 mg`、`twice daily` ↔ `once daily`。
- 時間與條件：`within 24 hours`、`after stopping`、`if symptoms worsen`。
- 藥名、禁忌、副作用、危機/急診指示的替換或刪除。
- omission、addition、untranslated English、簡體字注入、格式與段落破壞。

這些是「已知注入錯誤」而非人工 gold；它們只能驗證 judge 是否對指定變異敏感，不能證明原始臨床文字翻譯正確。資格資料須與正式 40 項 benchmark 分離，避免 prompt overfit。WMT25 也指出 LLM-as-a-judge 的效能會隨語言而變，且應用與人類相同的指令/評估條件，而不是只套英文 judge prompt。[WMT25 MIST findings](https://aclanthology.org/2025.wmt-1.23.pdf)

Judge 也必須先證明自己能理解來源英語，而不只是偏好流暢的中文或貼近 reference 的表面形式；研究發現 MT judge 有時無法有效利用 source，而 reference 對評分影響更大。[Lost in the Source Language](https://aclanthology.org/2024.findings-acl.211/)。因此 qualification 同時要有「無 reference、只靠 source 才能判斷」的最小對立題，以及有 reference 的標準題。

### 2.3 Judge selection 的依賴性

候選 judge 應記錄：模型家族、base checkpoint、fine-tune/adapter、tokenizer、量化、版本、system prompt、temperature/top-p/seed、reasoning mode。相同 base 或同一蒸餾來源視為同一家族；同一個候選翻譯模型不可同時以「候選」和「primary judge」身份出現。

將 qualification 的 item-level 分數做 judge correlation matrix。可用等相關近似的有效 judge 數作診斷：

```text
N_eff ≈ K^2 / (K + 2 * sum(pairwise_rho))
```

這不是正式抽樣模型，但能揭示「表面 5 judge、實際 2 個獨立訊號」。一家族間 correlation 高於約 0.80 時，應折疊為一個 family signal，或新增真正不同 family；不可只用更多同 lineage 小模型湊數。多家族 jury 減少 correlated bias 的實證與 logistic aggregation 見 [LLM-as-a-Jury](https://aclanthology.org/2026.eamt-1.7/)。

## 3. 評測資料、盲測與固定 prompt

### 3.1 40 個正式 item

建議從原始英文資料建立 40 個固定 item：

| 類別 | 數量 | 必測內容 |
|---|---:|---|
| 臨床敘述/病人病史 | 8 | 症狀、時間、否定、主訴、診斷不確定性 |
| 病人教育/心理衛教 | 8 | 語氣、可讀性、行動指示、段落與標題 |
| 藥理/安全 | 8 | 藥名、劑量、頻率、單位、禁忌、副作用、交互作用 |
| 量表/問卷/格式 | 4 | item numbering、Likert 選項、反向題、分數與欄位 |
| 危機/精神醫療安全 | 4 | 自傷/他傷、急診、求助、條件與時間門檻 |
| 圖片/文件翻譯 | 8 | OCR 文字、藥名/數字、版面與圖片語境 |

來源應盡量是原始英語撰寫，避免 translationese；WMT25 的資料設計也明確避免以翻譯文本當 source，以免影響評測。[WMT25 MIST findings](https://aclanthology.org/2025.wmt-1.23.pdf)

### 3.2 Blinding 與 prompt

- 每個 candidate 的 model name、參數量、量化、速度、檔案大小、既有分數全部替換為隨機 `cand-01`…`cand-07`。
- 每個 judge 每個 item 的 candidate order 由固定 seed 隨機化；A/B 測試另以相反順序重送。
- Judge prompt 固定包含 `source_language=English`、`target_language=Traditional Chinese`、`target_region=Taiwan`、`tgt_language_code=zh-TW`、`domain_description=psychology/psychiatry clinical or patient education`、保留段落與格式要求。WMT25 的正式 prompt 也採 target region、language code、domain instruction 與 paragraph preservation。[WMT25 prompt appendix](https://aclanthology.org/2025.wmt-1.23.pdf)
- 主要 pointwise track 不提供 reference；若有可靠 reference，另跑 reference-assisted track，不得混成同一分數。
- 保存 `prompt_sha256`、`payload_sha256`、`judge_model_sha256`、API raw response 與 parse result，讓後續能重播。

## 4. 無人工版三階段流程

### Stage 0：輸入與 deterministic gates

先不讓 judge 的主觀分數影響硬檢查：

1. Unicode、輸出非空、語言辨識、繁簡與臺灣字形檢查。
2. 找出 source 的數字、百分比、日期、時間、單位、劑量、頻率、藥名、否定詞、條件詞，對照 target 是否遺失、增加、改值或改關係。
3. 檢查未翻譯英文、control marker、JSON/XML/Markdown 結構、段落/標題/編號、圖片文字區塊。
4. 若 source 有明確危機或安全指示，任何 deterministic omission/negation/number/dose violation 直接 `critical_deterministic=true`。

deterministic gate 不能判斷自然度或所有語義，只能保證已知 token/格式未被破壞；通過 gate 不代表 translation pass。

### Stage 1：Pointwise MQM error-span judges

每個 judge 對每個 candidate 獨立輸出：

```json
{
  "candidate_id": "cand-03",
  "score_0_100": 84,
  "subscores": {
    "terminology_locale": 26,
    "semantic_safety": 23,
    "fluency": 18,
    "completeness_format": 17
  },
  "critical": false,
  "errors": [
    {
      "target_start": 42,
      "target_end": 48,
      "severity": "major",
      "type": "accuracy/omission",
      "evidence": "source condition is missing",
      "confidence": 0.88
    }
  ]
}
```

MQM taxonomy 應縮減到本案可重現的 `accuracy`（addition/mistranslation/omission/untranslated/negation/number/time）、`terminology`、`fluency`、`locale convention`、`style/register`、`format/markup`。MQM Council 將 terminology、accuracy、linguistic conventions、style、locale conventions、audience appropriateness、design/markup 作為可組合的分析框架，且允許依健康風險調整權重。[MQM Council](https://www.themqm.org/)、[MQM scoring model](https://www.themqm.org/mqm-pillars/the-mqm-scoring-models/)

評分預設為：terminology/locale 30、semantic/safety 30、fluency 20、completeness/format 20。critical error 不用平均分數抵銷；必須獨立記錄 `critical=true`。GEMBA-MQM V2 的 25/5/1（critical/major/minor）是可參考的 MQM badness mapping，不是本醫療 domain 的自然法則；若採用，需在報告明寫為預註冊權重。[GEMBA V2](https://aclanthology.org/2025.wmt-1.67/)

正式 5-judge run 每個 item 做一次；qualification、judge disagreement、critical flag 與 top-3 close cases 另做兩次 repeat。固定 temperature=0 時也要測 deterministic replay；若服務仍有變異，保存所有 runs，對分數取 median、對 categorical label 取 majority，對 critical 採 max/flag-preserving。

### Stage 2：Pairwise A/B/TIE 與順序交換

對兩個匿名 candidate，judge 只回答：

```json
{
  "a_or_b_or_tie": "A",
  "confidence": 0.76,
  "critical_override": false,
  "reason_codes": ["better_semantic_fidelity", "fewer_terminology_errors"]
}
```

每一對必須跑兩種順序：`A=cand-i,B=cand-j` 和 `A=cand-j,B=cand-i`。聚合前先把 A/B label 反解成 canonical `candidate_id`：若第一次選 A（cand-i），交換後應選 B（仍是 cand-i），這才是順序一致；若兩次都選 A 或兩次都選 B，實際偏好的 candidate 已改變，記為 `ABSTAIN/position_sensitive`，不可假裝成真正品質平手。兩次都回 TIE 才記為穩定 TIE。報告 `swap_consistency`、A/B win-rate delta、TIE rate。

pairwise 只比較 translation quality，不顯示模型名、速度或 pointwise 分數。EACL 的研究顯示 pairwise 對中型開放模型通常比絕對打分更有效，但 LLM 會受選項位置影響；因此順序交換是必要控制，不是可選的額外實驗。[LLM Comparative Assessment](https://aclanthology.org/2024.eacl-long.8/)、[Judging the Judges](https://aclanthology.org/2025.ijcnlp-long.18/)

### Stage 3：聚合與 decision gate

1. 以 target character offset 對 spans 做 clustering；建議 boundary IoU ≥ 0.50 視為同一 span。cluster 內 type 取 majority，severity 同時報告 modal 與 max；安全 gate 採 max，品質摘要採 modal/median。
2. major/minor span cluster 若有 ≥3/5 judge 命中，列為 `panel_consensus_error`；2/5 列 `disputed_error`；1/5 列 `singleton_error`。三種都保留原始 evidence，不把少數意見刪掉。這是本案預註冊聚合規則，不是 MQM 官方規定。
3. pointwise 最終穩健摘要取 5 judges 的 median；另報 arithmetic mean、IQR 與 judge correlation。`ICC(2,k)`只對 **k 位 judge 的 arithmetic mean** 解釋 panel reliability，不可拿來聲稱 median 的信度。
4. critical：deterministic violation 或 ≥2/5 judge critical → `FAIL`；1/5 → `REVIEW`；0/5 且 deterministic clean 才能進一般 ranking。沒有人工 review 時，`REVIEW` 不得寫成「安全通過」。
5. pairwise 先做逐對 majority（A/B/TIE），再做 Copeland 作透明 audit；同時擬合 Bradley–Terry（不允許 tie）或 Davidson tie model（允許 tie）。正式 rank 須在三者敏感度分析中大致一致；若 top-1 改變，報 `ranking_uncertain`。Pairwise 的跨 judge agreement 用 canonical candidate winner 的 Krippendorff nominal α/Gwet AC1，另報 swap consistency；`position_sensitive` 視為 abstention/missing，不視為 tie。

## 5. 7 個 candidate 的呼叫量與省成本方案

令 `N` 為 item 數、`K` 為 judge 數；每個 HTTP chat completion 計一個 judge call。

| 工作 | 呼叫量/每 item | 40 item、5 judge |
|---|---:|---:|
| pointwise 一次 | `7 × K` | 1,400 |
| full round-robin、單一順序 | `C(7,2) × K = 21K` | 4,200 |
| full round-robin、A/B 雙向 | `21 × K × 2 = 42K` | 8,400 |
| full 7 judge、A/B 雙向 | `42 × 7` | 11,760 |

推薦的省成本方案：

1. 所有 7 candidate 先做 5-judge pointwise：每 item 35 calls。
2. 依 pointwise median 排序，先比較相鄰 6 對，以 3 judge 做 A/B 雙向：`6×2×3=36` calls/item。
3. 對 top-3、pointwise 差距 ≤5 分、critical disagreement 或 swap inconsistency 的最多 3 對，升級 5 judge 雙向：最多 `3×2×5=30` calls/item。
4. 最壞約 101 calls/item，即 4,040 calls/40 item；只對真正有排名影響的 pairs 做 full 5-judge。若正式研究必須宣稱 7 個模型的完整 pairwise tournament，才使用 8,400 calls 的 full 5-judge 版本。

全 round-robin 的 pairwise 結果可用：

- **Majority：** 每對取 3/5 多數；最透明，但遇到 cycle 只能另處理。
- **Copeland：** wins − losses（tie 半分）；非參數、適合 audit，但忽略 margin。
- **Bradley–Terry：** 由 pairwise wins 估計 latent strength；用 [Bradley & Terry 1952](https://doi.org/10.1093/biomet/39.3-4.324) 原始模型。
- **Davidson：** Bradley–Terry 加 tie probability；本案應優先於把 TIE 任意拆成 0.5，見 [Davidson 1970](https://doi.org/10.1080/01621459.1970.10481082)。

沒有人工 gold 時不得用同一批結果訓練一個複雜的 learned weight；先用 equal judge weights、median、預註冊門檻。未來有人類 calibration 才可像 LLM-as-a-Jury 論文一樣用 logistic regression 學 judge weights。[LLM-as-a-Jury](https://aclanthology.org/2026.eamt-1.7/)

## 6. Reliability 統計：何時用什麼

Reliability 只回答「同一組 judge 是否穩定/一致」；validity 才回答「是否接近人類/真實翻譯品質」。一群 judge 共同漏判同一個劑量錯誤時，ICC、alpha、Kendall W 都可能很高，但 validity 仍然很差。

| 統計量 | 適用資料 | 本案使用方式 | 不可誤用 |
|---|---|---|---|
| `ICC(2,1)` | 連續 0–100 score；two-way random、single judge、absolute agreement | 把 judge 視為可能抽自更大 judge population，報單一 judge 的絕對一致性 | 不適用 categorical span/type；不把 Pearson correlation 當 ICC。 |
| `ICC(2,k)` | 連續 score；two-way random、average of k judges、absolute agreement | 主要報告 5-judge **arithmetic mean** 的 panel reliability；若要泛化到其它 judge | 不適用 median；score offset 重要時不可只用 consistency ICC。 |
| `ICC(3,k)` | 連續 score；two-way mixed/fixed、目前這組固定 judges、average、consistency | 報「目前選定 panel 的相對排序是否一致」；不宣稱可泛化到新 judge | 它不是 absolute agreement；軟體命名在 Shrout–Fleiss 與 McGraw–Wong 間不同，報出 model/type/unit 全名。原始分類見 [Shrout & Fleiss 1979](https://pubmed.ncbi.nlm.nih.gov/18839484/)、[McGraw & Wong 1996](https://doi.org/10.1037/1082-989X.1.1.30)。 |
| Krippendorff `α` | 多 judge、缺失值、nominal/ordinal/interval/ratio | `α_nominal` 用 error type；`α_ordinal` 用 minor/major/critical；`α_interval` 可做分數或嚴重度 numeric；span 先轉成同一 unit | 不把高 alpha 當 correctness；需報 distance function、missingness、unit 定義。Krippendorff 作者的原始工具與說明見 [Krippendorff Alpha](https://www.asc.upenn.edu/krippendorffs-alpha-reliability)。 |
| weighted kappa | 兩個 judge 的 ordinal labels | 對一對 judge 的 severity 或 pairwise confidence 做 linear/quadratic weighted κ；用於 judge-level audit | 5/7 judge 的主結果不應只平均兩兩 κ；rare critical 時 κ 受 prevalence 影響。 |
| Gwet `AC1/AC2` | nominal/ordinal，尤其某類極少或極多 | critical yes/no 用 AC1；severity ordinal 可用 AC2；同時報 raw agreement、recall、precision | 不以 AC1 取代所有 κ；它是 prevalence paradox 的 robustness check。原始方法見 [Gwet 2008](https://doi.org/10.1348/000711006X126600)。 |
| Kendall `W` | 多個 judge 對同一批 candidate 的 rank | 每個 item 或整體對 7 candidate 的 judge ranking concordance；報 W 與 bootstrap CI | 只表示 rank concordance，不表示 rank 對；原始多 ranking 問題見 [Kendall & Babington Smith 1939](https://doi.org/10.1214/aoms/1177732186)。 |

建議所有 reliability 以 source item 為 bootstrap unit，B=10,000、95% CI；不可把同一 item 的 5 個 judge calls 當成五個獨立 source observations。

ICC 必須同時寫出 model、type、definition（absolute agreement 或 consistency）、single/average measure 與 95% CI；不能只寫「ICC=0.82」。本案預設同時報 `ICC(2,1)` 與 `ICC(2,5)` absolute agreement；若 judge 永久固定，才另報 `ICC(3,5)` 作補充。選型與常見解讀方式參考 [Koo & Li 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/)；其常用區間為 `<0.50` poor、`0.50–0.75` moderate、`0.75–0.90` good、`>0.90` excellent，但應以 CI 而非單一點估計判斷，而且這些標籤不是 validity 或醫療安全標準。

本案的預註冊 reliability gate：`ICC(2,5)` 的 95% CI 下限至少 0.75 才發布 pointwise 硬排名，0.90 以上為期望；Krippendorff ordinal/nominal α 或 Gwet AC2 以 0.67 作最低暫定線、0.80 為較穩定目標；pairwise swap consistency 至少 0.90。任何 agreement 過線但 known-error critical recall 未達 0.90，或漏掉任一否定/劑量/自殺 sentinel，仍判定 panel validity 失敗，不得發布部署推薦。0.67/0.80 是常用的內容分析判讀界線而非自然定律，須連同 bootstrap CI、樣本數、類別盛行率及 raw agreement 一起報。[Krippendorff reliability resources](https://www.asc.upenn.edu/krippendorffs-alpha-reliability)

## 7. Error-span agreement 與 critical agreement

### 7.1 Span 對齊

將 target-side spans 正規化到 Unicode NFC 後以 character offset 保存；每一對 span 計：

```text
IoU = intersection_length / union_length
```

IoU ≥ 0.50 視為 match；再報 strict exact match（start/end 完全一致）、soft span precision/recall/F1、boundary distance。只重疊但 type 不同時，span 算 match、type 算錯；severity 另算。

### 7.2 類別與嚴重度

- error type：每個 matched span 的 macro-F1、micro-F1、confusion matrix、Krippendorff nominal α。
- severity：minor/major/critical 的 weighted κ、Krippendorff ordinal α、critical recall。
- critical detector：報 precision、recall、F1、false-negative count；醫療安全以 recall 優先，不能只報 accuracy。
- 無法對齊 spans 時，不把某 judge 的空 span 當「無錯誤真值」；記為 `unmatched/uncertain`。

## 8. Known-error synthetic anchors：無人工真值時如何做 validity proxy

在每個 domain 建立可驗證的 pair：`clean` 與只改一件事的 `mutated`。來源可以是已知保留語意的候選或經程式產生的 target；mutation 必須可逆、可記錄、不可依賴 judge 自己生成。

每個 judge 須滿足：

1. `clean` 分數高於 `mutated` 的 monotonicity ≥0.90。
2. critical mutation recall ≥0.90；對 clean 的 critical false-positive ≤0.10。
3. 對 whitespace、句號、段落等不影響語意的 mutation，分數差 ≤5 分。
4. A/B 順序交換後 winner 不變；否則標 `position_sensitive`。
5. model id、候選長度、標點 style 互換不應改變同一語義錯誤的 severity。

這些測試可驗證 evaluator 的行為契約；不能替代 40 項真實 domain translation 的人工 gold，也不能宣稱醫療安全已被證明。WMT25 的 metric shared task 顯示，即使 system-level LLM 指標很強，segment-level 的 reference-based metric 仍可能勝出，且 span error detection 的 precision/recall 仍是難題。[WMT25 automated evaluation findings](https://aclanthology.org/2025.wmt-1.24/)

## 9. Image translation 與 OCR 必須分開

### 9.1 OCR track

只有在擁有獨立 OCR ground truth（人工逐字轉錄、原始圖片文字檔或可審計的 OCR reference）時，才計 `OCR exact/char error rate/number exact`。judge 必須看到 image 與 OCR candidate，不能把「翻譯讀起來合理」當 OCR exact。

### 9.2 Image translation track

無獨立逐字 OCR 時，指標命名為 `image_translation_content_accuracy`，分解為：圖片文字/內容保留 40、藥名/劑量/數字 30、zh-TW 自然度與格式 30。保留 image input、mmproj、圖片 hash、crop/resize、OCR reference 狀態。

圖片的 critical gate 仍優先檢查藥名、劑量、數字、否定與安全指示；沒有 OCR gold 時，報告不得寫「OCR exact」。Appraise 與 Pearmut 都有 multimodal/圖片 annotation 支援，可作未來人工介面，但仍需自行提供醫療/zh-TW gold。[Appraise](https://github.com/AppraiseDev/Appraise)、[Pearmut](https://github.com/zouharvi/pearmut)

## 10. Speed 是獨立 Pareto 軸

對每個 candidate × item × endpoint 保存：`api_elapsed_s`、`prompt_tokens`、`completion_tokens`、`prompt_tok_s`、`generation_tok_s`、`cold_or_warm`、`cache_hit`、`timeout`、`HTTP status`。品質表只保存 quality/critical/ranking；速度表另算：

- warm median / p95 API elapsed；
- prompt tok/s 與 generation tok/s 的 median/p95；
- cold load 不與 warm throughput 混在平均值；
- image 與 text 分開；圖片 preprocessing/decode time 另列；
- Pareto-optimal candidates：沒有另一模型同時品質更高、critical 更少且速度更快者。

不要把 `quality + λ·speed` 當唯一排名；若必須選 deployment trade-off，先呈現品質排名和 speed Pareto front，再由部署需求選擇。

### 10.1 Embedder / reranker 的位置

- 通用 multilingual embedder 的 source–target cosine similarity只能當低成本異常偵測：它可能把關鍵否定、數字、劑量或時間翻錯但語意向量仍判得很近，因此不可計入 critical gate，也不可充當第六位 judge。
- 一般 reranker/cross-encoder 若沒有以 DA/MQM/ESA 人工標註訓練及在 English→zh-TW、精神醫學 domain 校準，輸出分數沒有可解釋的品質刻度；只能拿來 triage 或挑選需複評的樣本。
- 若要加入 learned metric，優先用專為 MT 且曾以人工 DA/MQM 資料訓練/驗證的 [COMET](https://github.com/Unbabel/COMET)、[xCOMET](https://aclanthology.org/2024.tacl-1.54/) 或 [MetricX](https://github.com/google-research/metricx)。它們作為與 LLM panel **獨立的輔助軸**，報 segment/system correlation、known-error sensitivity 與 disagreement cases，不與五位 LLM judge 簡單投票。WMT25 顯示 reference-based metric 在 segment-level 仍可能優於 LLM evaluator，支持保留這條獨立交叉檢查。[WMT25 automated evaluation findings](https://aclanthology.org/2025.wmt-1.24/)

## 11. 無人工版的正式 ranking 與門檻

### 11.1 分數

文字每個 candidate/item 的四個子分數分別採 0–30、0–30、0–20、0–20 點制，因此總分直接相加：

```text
text_score = terminology_locale_points
           + semantic_safety_points
           + fluency_points
           + completeness_format_points
```

若實作改以每項 0–100 正規化，才可使用 0.30/0.30/0.20/0.20 權重。圖片使用第 9 節的 40/30/30。最終穩健摘要用 5-judge median；同時保存 raw judge scores、arithmetic mean、IQR、只對 mean 解釋的 ICC 與 span evidence。

### 11.2 deployment status

以下是本專案的 pre-registered gate，不是臨床或法規標準：

| 狀態 | 條件 |
|---|---|
| `AUTO_PASS_FOR_BENCHMARK` | deterministic clean、0/5 critical、pointwise JSON valid、swap consistency ≥0.90、品質分數 ≥90；僅代表可進自動 benchmark ranking。 |
| `REVIEW_REQUIRED` | 任一 judge critical、A/B swap 不一致、score IQR >10、家族 dependency 過高、image OCR reference 缺失但需宣稱 OCR、或分數 80–89.99。無人工版不得對外稱安全通過。 |
| `FAIL_FOR_DEPLOYMENT` | deterministic critical、至少 2/5 critical、品質 <80、或出現藥名/劑量/否定/時間的 confirmed error。 |

正式模型排名先依 gate 排除 `FAIL`，再依 text/image quality；若 two rankings 在 bootstrap 95% CI 完全重疊或 top-1 在 majority/Copeland/BT/Davidson 間改變，報 tie/uncertain，不強行排序。

## 12. 未來人工 gold：MQM/ESA/cESA 作業流程

### 12.1 招募與訓練

未來至少招募 2 名臺灣正體中文母語、具專業翻譯經驗且熟悉心理/精神醫學術語的 annotator；critical disagreement 由第 3 名 senior bilingual/clinical linguist 仲裁。先用 10–15 個 training item、10 個 qualification item，所有 annotator 看到同一 glossary、severity examples、否定/數字/劑量/時間規則。

### 12.2 盲評與雙標

- 隱藏模型名、大小、速度與既有 automatic score；固定 random seed 和 candidate order。
- 100% critical candidate、至少 25% 隨機 item 由兩名 annotator independently 評；保留未仲裁 labels。
- 先標 error span/severity，再給 overall 0–100；這是 ESA 的基本順序，研究顯示它比直接給分更有效率，且不需完整 MQM expert taxonomy。[ESA](https://aclanthology.org/2024.wmt-1.131/)
- 對多候選同一 source，可用 cESA：同時看多個匿名 translation、標 major/minor spans，再給每個 output 絕對分數；cESA 已示範 text/image 等多模態，但目前主要證據為 English→Japanese，不是 zh-TW 醫療驗證。[Contrastive ESA](https://arxiv.org/abs/2607.26640)

### 12.3 仲裁與統計

仲裁只在 disagreement 和 critical items 進行；不得把仲裁結果回填成「原始一致」。報：weighted κ、Krippendorff α、ICC(A,k)/ICC(2,k)、span IoU/F1、critical recall、Kendall W。模型比較用 source-clustered paired bootstrap B=10,000、95% CI；21 個 candidate pair 若同時檢定，使用 Holm correction。人工 gold 建立後，才可用 correlation、calibration、error recall 對 judge 做 validity 評估。

## 13. 可直接採用的工具與資料

| 工具/資料 | 可直接用途 | 目前限制/授權注意 |
|---|---|---|
| [MQM Council](https://www.themqm.org/what-is-mqm/) | taxonomy、severity、scorecard；可縮減成醫療/zh-TW 子集 | MQM Council 網站內容標示 CC BY 4.0；重製須 attribution、license link、變更說明。[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)；taxonomy 本身不是 zh-TW/醫療 gold。 |
| [Appraise](https://github.com/AppraiseDev/Appraise) | DA、pairwise DA、multimodal、ESA、MQM campaign；可架設人工介面 | 官方 repo BSD-3-Clause；需自行檢查第三方 source/data 權利。ACL demo paper 說明其 campaign、ranking、error classification 能力。[Appraise paper](https://aclanthology.org/C18-2019/) |
| [Pearmut](https://github.com/zouharvi/pearmut) | static/本機 campaign；DA、ESA、ESA-AI、MQM、cESA、多模態；預設隱藏 model names 並可固定 shuffle seed | repo MIT；2026 工具/論文較新，應固定版本與 schema；醫療/zh-TW rubric 仍需自行設計。[Pearmut paper](https://arxiv.org/abs/2601.02933) |
| [WMT MQM human evaluation](https://github.com/google/wmt-mqm-human-evaluation) | professional/native-target MQM TSV、WMT20–23 比較與資料格式 | repo 標示 Apache-2.0，但內含 test/source data 的原始權利需逐檔檢查；語言主要 en-de、zh-en，沒有 en→zh-TW medical gold。 |
| [WMT25 human ESA/MQM](https://aclanthology.org/2025.wmt-1.24/) | 含 English→Chinese 的雙人 ESA 標註，可測試 score/span 管線與跨 judge agreement | Chinese 並非保證 zh-TW，且不是心理/精神醫學；可作 judge/general-pipeline sanity check，不可當本案 locale/domain gold。 |
| [WMT26 cESA data + Pearmut recipe](https://github.com/wmt-conference/wmt26-general-mt/) | 最新 cESA campaign、資料格式、多模態欄位與可重跑的人工作業範例 | 2026 結果與工具較新，須固定版本；語言與 domain 仍不等於臺灣精神醫學。 |
| [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval) | WMT metrics/MQM 輸入、system/segment-level correlation、paired significance 工具 | WMT gold 的語言、domain、license 不等於本案；不能把 WMT score 直接當 zh-TW clinical truth。 |
| [bio-MQM dataset](https://github.com/amazon-science/bio-mqm-dataset) | 最接近本案的 biomedical MQM：11 language pairs、專業醫療 annotator、MEDLINE/WMT21 衍生資料 | 不是 zh-TW 保證；source/data 的 NLM/WMT 權利要按 repo 與各檔案查核，不作無條件再發佈聲明；適合 domain-shift sanity check。 |
| [xCOMET/COMET](https://github.com/Unbabel/COMET) | reference-based/QE cross-check、error span/critical error diagnostic | metric 是另一個 automatic evaluator，不是人類真值；需固定 model version、license、語言與 domain calibration。 |

WMT MQM 與 bio-MQM 可用來驗證資料結構、錯誤類別、domain-shift 測試與統計腳本；不應直接把它們的分數併入七模型正式分數，除非 source、target locale、domain 和 reference 可比。

## 14. 建議 JSON schema

每個 `candidate × item × judge × run` 一列 JSON，另用 aggregation JSON 保存 rank；CSV 只作扁平檢視，不丟掉原始 spans。

```json
{
  "run": {
    "run_id": "2026-09-17-mj-v1",
    "prompt_sha256": "...",
    "judge_model_sha256": "...",
    "temperature": 0,
    "seed": 0,
    "blind_seed": 9182
  },
  "item": {
    "item_id": "psy-001",
    "domain": "psychiatry_safety",
    "input_type": "text",
    "source_en": "...",
    "target_locale": "zh-TW",
    "image_sha256": null,
    "ocr_reference_status": "not_available"
  },
  "candidate": {
    "blind_id": "cand-03",
    "model_id_hidden": "...",
    "raw_translation_zh_tw": "...",
    "source_raw": "..."
  },
  "judge": {
    "judge_id": "judge-04",
    "family": "family-x",
    "qualification_version": "q-v1"
  },
  "deterministic": {
    "language_ok": true,
    "traditional_ok": true,
    "numbers_ok": false,
    "drug_terms_ok": true,
    "format_ok": true,
    "critical_deterministic": true
  },
  "pointwise": {
    "score_0_100": 71,
    "subscores": {"terminology_locale": 25, "semantic_safety": 16, "fluency": 17, "completeness_format": 13},
    "critical": true,
    "errors": [{"target_start": 10, "target_end": 14, "severity": "critical", "type": "accuracy/number_dose", "evidence": "..."}]
  },
  "pairwise": [{"pair_id": "cand-03__cand-05", "orientation": "AB", "winner": "B", "confidence": 0.91, "reason_codes": ["dose_preservation"]}],
  "timings": {"api_elapsed_s": 1.2, "prompt_tok_s": 120, "generation_tok_s": 42, "cold": false, "timeout": false}
}
```

Aggregation 必須另存：`span_clusters`、`judge_votes`、`critical_vote_count`、`swap_consistency`、`ICC_2_1`、`ICC_2_k`、`ICC_3_k`、`alpha_nominal`、`alpha_ordinal`、`AC1`、`AC2`、`Kendall_W`、`Copeland`、`Bradley_Terry`、`Davidson`、bootstrap CI、`deployment_status`、`ranking_uncertain`。

## 15. 限制與不可宣稱事項

1. 五個 LLM judge 的高一致性不等於翻譯正確；無人工 gold 的結果應稱為 `automated panel assessment`。
2. synthetic anchors 只驗證已知錯誤的敏感度；它們不能覆蓋自然語意、專業術語歧義或圖片視覺推理。
3. GEMBA、COMET、MetricX、xCOMET、BLEU/chrF/BERTScore 都可能有 language、reference、style、translationese 或 domain bias；WMT25 的結果也顯示 segment-level reference-based 與 LLM judge 的優勢不固定。[WMT25 automated evaluation findings](https://aclanthology.org/2025.wmt-1.24/)
4. judge model 若與七個 candidate 同家族，必須在主排名中排除或另報 dependency sensitivity；不得把相關 judge 的多數票當獨立臨床審查。
5. 40 項只能支持初步 model selection；要宣稱可部署到醫療/病人-facing workflow，仍需臺灣繁中專業人工 MQM/ESA/cESA calibration、critical-error review、版本鎖定與回歸測試。

**預期檔案位置：** `/benchmark/translation_benchmark_20260916_luna/multi_judge_evaluation_protocol.md`。本協議不修改既有 `deployment_*`、`template_audit*`、raw JSONL、CSV、正式 `models.ini` 或服務。
