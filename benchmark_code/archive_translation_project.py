#!/usr/bin/env python3
"""Build a verifiable, non-destructive project handover; no model calls."""
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'translation_project_archive_20261025'
PROJECTS = ['translation_benchmark_20260916_luna', 'translation_benchmark_20261022_judge_pilot',
            'translation_benchmark_20261023_new_materials', 'translation_eval_materials_20261022',
            'translation_materials_psych_20261022_luna']
REC = ROOT / PROJECTS[2] / 'judge_evaluation/recovery_20261025_failed_jobs'
REPORTS = REC / 'post_recovery_reports'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write(name, content):
    (OUT / name).write_text(content, encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((REPORTS / 'analysis_manifest.json').read_text())
    assert manifest['recovery']['terminal']['complete']
    assert manifest['recovery']['same_subset_invariance']['status'] == 'PASS'
    assert not manifest['recovery']['recovery_validation_issues']
    for name, digest in manifest['report_sha256'].items():
        assert sha(REPORTS / name) == digest, name
    latest = OUT / 'latest_results'
    latest.mkdir(exist_ok=True)
    for p in REPORTS.iterdir():
        if p.is_file(): shutil.copy2(p, latest / p.name)
    stamp = datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
    write('README.md', f'''# 心理學與精神醫學英中翻譯評測：專案封存

封存時間（台北）：{stamp}

**狀態：所有排定測試與授權重試已結束；仍有 1 筆 Qwen 評分無效。**
這是完成執行並揭露缺失的研究封存，不代表每個計畫欄位都有有效分數。
Gemini 重試 23/23 有效；Qwen 唯一重試再次產生總分 90、分項加總 91，保留原樣且不納入有效分數。

## 從這裡開始

- [專案目的、方法與最新結果](PROJECT_REPORT.md)
- [執行過程與限制](PROCESS_AND_LIMITATIONS.md)
- [程式、環境與重現說明](REPRODUCIBILITY.md)
- [翻譯成果與各 judge 對照 CSV](latest_results/translation_comparison.csv)
- [最新統計報告](latest_results/analysis_report.md)
- [重試前後與同子集驗證](latest_results/recovery_before_after.md)
- `project_files.zip`：完整專案工作檔、原始回覆、素材、舊報告、程式、模板與設定快照。
- `FILE_MANIFEST.csv`：ZIP 內每一檔案的相對路徑、大小、SHA-256。
- `DELIVERY_SHA256.csv`：交付檔案校驗值；NAS 驗證另見 `TRANSFER_VERIFICATION.json`。

解壓縮後會看到五個原始專案資料夾。歷史報告是當時快照；最新結論以本目錄及 `latest_results/` 為準。
未複製 GGUF 權重、憑證、工作區其他專案或可重建快取；模型資訊與來源依專案 inventory/metadata 保留。
沒有新生成 PDF；ZIP 中可能保留先前已存在的 PDF 素材或歷史輸出。
''')
    table = ['|模型|文字平均分（29篇、5 judges）|圖片平均分（4張、4 judges）|', '|---|---:|---:|']
    means = rows(REPORTS / 'common_case_model_summary.csv')
    imgs = {r['model_id']:r for r in means if r['track']=='image'}
    for r in sorted((r for r in means if r['track']=='text'), key=lambda r:float(r['common_case_panel_mean']), reverse=True):
        table.append(f"|{r['model_name']}|{float(r['common_case_panel_mean']):.3f}|{float(imgs[r['model_id']]['common_case_panel_mean']):.3f}|")
    write('PROJECT_REPORT.md', '''# 專案目的、方法與結果

## 目的與範圍

比較 TranslateGemma 27B、12B、4B，以及 Gemma 4 E4B、12B、26B、31B QAT 的心理學／精神醫學英文轉繁體中文品質與效率，包含圖片內文字翻譯。
最終用途是輔助選擇可由 llama.cpp 提供服務、再接 LiteLLM／Open WebUI 的模型。測試結果不等同部署驗收或臨床安全認證。

## 資料與實驗設計

本輪有 30 篇文字（12 個來源頁）及 4 張圖片，每項由 7 個模型翻譯，共 238 份固定候選輸出。
來源包括 NIMH、NLM、CDC、VA，以及 WHO PDF 圖片；圖片另有自擬臨床筆記與用藥標籤。
來源 URL、引用、擷取方式、授權與素材 hash 見 ZIP 的材料 manifest、materials_audit.md 與原始來源檔。
WHO 圖片參考文字為機器擷取／AI 視覺檢查，並非人工 gold standard。本輪沒有加入人工評分。

五位 judge：Qwen 3.8 Flash Next、NVIDIA Nemotron 3.5 Super VL、Mistral Medium 3.5、GPT-6.1 SOL、Gemini 3.8 Flash High。
本機 judge 使用已驗證 thinking 設定；雲端沿用各既定路徑。Mistral 本次沒有原生圖像路徑，故圖片 panel 固定為其他四位，不補零。
來源、候選、blind ID、A/B 順序與提示均凍結。分項為術語／地區用語 30、語義／安全 30、流暢 20、完整／格式 20，總分須與分項相等。
錯誤標註另保留類型、嚴重度、來源證據、目標引文及信心。結構有效不代表每個錯誤指控都正確；證據診斷與純量有效性分開記錄。
兩兩比較包含 A/B 與 B/A，方向不一致標記 ABSTAIN；屬固定 spanning-chain 稽核，不是所有模型完整 round-robin。

## 完成數

|Judge|有效單筆評分|有效兩兩比較方向|
|---|---:|---:|
|Qwen|237/238|72/72|
|Nemotron|238/238|72/72|
|Mistral|210/210（僅文字）|48/48（僅文字）|
|GPT-6.1 SOL|238/238|72/72|
|Gemini|238/238|72/72|

共 1161 份有效單筆評分、336 個有效比較方向；1 筆無效、0 筆待處理，另有 Mistral 52 個設計上不支援的圖片欄位（28 單筆＋24 比較方向）。
Qwen 重試仍為 90 ≠ 27+29+17+18=91；沒有手動修正或反覆重試到成功。

## 最新描述性比較

''' + '\n'.join(table) + '''

文字共同集排除 text-15，以相同 29 篇比較所有候選，形成 203 個 candidate×item 單位、每單位 5 judges。整體原設計為 210 單位，其中 209 個可取得完整五評分；共同集為了各候選等量比較而整篇排除。
圖片為完整 28 個 candidate×item 單位、每單位 4 judges。部分 CSV 仍用 common-case 欄位標籤，其圖片完整性應以 reliability_summary 的 FULL_COMPLETE_BALANCED 為準。
Gemma 4 26B 文字均分最高，31B QAT 非常接近；這不是顯著差異證明。圖片本小樣本由 31B QAT 均分最高，不能外推為一般 OCR 準確率。

## 信度與不確定性

|分析集|ICC(2,1)|ICC(2,k)|k|系統層級 Kendall W|
|---|---:|---:|---:|---:|
|文字 29 篇|0.581387|0.874122|5|0.848571|
|圖片 4 張|0.513373|0.808424|4|0.852679|

採雙向隨機、絕對一致性 ICC。ICC(2,k) 代表 panel 平均分的信度，不能當單一 judge 信度。ICC 信賴區間未計算；文字模型均分另有以 12 來源頁為 cluster 的 10,000 次 bootstrap，勿與 ICC CI 混淆。圖片樣本太少且來源相關，不計圖片 CI。
原先 19 篇文字／2 張圖片共同子集重新計算完全一致：文字 ICC(2,5)=0.833384、圖片 ICC(2,4)=0.441100。擴充後數值變化反映樣本組成，不代表 judge 本身能力改善。
同家族依賴不能靠增加相近模型數解決；本案以跨家族 panel、leave-one-judge/family-out 敏感度分析檢查穩定性，未宣稱已統計消除所有家族偏差。

## 效率與部署解讀

翻譯速度與 judge 評分速度分開；完整分布見 translation_generation_summary.csv、judge_summary.csv。原 30 篇文字 warm request 中位數約：26B 0.686s、31B QAT 2.511s、12B 1.248s、E4B 0.751s、TG27B 3.850s、TG12B 1.850s、TG4B 0.759s。
這些是本機既定環境請求時間，不含冷載入，也不是跨硬體通用效能宣稱。Mistral 評分較慢（單筆中位數約 181s、p95 約 843s），但未因速度而刪除結果。
模板、實際 request render、thinking 與原生圖片證據見 technical_smoke、template_audit、local_runner 與 raw preflight。正式 LiteLLM／Open WebUI 串接尚未完成端到端驗證，不建議直接把研究設定視為正式部署核准。

## 方法來源與引用

- ICC 定義與解讀：[Koo & Li (2016)](https://pubmed.ncbi.nlm.nih.gov/27330520/)。
- 詳細 MQM、人工評估、兩兩比較、偏差控制及 reviewer reliability 設計來源，見 ZIP 中 `translation_benchmark_20260916_luna/multi_judge_evaluation_protocol.md`。
- 本輪執行規約見 `translation_benchmark_20261023_new_materials/judge_evaluation/protocol.md` 及 `analysis_runner/analysis_policy_amendment.md`。
- 人工審查框架已在上述方法文檔中說明，但本次未執行人工評分，也未計算 embedding／reranker 為主要品質結論。
''')
    write('PROCESS_AND_LIMITATIONS.md', '''# 過程與已知限制

1. 早期七模型翻譯、模板稽核與部署格式探索存於 20260916_luna；歷史報告保留，不回寫。
2. 20261022 進行 judge pilot、思考模式及錯誤格式排查，建立跨家族信度與 AB/BA 設計。應參考 local_thinking_combined_diagnostic_20261023，勿誤用較早 pilot 分數。
3. 20261023 凍結新素材與 238 份候選，建立每位 judge 的盲評封包，依原始模板與模式執行。
4. 本機大模型採排程與記憶體檢查；雲端 GPT 使用 Codex CLI、Gemini 使用 Antigravity 原生 UI。LiteLLM 依使用者指示暫停。
5. 初始批次完成後保留所有有效資料，僅對 24 個授權失敗／缺失項目重試：Gemini 23、Qwen 1。
6. Gemini 23 筆回覆皆結構有效。整合器曾誤把 missing parent 的 null attempt 視為錯誤，另有舊路徑重複前綴；依 provenance 修復驗證，沒有改動原始回覆或分數。
7. Qwen 原本等待 model+mmproj+32GiB。使用者於台北 2026-10-25 指示強制開始並由 root 直接執行，本次取消 32GiB 預留，仍核對模型大小、獨占隔離 port 與既定生成設定。14:15 左右載入，14:17 左右完成回覆並清理；這項操作偏離原記憶體政策，已記錄於 recovery manifest。未調低量化、token budget 或 thinking。
8. Qwen 新回覆仍是總分加總不符，判 invalid。沒有為了補齊資料再要求模型改分。整體全部 terminal，但並非全欄位 valid。
9. 最終彙整曾將 batch_manifest 誤收為 attempt；已排除特定 schema 的無評分執行清單，仍保留在 hash inventory。實際模型回覆不受此修正影響。
10. 最新報告另存 post_recovery_reports；原 baseline、原有效分數、候選輸出及原始回覆均保留。同子集數值不變檢查 PASS；recovery tests 5/5、aggregator tests 22/22 PASS。

## 必須保留的限制

- 文字只有 30 篇、圖片只有 4 張，領域與版式涵蓋有限；一筆無效使共同文字集為 29 篇。
- 五 judge 評分不是人工 gold standard；ICC 高不保證正確，也不等於臨床安全。
- 圖片結果是翻譯品質評分，不是逐字 OCR 準確率；WHO 參考內容未經人工標準答案審核。
- 兩兩比較不是完整全配對；不從不完整比較圖推導完整勝率排名。
- 部分原始報告的 checkpoint missing 表示歷史宣告，最新選定結果已由 recovery 補齊；請以最新 selected rows 為準。
- Raw 內本機絕對路徑與當時程序 PID 是歷史證據，不代表目前仍在執行。
- 引文匹配警示、英文保留旗標與 critical 指控須保留為獨立診斷，不自動改純量分數。
- 封存時未重新呼叫模型、未再啟動 LiteLLM、未更動正式服務。
''')
    write('REPRODUCIBILITY.md', '''# 程式與重現說明

## 封存結構

解壓 `project_files.zip` 後：

- `translation_benchmark_20260916_luna/`：原始比較、七模型報告產生器、模板、GGUF metadata、方法文獻。
- `translation_benchmark_20261022_judge_pilot/`：pilot、thinking／format recovery、信度與 judge 校準程式及原始證據。
- `translation_benchmark_20261023_new_materials/`：凍結素材、翻譯 driver、238 個輸出、技術 smoke、judge public/private mappings、local/cloud runner、analysis runner、測試與所有 raw。
- `translation_eval_materials_20261022/`、`translation_materials_psych_20261022_luna/`：素材候選、來源 manifest、收集說明。
- `AGENTS.md`、`translategemma-en-zh-tw.jinja`：原工作區執行規範與翻譯模板。
- `archive_translation_project.py`：本封存建立程式。

## 主要程式入口

新素材專案的根目錄 driver／README 為翻譯入口；judge_evaluation/build_packets.py 建立封包，local_runner/judge_driver.py 執行本機，cloud_runner/ 保存 GPT CLI 與 Gemini UI 工作流。
analysis_runner/aggregate_new_panel.py 是原始批次彙整；**最新重試資料必須使用 recovery_20261025_failed_jobs/analysis/post_recovery_runner.py**。
retry_failed_job.py 與 recovery_controller.py 是此次單筆操作工具，保留作稽核，不應未審核就再次啟動。

## 離線驗證

程式保留原始 `/benchmark` 絕對路徑，raw 也以此記錄來源。封存 ZIP 不是可任意搬移後直接重跑的 turnkey package。
可在隔離機器相同路徑解壓，或在副本中一致調整程式根路徑；不要修改原始證據檔內路徑／hash。
原主機可執行：

```sh
cd /benchmark
python3 translation_benchmark_20261023_new_materials/judge_evaluation/recovery_20261025_failed_jobs/analysis/post_recovery_runner.py --status
python3 translation_benchmark_20261023_new_materials/judge_evaluation/recovery_20261025_failed_jobs/analysis/test_post_recovery.py
python3 -m unittest discover -s translation_benchmark_20261023_new_materials/judge_evaluation/analysis_runner/tests -p 'test_*.py'
```

`--refresh` 只重建獨立 post_recovery_reports，不呼叫模型。仍應先備份派生輸出；封存 hash 檔不會自動更新。
最新 CSV 可用標準 UTF-8 CSV reader 開啟；逐筆對照含 source、translation、judge 分數與有效性。原始盲碼對照為 private/ 下的實驗 keymap，不是 API key。

## 若要重新執行模型

需要另外取得相同 GGUF/mmproj 權重與已驗證模板、macOS Metal 環境、足夠 RAM，以及 llama.cpp 0.7.0-dev build 524 / commit 620d2225c。
權重路徑、分片／mmproj hash、runtime command 與生成參數保存在 model_inventory、server_preflight、settings_provenance 及 runner config。第三方 runtime 與權重未複製到本研究封存。
本機參數包括 context 32768、max output 16384、reasoning budget 8192、temperature 0、top_p 1、seed 20261023、parallel 1、cache/speculative off；各模型 thinking 欄位依原 config。
雲端模型可用性、版本及 UI 會改變；需原有帳戶登入。封存不含登入資料、API key 或密碼。
禁止將本次強制載入的 0GiB margin 視為常態建議；重新測試應預設恢復 32GiB margin。
要接正式 llama.cpp → LiteLLM → Open WebUI，必須另做端到端模板／圖片請求驗證。

## 校驗

FILE_MANIFEST.csv 可逐檔核對 ZIP 成員 SHA-256；DELIVERY_SHA256.csv 核對本資料夾交付檔案。TRANSFER_VERIFICATION.json 記錄實際 NAS 讀回比對結果。
cache、鎖檔、live guard、憑證及 GGUF 不屬交付程式，排除項目見 EXCLUDED_FILES.csv。歷史 raw 不會為了封存重新格式化或修改。
''')
    files=[]; excluded=[]
    for project in PROJECTS:
        for p in sorted((ROOT/project).rglob('*')):
            if not p.is_file(): continue
            rel=p.relative_to(ROOT)
            if p.is_symlink() or any(x in {'__pycache__','.git','.DS_Store'} for x in rel.parts) or p.suffix in {'.pyc','.gguf','.key'} or p.name in {'.refresh.lock','controller.guard.json'}:
                excluded.append((str(rel),'cache/lock/guard/credential/model/symlink')); continue
            if p.name.startswith('.env'):
                excluded.append((str(rel),'environment credentials')); continue
            files.append(p)
    files += [ROOT/n for n in ('AGENTS.md','translategemma-en-zh-tw.jinja','archive_translation_project.py')]
    with (OUT/'EXCLUDED_FILES.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['path','reason']);w.writerows(excluded)
    inventory=[]
    with zipfile.ZipFile(OUT/'project_files.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            rel=str(p.relative_to(ROOT)); digest=sha(p)
            z.write(p,rel);inventory.append((rel,p.stat().st_size,digest))
    with (OUT/'FILE_MANIFEST.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['path','bytes','sha256']);w.writerows(inventory)
    with zipfile.ZipFile(OUT/'project_files.zip') as z:
        for rel,size,digest in inventory:
            assert hashlib.sha256(z.read(rel)).hexdigest()==digest,rel
    with (OUT/'DELIVERY_SHA256.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['path','bytes','sha256'])
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name not in {'DELIVERY_SHA256.csv','TRANSFER_VERIFICATION.json'}:
                w.writerow([str(p.relative_to(OUT)),p.stat().st_size,sha(p)])
    print(json.dumps({'output':str(OUT),'archived_files':len(inventory),'excluded':len(excluded),'zip_bytes':(OUT/'project_files.zip').stat().st_size,'zip_hash_verified':True},ensure_ascii=False))

if __name__=='__main__': main()
