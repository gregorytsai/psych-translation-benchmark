#!/usr/bin/env python3
"""Build the deployment-valid Markdown and DOCX reports.

This builder reads deployment_scores.json and the five corrected OAI raw JSONL
files only.  It deliberately never imports or reads the historical scores as
an input to rankings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import build_report as _style_helpers

# Use a CJK font that is present in the LibreOffice rendering environment.
# The historical builder's Arial Unicode MS fallback rendered Chinese as
# missing-glyph boxes in the deployment report, so set the helper module's
# shared font before importing its style functions.
_style_helpers.FONT = "Heiti TC"

from build_report import (
    LIGHT_BORDER,
    MID,
    NAVY,
    PALE,
    add_body,
    add_bullet,
    add_hyperlink,
    add_page_number,
    add_table,
    clean_output,
    keep_table_row_together,
    remove_paragraph_borders,
    set_run_font,
)

FONT = "Heiti TC"


ROOT = Path(__file__).resolve().parent
OUT_MD = ROOT / "report_deployment.md"
OUT_DOCX = ROOT / "deployment_translation_benchmark_report.docx"

CASE_TITLES = {
    "panic_disorder": "恐慌症",
    "depression_behavioral_activation": "重度憂鬱症與行為啟動",
    "ssri_pharmacology": "SSRI 與血清素症候群",
}
CASE_ORDER = ["panic_disorder", "depression_behavioral_activation", "ssri_pharmacology"]
ALL_CASE_ORDER = CASE_ORDER + ["image_clinical_note"]


def load_raw_outputs(scores: dict) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    for model, info in scores["models"].items():
        rows = [json.loads(line) for line in Path(info["source_raw"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        outputs[model] = {
            row["id"]: row["response"]["choices"][0]["message"]["content"]
            for row in rows
        }
    return outputs


def md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def build_markdown(scores: dict, audit: dict, outputs: dict[str, dict[str, str]]) -> str:
    lines: list[str] = []
    lines.append("# 部署有效翻譯模型評測報告")
    lines.append("")
    lines.append("**評測日期** 2026-09-16　　**評閱** Luna max　　**工具鏈** llama.cpp b10964 → LiteLLM → Open WebUI")
    lines.append("")
    lines.append("## 結論")
    lines.append("")
    lines.append("本報告只使用五份正確 OAI/template raw JSONL，沒有把歷史手工 prompt 分數混入正式排名。以文字平均與單一圖片翻譯準確率各占一半的描述性平均排序，Gemma 4 E4B 排名第一（95.34），TranslateGemma 27B 第二（95.00）。若只看文字，Gemma 4 12B 第一（96.67）；若只看圖片翻譯，Gemma 4 E4B 第一（97）。")
    lines.append("")
    lines.append("這個合併平均只用來提供跨模態的閱讀順序，不是新的原始 rubric，也不是 OCR 分數。圖片 raw 沒有獨立逐字 OCR channel，因此本報告使用「圖片翻譯準確率」，不宣稱 OCR exact。")
    lines.append("")
    lines.append("## 最終排名")
    lines.append("")
    lines.append("| 排名 | 模型 | 文字平均 | 圖片翻譯準確率 | 描述性平均 | template 狀態 | 可部署 |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for item in scores["final_ranking"]:
        model = item["model"]
        info = scores["models"][model]
        lines.append(f"| {item['rank']} | {model} | {item['text_average']:.2f} | {item['image_translation_accuracy']} | {item['combined_mean_for_ranking_only']:.2f} | {item['template_status']} | {'是' if info['candidate_deployable'] else '否'} |")
    lines.append("")
    lines.append("## 匿名評分與 mapping")
    lines.append("")
    lines.append("評分前以固定 seed `20260916` 將五個來源隨機排列，先以 M01-M05 匿名代號閱讀與評分，再重新開啟 mapping。這降低了先看到模型名稱對術語偏好的影響；mapping 與 raw 檔案仍完整記錄在 [deployment_scores.json](/benchmark/translation_benchmark_20260916_luna/deployment_scores.json:1)。")
    lines.append("")
    lines.append("| 匿名代號 | 評分順序 | 實際模型 | raw JSONL |")
    lines.append("|---|---:|---|---|")
    for label in scores["blind_review"]["label_order"]:
        m = scores["blind_review"]["mapping"][label]
        lines.append(f"| {label} | {scores['blind_review']['label_order'].index(label)+1} | {m['model']} | [{Path(m['raw_file']).name}]({m['raw_file']}:1) |")
    lines.append("")
    lines.append("## 評分方法")
    lines.append("")
    lines.append("文字每案 100 分：術語與領域準確度 30、語義忠實度 30、台灣正體中文流暢度 20、完整性與格式 20。每一筆具體扣分理由都在下方表格及 JSON。圖片因只有翻譯輸出，改為內容保留 40、藥名劑量數字 30、台灣正體自然度 30，合計圖片翻譯準確率 100 分。")
    lines.append("")
    lines.append("評分只讀以下五份來源：" + "、".join(f"[{Path(info['source_raw']).name}]({info['source_raw']}:1)" for info in scores["models"].values()) + "。")
    lines.append("")
    lines.append("## 文字細項分數")
    lines.append("")
    lines.append("| 模型 | 案例 | 術語 30 | 語義 30 | 流暢 20 | 完整格式 20 | 總分 | 具體錯誤 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for rank in sorted(scores["final_ranking"], key=lambda x: x["rank"]):
        model = rank["model"]
        blind = scores["models"][model]["blind_id"]
        for case_id in CASE_ORDER:
            item = scores["blind_review"]["blinded_score_matrix"][blind]["text"][case_id]
            lines.append(f"| {model} | {CASE_TITLES[case_id]} | {item['terminology']} | {item['semantic_fidelity']} | {item['traditional_fluency']} | {item['completeness_format']} | {item['total']} | {md_cell('；'.join(item['errors']))} |")
    lines.append("")
    lines.append("## 三段文字完整譯文比較")
    lines.append("")
    for case_id in CASE_ORDER:
        lines.append(f"### {CASE_TITLES[case_id]}")
        lines.append("")
        lines.append("| 模型 | 總分 | 完整譯文 |")
        lines.append("|---|---:|---|")
        for rank in sorted(scores["final_ranking"], key=lambda x: x["rank"]):
            model = rank["model"]
            blind = scores["models"][model]["blind_id"]
            item = scores["blind_review"]["blinded_score_matrix"][blind]["text"][case_id]
            lines.append(f"| {model} | {item['total']} | {md_cell(item['output'])} |")
        lines.append("")
    lines.append("## 圖片翻譯比較")
    lines.append("")
    lines.append("測試圖是本機合成的精神科追蹤文字。五份新 raw 都只有翻譯，沒有獨立 OCR 文字；所以表格不列 OCR exactness，改列內容保留、藥名劑量數字與正體自然度。")
    lines.append("")
    lines.append("| 模型 | 內容 40 | 藥名劑量數字 30 | 正體自然度 30 | 圖片翻譯準確率 | 具體錯誤 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for rank in sorted(scores["final_ranking"], key=lambda x: x["rank"]):
        model = rank["model"]
        blind = scores["models"][model]["blind_id"]
        item = scores["blind_review"]["blinded_score_matrix"][blind]["image"]
        lines.append(f"| {model} | {item['content_retention_40']} | {item['medication_dose_numbers_30']} | {item['traditional_naturalness_30']} | {item['total']} | {md_cell('；'.join(item['errors']))} |")
    lines.append("")
    lines.append("### 圖片完整譯文")
    lines.append("")
    for rank in sorted(scores["final_ranking"], key=lambda x: x["rank"]):
        model = rank["model"]
        blind = scores["models"][model]["blind_id"]
        item = scores["blind_review"]["blinded_score_matrix"][blind]["image"]
        lines.append(f"**{model}**（{item['total']}）")
        lines.append("")
        lines.append("```text")
        lines.append(item["output"])
        lines.append("```")
        lines.append("")
    lines.append("## 速度統計")
    lines.append("")
    lines.append("下表由各 raw 的 `__verbose.timings` 與 API wall elapsed 計算。本次是 warm request：server 已在 request 前完成模型載入，API elapsed 不含 cold startup/load；cold timing 未測量。每個檔案四案在同一個 sequential slot 執行，prefix/KV cache 狀態會影響 prompt 速度，因此 generation tok/s 比 prompt tok/s 更可比。")
    lines.append("")
    lines.append("| 模型 | 文字 API 平均秒 | 圖片 API 秒 | 文字 prompt tok/s | 圖片 prompt tok/s | 文字 generation tok/s | 圖片 generation tok/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for rank in sorted(scores["final_ranking"], key=lambda x: x["rank"]):
        model = rank["model"]
        b = scores["blind_review"]["blinded_score_matrix"][scores["models"][model]["blind_id"]]
        ta, im = b["timings"]["text_average"], b["timings"]["image"]
        lines.append(f"| {model} | {ta['api_elapsed_seconds']:.3f} | {im['api_elapsed_seconds']:.3f} | {ta['prompt_tok_s']:.3f} | {im['prompt_tok_s']:.3f} | {ta['generation_tok_s']:.3f} | {im['generation_tok_s']:.3f} |")
    lines.append("")
    lines.append("## Template 與部署狀態")
    lines.append("")
    lines.append("| 模型 | template 狀態 | 可部署 | 條件 |")
    lines.append("|---|---|---|---|")
    for item in scores["final_ranking"]:
        model = item["model"]
        status = next(x for x in audit["deployment_status"] if x["model"] == model)
        condition = {
            "translategemma-4b": "custom en→zh-TW adapter、matching mmproj、ubatch 512",
            "translategemma-12b": "custom en→zh-TW adapter、matching mmproj、ubatch 2048",
            "translategemma-27b": "外掛 custom adapter（GGUF 無 template）、matching mmproj、ubatch 2048",
            "gemma-4-e4b": "embedded official template、mmproj、deepseek、enable_thinking=false、ubatch 4096",
            "gemma-4-12b": "embedded official template、mmproj、deepseek、enable_thinking=false、ubatch 4096",
        }[model]
        lines.append(f"| {model} | {status['template_status']} | {'是' if status['candidate_deployable'] else '否'} | {condition} |")
    lines.append("")
    lines.append("TranslateGemma 的官方 template 需要 user content item 內的 `source_lang_code` 與 `target_lang_code`；llama.cpp b10964 OAI normalizer 只保留 `text` / `media_marker`，因此三個 TG alias 固定為 English → Traditional Chinese。Gemma4 12B 的 empty thought block 必須由 `reasoning-format=deepseek` 解析，不能用 `none`。")
    lines.append("")
    lines.append("## Owner chain")
    lines.append("")
    lines.append("Open WebUI 只送 OpenAI messages 到 LiteLLM；LiteLLM 以 `openai/<llama-server-model-id>` 轉發 `/v1/chat/completions`；llama.cpp 是唯一 template owner，並負責 media marker → matching mmproj → image token。任何上游都不可預渲染 `<start_of_turn>`、`<|turn>` 或 `<|channel>`。")
    lines.append("")
    lines.append("Open WebUI base URL：`http://litellm:4000/v1`。LiteLLM upstream：`http://llama-server:8080/v1`。圖片使用標準 `content=[{\"type\":\"text\"...},{\"type\":\"image_url\",\"image_url\":{\"url\":...}}]`；container 間不要使用 127.0.0.1。完整候選檔是 [litellm.audit.yaml](/benchmark/translation_benchmark_20260916_luna/litellm.audit.yaml:1)。")
    lines.append("")
    lines.append("## 舊結果作廢")
    lines.append("")
    lines.append("舊 `run_text_benchmark.sh` 使用 b10679 `llama-completion` 並手工拼 `<start_of_turn>`、`<|turn>`、`<|channel>final`，不能證明 b10964 server 只套一次 template。舊 TranslateGemma 圖片使用 `--no-jinja --chat-template gemma`，不是官方 TG template 或本次 adapter。舊 Gemma4 圖片也走 CLI/mtmd 而非 OAI server，12B 另有 `ubatch=512` assertion 風險。舊分數保留作歷史資料，但不參與本報告任何分數、平均或排名。")
    lines.append("")
    lines.append("## 候選設定與重現")
    lines.append("")
    lines.append("完整 llama.cpp preset：[models.audit.ini](/benchmark/translation_benchmark_20260916_luna/models.audit.ini:1)。關鍵設定是 TG 使用 custom adapter；Gemma4 使用 embedded template、matching mmproj、`reasoning-format=deepseek`、`chat-template-kwargs={\"enable_thinking\":false}`，Gemma4 圖片使用 `ubatch-size=4096`。完整原始輸出與 metadata 請見 [template_audit.md](/benchmark/translation_benchmark_20260916_luna/template_audit.md:1) 與 [template_audit.json](/benchmark/translation_benchmark_20260916_luna/template_audit.json:1)。")
    lines.append("")
    lines.append("重現核心命令：")
    lines.append("")
    lines.append("```bash")
    lines.append("/opt/homebrew/bin/llama-server --models-preset /benchmark/translation_benchmark_20260916_luna/models.audit.ini --models-max 1 --no-models-autoload --host 127.0.0.1 --port 18085 --no-webui --jinja --no-warmup")
    lines.append("python3 /benchmark/translation_benchmark_20260916_luna/run_template_audit.py --base http://127.0.0.1:<isolated-port> --model <alias> --reasoning-format deepseek --user-instruction 'Translate the English text into Taiwan Traditional Chinese. Output only the translation; do not summarize, explain, or add safety advice.' --out <raw JSONL>")
    lines.append("PY=/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3; $PY score_deployment.py")
    lines.append("PY=/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3; $PY build_deployment_report.py")
    lines.append("```")
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    for item in scores["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("正式服務檔案沒有由本階段修改或重啟。這份報告可用於部署前審核，不取代臨床人工覆核、較大測試集或 streaming/tool path 的另行驗證。")
    lines.append("")
    return "\n".join(lines)


def add_heading_black(doc: Document, text: str, level: int):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        set_run_font(run, size={1: 17, 2: 13, 3: 11.5}.get(level, 11), bold=True, color="000000")
    return p


def set_doc_style(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.font.size = Pt(10.5)
    for style_name, size, before, after in (("Title", 23, 0, 12), ("Heading 1", 17, 14, 7), ("Heading 2", 13, 10, 5), ("Heading 3", 11.5, 8, 4)):
        st = styles[style_name]
        st.font.name = FONT
        st._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True
        if style_name == "Title":
            style_p_pr = st._element.get_or_add_pPr()
            style_p_bdr = style_p_pr.find(qn("w:pBdr"))
            if style_p_bdr is not None:
                style_p_pr.remove(style_p_bdr)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = header.add_run("部署有效翻譯評測  2026 年 9 月 16 日")
    set_run_font(r, size=8.3, color=MID)
    add_page_number(section.footer.paragraphs[0])


def add_docx_full_translation_table(doc: Document, case_id: str, scores: dict, outputs: dict) -> None:
    add_heading_black(doc, CASE_TITLES[case_id], 2)
    rows = []
    for rank in sorted(scores["final_ranking"], key=lambda x: x["rank"]):
        model = rank["model"]
        blind = scores["models"][model]["blind_id"]
        item = scores["blind_review"]["blinded_score_matrix"][blind]["text"][case_id]
        rows.append([model, item["output"], f"{item['total']}\n" + "\n".join(item["errors"])])
    table = add_table(doc, ["模型", "完整譯文", "總分與錯誤"], rows, [1.30, 4.15, 1.40], font_size=8.35)
    for row in table.rows[1:]:
        keep_table_row_together(row)


def build_docx(scores: dict, audit: dict, outputs: dict[str, dict[str, str]]) -> None:
    doc = Document()
    set_doc_style(doc)
    title = doc.add_paragraph(style="Title")
    remove_paragraph_borders(title)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(72)
    r = title.add_run("部署有效翻譯模型評測報告")
    set_run_font(r, size=23, bold=True, color="000000")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(13)
    r = subtitle.add_run("llama cpp LiteLLM Open WebUI")
    set_run_font(r, size=13, color=MID)
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(25)
    r = meta.add_run("評測日期  2026 年 9 月 16 日\n評閱  Luna max\n資料來源  五份正確 OAI template raw JSONL")
    set_run_font(r, size=10.2, color=MID)
    lead = doc.add_paragraph()
    lead.paragraph_format.left_indent = Inches(0.4)
    lead.paragraph_format.right_indent = Inches(0.4)
    lead.paragraph_format.line_spacing = 1.28
    lead.paragraph_format.space_after = Pt(10)
    r = lead.add_run("結論  ")
    set_run_font(r, size=12, bold=True)
    r = lead.add_run("Gemma 4 E4B 在文字與圖片的描述性合併平均最高（95.34），TranslateGemma 27B 緊接其後（95.00）。Gemma 4 12B 的文字平均最高（96.67），但只有在 deepseek reasoning parser 與 ubatch 4096 下才是乾淨的部署路徑。這些名次只根據新 OAI/template raw，舊手工 prompt 分數不參與排名。")
    set_run_font(r, size=11.2)
    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = note.add_run("圖片 raw 沒有獨立 OCR channel，本報告只報圖片翻譯準確率，不報 OCR exactness。")
    set_run_font(r, size=9.5, italic=True, color=MID)
    doc.add_page_break()

    add_heading_black(doc, "結論與排名", 1)
    add_body(doc, "本報告以正確的 OpenAI-compatible server request 為品質來源。每個模型都完成 3 段文字與 1 張圖片測試。描述性合併平均把文字平均與圖片翻譯準確率各占一半；圖片只有一個合成案例，因此不應解讀成全面能力分數。")
    ranking_rows = []
    for item in scores["final_ranking"]:
        model = item["model"]
        info = scores["models"][model]
        ranking_rows.append([str(item["rank"]), model, f"{item['text_average']:.2f}", str(item["image_translation_accuracy"]), f"{item['combined_mean_for_ranking_only']:.2f}", item["template_status"], "是" if info["candidate_deployable"] else "否"])
    add_table(doc, ["排名", "模型", "文字平均", "圖片翻譯", "描述性平均", "template", "可部署"], ranking_rows, [0.45, 1.55, 0.75, 0.80, 0.85, 2.00, 0.55], font_size=8.6)
    add_body(doc, "只看文字時，Gemma 4 12B 為 96.67 分，TranslateGemma 27B 為 94.00 分，Gemma 4 E4B 為 93.67 分。只看圖片翻譯時，Gemma 4 E4B 為 97 分，TranslateGemma 27B 為 96 分。TranslateGemma 4B 的主要扣分來自把 panic disorder 翻成焦慮症；這是內容錯誤，不是 template 問題。")

    add_heading_black(doc, "匿名評分方法", 1)
    add_body(doc, "評分前以 Random(20260916) 將五個 raw 來源隨機排列，先以 M01 到 M05 匿名代號評分，再依 mapping 還原模型名稱。完整 mapping、每案分項與評分理由保存在 deployment_scores.json。")
    mapping_rows = []
    for i, label in enumerate(scores["blind_review"]["label_order"], 1):
        m = scores["blind_review"]["mapping"][label]
        mapping_rows.append([label, str(i), m["model"], Path(m["raw_file"]).name])
    add_table(doc, ["匿名", "順序", "模型", "raw 檔"], mapping_rows, [0.65, 0.65, 1.75, 3.75], font_size=8.8)
    add_body(doc, "文字 rubric 為術語與領域準確度 30、語義忠實度 30、台灣正體中文流暢度 20、完整性與格式 20。圖片 raw 只有翻譯，因此改評內容保留 40、藥名劑量數字 30、台灣正體自然度 30；不宣稱 OCR exactness。")

    add_heading_black(doc, "文字分項分數", 1)
    score_rows = []
    for item in scores["final_ranking"]:
        model = item["model"]
        blind = scores["models"][model]["blind_id"]
        for case_id in CASE_ORDER:
            s = scores["blind_review"]["blinded_score_matrix"][blind]["text"][case_id]
            score_rows.append([model, CASE_TITLES[case_id], str(s["terminology"]), str(s["semantic_fidelity"]), str(s["traditional_fluency"]), str(s["completeness_format"]), str(s["total"])])
    score_table = add_table(doc, ["模型", "案例", "術語", "語義", "流暢", "完整格式", "總分"], score_rows, [1.45, 1.35, 0.52, 0.52, 0.52, 0.78, 0.55], font_size=8.0)
    for row in score_table.rows[1:]:
        keep_table_row_together(row)
    add_body(doc, "主要品質差異是可觀察的詞彙與臨床語義問題。TG4 在恐慌症案例將 panic disorder / panic attacks 翻成焦慮症 / 焦慮發作；TG12 在 SSRI 案例將 agitation 翻成焦慮；E4B 在 SSRI 案例也有相同的 agitation 近似錯誤。Gemma 4 12B 的錯誤較少，主要是部分術語與語域選擇。")

    add_heading_black(doc, "三段文字完整譯文比較", 1)
    add_body(doc, "以下為五份 corrected OAI/template raw 的完整 response content。沒有加入舊 scores，也沒有以舊 CLI 輸出替換這些內容。")
    for case_id in CASE_ORDER:
        add_docx_full_translation_table(doc, case_id, scores, outputs)

    add_heading_black(doc, "圖片翻譯評分", 1)
    add_body(doc, "圖片案例是本機合成的精神科追蹤文字。新 raw 每案只有翻譯 output，沒有獨立 OCR channel，因此下表使用圖片翻譯準確率，不填 OCR exactness。")
    image_rows = []
    for item in scores["final_ranking"]:
        model = item["model"]
        blind = scores["models"][model]["blind_id"]
        s = scores["blind_review"]["blinded_score_matrix"][blind]["image"]
        image_rows.append([model, str(s["content_retention_40"]), str(s["medication_dose_numbers_30"]), str(s["traditional_naturalness_30"]), str(s["total"])])
    add_table(doc, ["模型", "內容保留 40", "藥名劑量數字 30", "正體自然度 30", "準確率"], image_rows, [1.65, 1.0, 1.35, 1.25, 0.75], font_size=8.5)
    for item in scores["final_ranking"]:
        model = item["model"]
        blind = scores["models"][model]["blind_id"]
        s = scores["blind_review"]["blinded_score_matrix"][blind]["image"]
        add_heading_black(doc, model, 3)
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(s["output"])
        set_run_font(r, size=9.3)
        add_body(doc, "；".join(s["errors"]))

    add_heading_black(doc, "速度統計", 1)
    add_body(doc, "數值來自每份 raw JSONL 的 __verbose.timings 與 API elapsed_seconds。本次是 warm request：模型在 request 前已載入，因此 API elapsed 不含 cold startup/load；cold timing 未測量。每個模型四案在同一個 sequential slot 執行，cache tokens 與共同 prefix 會影響 prompt 速度，generation tok/s 比 prompt tok/s 更可比。")
    timing_rows = []
    for item in scores["final_ranking"]:
        model = item["model"]
        b = scores["blind_review"]["blinded_score_matrix"][scores["models"][model]["blind_id"]]
        ta, im = b["timings"]["text_average"], b["timings"]["image"]
        timing_rows.append([model, f"{ta['api_elapsed_seconds']:.3f}", f"{im['api_elapsed_seconds']:.3f}", f"{ta['prompt_tok_s']:.1f}", f"{im['prompt_tok_s']:.1f}", f"{ta['generation_tok_s']:.1f}", f"{im['generation_tok_s']:.1f}"])
    add_table(doc, ["模型", "文字 API 秒", "圖片 API 秒", "文字 prompt/s", "圖片 prompt/s", "文字生成/s", "圖片生成/s"], timing_rows, [1.55, 0.80, 0.80, 0.90, 0.90, 0.90, 0.90], font_size=8.0)

    add_heading_black(doc, "Template 與部署條件", 1)
    status_rows = []
    conditions = {
        "translategemma-4b": "custom en→zh-TW adapter；matching mmproj；ubatch 512",
        "translategemma-12b": "custom en→zh-TW adapter；matching mmproj；ubatch 2048",
        "translategemma-27b": "外掛 adapter（GGUF 無 template）；matching mmproj；ubatch 2048",
        "gemma-4-e4b": "embedded official；mmproj；deepseek；enable_thinking=false；ubatch 4096",
        "gemma-4-12b": "embedded official；mmproj；deepseek；enable_thinking=false；ubatch 4096",
    }
    for item in scores["final_ranking"]:
        model = item["model"]
        info = scores["models"][model]
        status_rows.append([model, item["template_status"], "是" if info["candidate_deployable"] else "否", conditions[model]])
    add_table(doc, ["模型", "template", "可部署", "必要條件"], status_rows, [1.45, 1.55, 0.65, 3.2], font_size=8.0)
    add_body(doc, "TranslateGemma 官方 template 要求 source_lang_code / target_lang_code，但 llama.cpp b10964 OpenAI parser 只保留 text / media_marker。三個 TG alias 因此固定 English → zh-TW，並使用 candidate adapter。TG27 的 GGUF 沒有 embedded tokenizer.chat_template。Gemma4 12B 的 empty thought channel 需用 reasoning-format=deepseek 解析；reasoning-format=none 會把 marker 留在 content。")
    add_body(doc, "隔離 server 實測五個 alias 均完成 4/4 HTTP 200，prompt 各只有一個 user/model turn；TG image prompt 有 start/end image，Gemma4 image prompt 有 image token pair。Gemma4 image 以 ubatch 512 實測 abort，4096 後通過。")

    add_heading_black(doc, "Owner chain", 1)
    add_body(doc, "Open WebUI 只送 OpenAI messages 到 LiteLLM。LiteLLM 以 openai/<llama-server-model-id> 映射並轉發 /v1/chat/completions。llama.cpp 是唯一 template owner，負責一次 Jinja、media marker 與 matching mmproj。任何上游都不可預渲染 turn/channel markers。Open WebUI base URL 應是 http://litellm:4000/v1；LiteLLM upstream 應是 http://llama-server:8080/v1。container 間不要使用 127.0.0.1。")
    add_body(doc, "TranslateGemma 需避免 system role；LiteLLM candidate 對三個 alias 設 supports_system_message=false。圖片使用標準 content text part 加 image_url part。Gemma4 翻譯 request 應附帶明確翻譯指令；streaming 與 tools 仍需在正式 proxy path 另行 smoke test。")

    add_heading_black(doc, "舊結果作廢原因", 1)
    add_body(doc, "舊文字 harness 使用 b10679 llama-completion 並手工加入模型 markers；舊 TranslateGemma 圖片使用 --no-jinja --chat-template gemma；舊 Gemma4 圖片使用 CLI/mtmd 而不是 OAI server，且 12B 使用 ubatch 512。這些輸出可以保留作歷史參考，但不能驗證本次部署鏈路，也沒有參與本報告分數與排名。")

    add_heading_black(doc, "限制", 1)
    for item in scores["limitations"]:
        add_bullet(doc, item)
    add_body(doc, "正式 models.ini、舊 custom template 與 launchd service 沒有由本階段修改或重啟。")

    add_heading_black(doc, "資料與重現", 1)
    paths = [
        ["deployment score", str(ROOT / "deployment_scores.json")],
        ["summary CSV", str(ROOT / "deployment_results_summary.csv")],
        ["report Markdown", str(OUT_MD)],
        ["template audit", str(ROOT / "template_audit.json")],
        ["llama candidate", str(ROOT / "models.audit.ini")],
        ["LiteLLM candidate", str(ROOT / "litellm.audit.yaml")],
        ["corrected raw JSONL", "五個 raw_template_audit_*.jsonl（路徑列於 deployment_scores.json）"],
    ]
    add_table(doc, ["資料", "路徑"], paths, [1.6, 5.25], font_size=8.5)
    add_body(doc, "完整來源 URL 與 template/parser 證據請見 template_audit.md 與 template_audit.json。品質報告不把舊 scores.json 或 manual_scores.json 當作正式來源。")

    doc.core_properties.title = "部署有效翻譯模型評測報告"
    doc.core_properties.subject = "llama.cpp LiteLLM Open WebUI deployment-valid translation evaluation"
    doc.core_properties.author = "Luna max"
    doc.save(OUT_DOCX)


def main() -> int:
    scores = json.loads((ROOT / "deployment_scores.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "template_audit.json").read_text(encoding="utf-8"))
    outputs = load_raw_outputs(scores)
    markdown = build_markdown(scores, audit, outputs)
    OUT_MD.write_text(markdown, encoding="utf-8")
    build_docx(scores, audit, outputs)
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_DOCX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
