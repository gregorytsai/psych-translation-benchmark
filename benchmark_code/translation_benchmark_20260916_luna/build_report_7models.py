#!/usr/bin/env python3
"""Build the seven-model deployment Markdown report and RFC4180 CSV.

The inputs are the five corrected raw OpenAI/template runs from the previous
phase plus the two new isolated-server runs.  No historical CLI scores are
loaded.  The score notes are keyed by model only inside this reproducible
builder; the report records the fixed random blind-label mapping used for the
review pass.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report_deployment_7models.md"
CSV_OUT = ROOT / "deployment_translation_results_7models.csv"

RAW_FILES = {
    "translategemma-4b": "raw_template_audit_tg4_v2.jsonl",
    "translategemma-12b": "raw_template_audit_tg12.jsonl",
    "translategemma-27b": "raw_template_audit_tg27.jsonl",
    "gemma-4-e4b": "raw_template_audit_gemma4_e4b.jsonl",
    "gemma-4-12b": "raw_template_audit_gemma4_12b_v2.jsonl",
    "gemma-4-26b": "raw_template_audit_gemma4_26b.jsonl",
    "gemma-4-31b-qat": "raw_template_audit_gemma4_31b_qat.jsonl",
}

DISPLAY_NAMES = {
    "translategemma-4b": "TranslateGemma 4B",
    "translategemma-12b": "TranslateGemma 12B",
    "translategemma-27b": "TranslateGemma 27B",
    "gemma-4-e4b": "Gemma 4 E4B",
    "gemma-4-12b": "Gemma 4 12B",
    "gemma-4-26b": "Gemma 4 26B",
    "gemma-4-31b-qat": "Gemma 4 31B QAT",
}

TEXT_CASES = [
    "panic_disorder",
    "depression_behavioral_activation",
    "ssri_pharmacology",
]
ALL_CASES = TEXT_CASES + ["image_clinical_note"]


def _text_score(term: int, semantic: int, fluency: int, complete: int, errors: list[str]) -> dict:
    return {
        "terminology": term,
        "semantic": semantic,
        "fluency": fluency,
        "completeness": complete,
        "total": term + semantic + fluency + complete,
        "errors": errors,
    }


def _image_score(content: int, medication: int, naturalness: int, errors: list[str]) -> dict:
    return {
        "content": content,
        "medication": medication,
        "naturalness": naturalness,
        "total": content + medication + naturalness,
        "errors": errors,
    }


# These are the same rubric decisions recorded in the five-model deployment
# review, extended with a fresh review of the two new raw outputs.
SCORES = {
    "translategemma-4b": {
        "panic_disorder": _text_score(20, 25, 17, 19, [
            "焦慮症/焦慮發作改變了 panic disorder/panic attacks 的專業意義。",
            "藥物、疾病比 source 的 substance/medical condition 窄；通過與避免的情境較不自然。",
        ]),
        "depression_behavioral_activation": _text_score(23, 27, 16, 19, [
            "行為激活不是首選的臺灣臨床用語；藥物使用窄化 substance use。",
            "審查遇到的障礙在此語境較生硬。",
        ]),
        "ssri_pharmacology": _text_score(28, 28, 18, 20, [
            "危及生命比 potentially serious 更強；興奮不如躁動精確。",
        ]),
        "image_clinical_note": _image_score(28, 27, 23, [
            "焦慮發作改變了 panic frequency；舍曲林 50 毫克每天語序生硬。",
        ]),
    },
    "translategemma-12b": {
        "panic_disorder": _text_score(27, 28, 18, 20, [
            "驚恐症可理解，但臺灣較偏好恐慌症；藥物窄化 source 的 substance。",
        ]),
        "depression_behavioral_activation": _text_score(27, 28, 17, 20, [
            "藥物濫用窄化 substance use；干預措施在臺灣臨床文體較不自然。",
        ]),
        "ssri_pharmacology": _text_score(26, 27, 17, 19, [
            "焦慮不是 agitation 的最佳譯法；反射過度/肌肉抽搐不如 hyperreflexia/clonus 精確。",
            "SSRI 與中文之間的空格格式略差。",
        ]),
        "image_clinical_note": _image_score(37, 28, 25, [
            "加入請、並注意略改變祈使語氣；每週5次與 50毫克 的空格格式不足。",
        ]),
    },
    "translategemma-27b": {
        "panic_disorder": _text_score(29, 29, 19, 20, [
            "藥物窄化 source 的 substance；持續的對再次發作語序略不自然。",
        ]),
        "depression_behavioral_activation": _text_score(25, 27, 17, 19, [
            "行為激活/日程不是首選臺灣用語；藥物濫用窄化 substance use。",
            "惡性循環是 source 未明說的小幅增譯。",
        ]),
        "ssri_pharmacology": _text_score(29, 29, 19, 20, [
            "多種加入了 source 未要求的輕微量化詞。",
        ]),
        "image_clinical_note": _image_score(39, 30, 27, [
            "複診可理解但追蹤更自然；四周應優先寫四週。",
        ]),
    },
    "gemma-4-e4b": {
        "panic_disorder": _text_score(29, 29, 18, 20, [
            "規避的情境不如迴避的情境自然。",
        ]),
        "depression_behavioral_activation": _text_score(28, 29, 19, 19, [
            "減少強化維持低情緒的關係被壓縮，清晰度略降。",
        ]),
        "ssri_pharmacology": _text_score(27, 27, 17, 19, [
            "焦慮不是 agitation 的最佳譯法；肌張性抽搐是 clonus 的近似譯法。",
            "結合了血清素能藥物後語法生硬並壓縮 interaction 子句。",
        ]),
        "image_clinical_note": _image_score(39, 30, 28, [
            "四周較不符合臺灣習慣的四週；自殺念頭不如自殺意念精確。",
        ]),
    },
    "gemma-4-12b": {
        "panic_disorder": _text_score(29, 30, 19, 20, [
            "內在感覺暴露可理解，但內感受暴露是較標準的技術詞。",
        ]),
        "depression_behavioral_activation": _text_score(29, 29, 18, 20, [
            "日程不如日程安排自然；精神病狀不如 psychotic symptoms 精確。",
        ]),
        "ssri_pharmacology": _text_score(29, 29, 18, 20, [
            "斷藥症狀可理解但不如 discontinuation symptoms 標準；肌陣攣接近但不完全等於 clonus。",
        ]),
        "image_clinical_note": _image_score(37, 27, 24, [
            "經過四週後略冗餘；50 毫克 Sertraline 的中文語序與 proper name 呈現較不自然。",
        ]),
    },
    "gemma-4-26b": {
        "panic_disorder": _text_score(29, 30, 19, 20, [
            "心理衛教可接受，但心理教育是更直接的臺灣正式用語。",
        ]),
        "depression_behavioral_activation": _text_score(28, 29, 19, 20, [
            "行為活化療法不如行為啟動常用；社交退縮比 source 的 generic withdrawal 稍窄。",
        ]),
        "ssri_pharmacology": _text_score(27, 29, 18, 20, [
            "回收抑制劑不是臺灣較標準的再攝取抑制劑；其餘藥理症狀與急迫性保留良好。",
        ]),
        "image_clinical_note": _image_score(40, 30, 28, [
            "Sertraline 保留英文 proper name 可接受，但中文藥名會更符合臺灣病歷呈現。",
        ]),
    },
    "gemma-4-31b-qat": {
        "panic_disorder": _text_score(30, 30, 19, 20, [
            "眩暈可接受，但臺灣較常用頭暈；其餘術語與資訊完整。",
        ]),
        "depression_behavioral_activation": _text_score(28, 29, 18, 20, [
            "行為活化/干預較不偏臺灣臨床用語；日程不如日程安排自然。",
        ]),
        "ssri_pharmacology": _text_score(27, 28, 17, 20, [
            "再回收抑制劑不是首選術語；臨床療效與副作用的比較句語法不順；信號宜作訊號。",
        ]),
        "image_clinical_note": _image_score(40, 30, 28, [
            "監控噁心感略不如監測自然；Sertraline 保留英文 proper name 可接受。",
        ]),
    },
}


STATUS = {
    "translategemma-4b": {
        "template_status": "WARN",
        "deployable": "yes",
        "requirements": "custom fixed en→zh-TW adapter; matching mmproj; jinja=true; reasoning=off; ubatch=512; no upstream typed TG fields through OAI normalizer",
        "alias": "translategemma-4b-en-zh-tw",
    },
    "translategemma-12b": {
        "template_status": "WARN",
        "deployable": "yes",
        "requirements": "custom fixed en→zh-TW adapter; matching mmproj; jinja=true; reasoning=off; ubatch=2048; no upstream typed TG fields through OAI normalizer",
        "alias": "translategemma-12b-en-zh-tw",
    },
    "translategemma-27b": {
        "template_status": "FAIL (embedded missing; custom adapter required)",
        "deployable": "yes (custom adapter)",
        "requirements": "custom fixed en→zh-TW adapter; matching mmproj; jinja=true; reasoning=off; ubatch=2048; never use missing embedded template",
        "alias": "translategemma-27b-en-zh-tw",
    },
    "gemma-4-e4b": {
        "template_status": "PASS (parser override)",
        "deployable": "yes",
        "requirements": "embedded template; matching mmproj; reasoning=off; reasoning-format=deepseek; enable_thinking=false; batch=2048; ubatch=4096; ctx=8192; parallel=1; MTP off for fair baseline",
        "alias": "gemma-4-e4b",
    },
    "gemma-4-12b": {
        "template_status": "PASS (parser override)",
        "deployable": "yes",
        "requirements": "embedded template; matching mmproj; reasoning=off; reasoning-format=deepseek; enable_thinking=false; batch=2048; ubatch=4096; ctx=8192; parallel=1; MTP off for fair baseline",
        "alias": "gemma-4-12b",
    },
    "gemma-4-26b": {
        "template_status": "PASS (parser override)",
        "deployable": "yes",
        "requirements": "embedded template; matching mmproj; reasoning=off; reasoning-format=deepseek; enable_thinking=false; batch=2048; ubatch=4096; ctx=8192; parallel=1; MTP off; exact 26B-A4B GGUF/mmproj pair",
        "alias": "gemma-4-26b",
    },
    "gemma-4-31b-qat": {
        "template_status": "PASS (parser override)",
        "deployable": "yes",
        "requirements": "embedded template; matching mmproj; reasoning=off; reasoning-format=deepseek; enable_thinking=false; batch=2048; ubatch=4096; ctx=8192; parallel=1; MTP off; exact 31B QAT GGUF/mmproj pair",
        "alias": "gemma-4-31b-qat",
    },
}


NEW_METADATA = {
    "gemma-4-26b": {
        "model_path": "/benchmark/Gemma4 26b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "model_sha256": "a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891",
        "model_bytes": 14249047104,
        "general_name": "Gemma-4 26B-A4B IT (smart Q4_0, QAT-lossless)",
        "size_label": "128x2.6B",
        "architecture": "gemma4",
        "context": 262144,
        "template_len": 18922,
        "template_sha256": "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b",
        "mmproj_path": "/benchmark/Gemma4 26b/mmproj-F16.gguf",
        "mmproj_sha256": "d00f211a7d4f7fb19bd9b75d8e9342eccffb5920b08fd9562167560fdfcc5dd1",
        "mmproj_bytes": 1193058784,
        "mmproj_name": "Gemma-4-26B-A4B-It",
    },
    "gemma-4-31b-qat": {
        "model_path": "/benchmark/gemma-4-31B-it-qat-UD-Q4_K_XL/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "model_sha256": "00b5a7c497f0c8934033088c10a7fa9a4c015e46ee6d89e9c6890650ba5d0e71",
        "model_bytes": 17287670048,
        "general_name": "Gemma-4 31B IT (smart Q4_0, QAT-lossless)",
        "size_label": "31B",
        "architecture": "gemma4",
        "context": 262144,
        "template_len": 18922,
        "template_sha256": "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b",
        "mmproj_path": "/benchmark/gemma-4-31B-it-qat-UD-Q4_K_XL/mmproj-F16.gguf",
        "mmproj_sha256": "4775e1bf6ef6f8df94caed2d672c74432f0722565e5225a170c72949ebe4cf23",
        "mmproj_bytes": 1198957024,
        "mmproj_name": "Gemma-4-31B-It",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row.get("id") for row in rows] != ALL_CASES:
        raise ValueError(f"unexpected case order in {path}")
    for row in rows:
        if row.get("http_status") != 200:
            raise ValueError(f"non-200 response in {path}: {row.get('http_status')}")
        content = row.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        if any(marker in content for marker in ("<|turn>", "<|channel>", "<start_of_turn>", "<start_of_image>")):
            raise ValueError(f"control marker leaked in response {path} {row['id']}")
    return rows


def response_text(row: dict) -> str:
    return row["response"]["choices"][0]["message"].get("content", "")


def timing(row: dict) -> dict:
    t = row["response"].get("__verbose", {}).get("timings", {})
    return {
        "api_elapsed_s": row.get("elapsed_seconds"),
        "prompt_tokens": t.get("prompt_n"),
        "generation_tokens": t.get("predicted_n"),
        "prompt_tok_s": t.get("prompt_per_second"),
        "generation_tok_s": t.get("predicted_per_second"),
        "prompt_ms": t.get("prompt_ms"),
        "generation_ms": t.get("predicted_ms"),
        "cache_tokens": t.get("cache_n"),
    }


def mean(values: list[float], digits: int = 3) -> float:
    return round(statistics.mean(values), digits)


def md_inline(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def main() -> int:
    inputs = [json.loads(line) for line in (ROOT / "inputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    source_text = {item["id"]: item["english"] for item in inputs}
    image_case = json.loads((ROOT / "image_case.json").read_text(encoding="utf-8"))
    source_text["image_clinical_note"] = image_case["english_exact"]

    raw = {model: read_jsonl(ROOT / filename) for model, filename in RAW_FILES.items()}
    rows_by_model = {model: {row["id"]: row for row in rows} for model, rows in raw.items()}

    # Fixed random order for the seven-model blind pass.  The mapping is
    # printed in the report after the score tables so it remains auditable.
    blind_order = list(RAW_FILES)
    random.Random(20260917).shuffle(blind_order)
    blind_label = {model: f"M{i:02d}" for i, model in enumerate(blind_order, 1)}

    stats = {}
    for model in RAW_FILES:
        text_scores = {case: SCORES[model][case] for case in TEXT_CASES}
        image_score = SCORES[model]["image_clinical_note"]
        text_timings = [timing(rows_by_model[model][case]) for case in TEXT_CASES]
        image_timing = timing(rows_by_model[model]["image_clinical_note"])
        text_average = round(statistics.mean(score["total"] for score in text_scores.values()), 2)
        image_accuracy = image_score["total"]
        combined = round((text_average + image_accuracy) / 2, 2)
        stats[model] = {
            "text": text_scores,
            "image": image_score,
            "text_average": text_average,
            "image_accuracy": image_accuracy,
            "combined": combined,
            "text_timings": text_timings,
            "image_timing": image_timing,
            "text_timing_avg": {
                "api_elapsed_s": mean([x["api_elapsed_s"] for x in text_timings]),
                "prompt_tok_s": mean([x["prompt_tok_s"] for x in text_timings]),
                "generation_tok_s": mean([x["generation_tok_s"] for x in text_timings]),
                "prompt_tokens": mean([x["prompt_tokens"] for x in text_timings]),
                "generation_tokens": mean([x["generation_tokens"] for x in text_timings]),
            },
            "source_raw": str(ROOT / RAW_FILES[model]),
        }

    ranking = sorted(stats, key=lambda model: (-stats[model]["combined"], -stats[model]["text_average"], model))
    ranks = {model: i for i, model in enumerate(ranking, 1)}

    fieldnames = [
        "rank", "model", "case_id", "input_type", "source_english", "translation_zh_tw",
        "terminology_or_content_score", "semantic_or_medication_score", "fluency_score",
        "completeness_score", "total_score", "model_text_average", "model_image_accuracy",
        "model_combined_mean", "api_elapsed_s", "prompt_tok_s", "generation_tok_s",
        "template_status", "deployable", "deployment_requirements", "errors", "source_raw",
    ]
    csv_rows = []
    for model in ranking:
        st = stats[model]
        for case in ALL_CASES:
            score = st["text"][case] if case in TEXT_CASES else st["image"]
            tm = timing(rows_by_model[model][case])
            is_image = case == "image_clinical_note"
            csv_rows.append({
                "rank": ranks[model],
                "model": model,
                "case_id": case,
                "input_type": "image" if is_image else "text",
                "source_english": source_text[case],
                "translation_zh_tw": response_text(rows_by_model[model][case]),
                "terminology_or_content_score": score["content"] if is_image else score["terminology"],
                "semantic_or_medication_score": score["medication"] if is_image else score["semantic"],
                "fluency_score": score["naturalness"] if is_image else score["fluency"],
                "completeness_score": "" if is_image else score["completeness"],
                "total_score": score["total"],
                "model_text_average": st["text_average"],
                "model_image_accuracy": st["image_accuracy"],
                "model_combined_mean": st["combined"],
                "api_elapsed_s": tm["api_elapsed_s"],
                "prompt_tok_s": tm["prompt_tok_s"],
                "generation_tok_s": tm["generation_tok_s"],
                "template_status": STATUS[model]["template_status"],
                "deployable": STATUS[model]["deployable"],
                "deployment_requirements": STATUS[model]["requirements"],
                "errors": "; ".join(score["errors"]),
                "source_raw": st["source_raw"],
            })

    with CSV_OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(csv_rows)

    def req_short(model: str) -> str:
        return STATUS[model]["requirements"]

    report: list[str] = []
    report.append("# 七模型部署有效翻譯評測報告\n")
    report.append("本報告只把正確 OAI/template 路徑產生的七份 raw JSONL 納入品質分數；舊手工 CLI/template 結果不參與排名。輸出為 warm、temperature=0、top_p=1、stream=false、單一 sequential slot、MTP 關閉的隔離 llama.cpp server 測試。\n")
    report.append("## 結論先行\n")
    report.append(f"整體首選是 **{DISPLAY_NAMES[ranking[0]]}**（文字 {stats[ranking[0]]['text_average']:.2f}、圖片翻譯準確率 {stats[ranking[0]]['image_accuracy']}、等權描述性平均 {stats[ranking[0]]['combined']:.2f}）。文字首選是 **{DISPLAY_NAMES[max(stats, key=lambda m: (stats[m]['text_average'], stats[m]['combined']))]}**；圖片首選是 **{DISPLAY_NAMES[max(stats, key=lambda m: (stats[m]['image_accuracy'], stats[m]['text_average']))]}**。新增後首選改為 26B：它在本樣本同時維持 Gemma4 級文字品質與最高圖片分數，速度仍接近 E4B。31B QAT 沒有在品質上超過 26B、E4B 或 12B，且 generation 約 32 tok/s；除非需要相對 TG27 的較高品質/較小 GGUF，否則容量與速度成本不值得。\n")
    report.append("相較舊五模型，整體排名由 E4B（95.34）變為 26B（97.00）；文字首選沒有改變，仍是 Gemma 4 12B（96.67），其次是 26B（96.00）。本報告把文字首選定義為最高 `model_text_average`，故 12B 仍居首；26B 是整體等權首選。\n")

    report.append("## 最終排名\n")
    report.append("`combined_mean` 是文字平均與單張合成圖片翻譯準確率的等權描述性平均，不是 OCR exactness。\n")
    report.append("| rank | model | text average | image translation accuracy | combined mean | template | deployable | text gen tok/s | image gen tok/s |\n|---:|---|---:|---:|---:|---|---|---:|---:|")
    for model in ranking:
        st = stats[model]
        report.append(f"| {ranks[model]} | {DISPLAY_NAMES[model]} | {st['text_average']:.2f} | {st['image_accuracy']} | {st['combined']:.2f} | {STATUS[model]['template_status']} | {STATUS[model]['deployable']} | {st['text_timing_avg']['generation_tok_s']:.3f} | {st['image_timing']['generation_tok_s']:.3f} |")
    report.append("")

    report.append("## 新增模型 GGUF/mmproj 精確驗證\n")
    report.append("檔案以 workspace 的 GGUF header reader 逐一讀取；26B 沒有以 31B 或約 26 GiB 檔案替代。兩個 model GGUF 都是 `general.architecture=gemma4`、`general.quantization_version=2`、`general.file_type=2`（Q4_0/QAT），`tokenizer.chat_template` 長度 18,922 bytes，template SHA 相同；matching mmproj 是 `architecture=clip` 且 basename/finetune 對應尺寸。\n")
    report.append("| model | exact GGUF / size | model SHA-256 | GGUF metadata identity | embedded template | exact matching mmproj / SHA-256 |\n|---|---|---|---|---|---|")
    for model in ("gemma-4-26b", "gemma-4-31b-qat"):
        m = NEW_METADATA[model]
        report.append(f"| {DISPLAY_NAMES[model]} | `{m['model_path']}` ({m['model_bytes']:,} bytes) | `{m['model_sha256']}` | `{m['general_name']}`; size_label `{m['size_label']}`; context `{m['context']}` | present, {m['template_len']} bytes, `{m['template_sha256']}` | `{m['mmproj_path']}` ({m['mmproj_bytes']:,} bytes); `{m['mmproj_sha256']}`; `{m['mmproj_name']}` |")
    report.append("")
    report.append("兩個 embedded template 的 SHA 與既有 Gemma4 12B extraction 相同，並包含 `<|turn>`, `<|channel>`, `<|image|>` 及 `add_generation_prompt` 邏輯；實際 server prompt 顯示每 request 僅一個 user turn、一個 model generation turn，圖片僅一對 image marker。llama.cpp b10964 在啟動時把 Gemma4 的 GGUF `add_bos_token=false` override 成 true，這是 runtime 的 Gemma4 相容行為，不是手工 prompt。\n")
    report.append("官方/一手格式參照：[Google Gemma 4 26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/main)、[Google Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it/tree/main)、[Google Gemma 4 12B chat template](https://huggingface.co/google/gemma-4-12B-it/blob/main/chat_template.jinja)、[Google Gemma tokenizer config](https://huggingface.co/google/gemma-4-12B/blob/main/tokenizer_config.json)、[llama.cpp b10964 Gemma4 template](https://github.com/ggml-org/llama.cpp/blob/b29c606/models/templates/google-gemma-4-31B-it.jinja)。本地 source template 與 embedded byte-for-byte 版本不同，但 control-token/parser 契約一致；本次部署使用 GGUF embedded template，未以外部 generic Gemma template 覆蓋。\n")

    report.append("## 正式 endpoint 核對與實際測試鏈路\n")
    report.append("在測試開始前唯讀呼叫 `http://127.0.0.1:8080/health`、`/props`、`/v1/models`；當時三個 endpoint 都是 connection refused。`launchctl print system/com.scc.llama-server` 顯示 launchd job running，但正式 script 仍指向 `/benchmark/models.ini` 的 router。該 preset 的可觀察條件是 global `reasoning=auto`，26B/31B 為 `ubatch-size=2048` 並帶 MTP；31B 還有 preserve_thinking=true。這與本次公平條件不一致，因此沒有把正式服務的不可連線狀態當成通過，也沒有重啟或修改它。\n")
    report.append("實際使用的 isolated endpoints/aliases：\n\n- `http://127.0.0.1:18086/v1` / `gemma4-26b-audit`\n- `http://127.0.0.1:18087/v1` / `gemma4-31b-qat-audit`\n\n兩端均使用 `--jinja --reasoning off --reasoning-format deepseek --chat-template-kwargs '{\"enable_thinking\":false}' --ctx-size 8192 --parallel 1 --batch-size 2048 --ubatch-size 4096 --no-cont-batching`，且刻意不提供 `--model-draft` 或 `--spec-type` 以關閉 MTP/speculative decoding；兩端均掛載同目錄 matching mmproj。`--chat-template-kwargs` 會產生 b10964 的 deprecation warning，但仍被接受；`reasoning off` 同時確保沒有 thought 內容被輸出。\n")
    report.append("測試命令（server 先載入完成再呼叫）：\n\n```bash\n/opt/homebrew/bin/llama-server --model '<exact GGUF>' --mmproj '<matching mmproj>' --host 127.0.0.1 --port <18086|18087> --alias <audit alias> --jinja --no-warmup --reasoning off --reasoning-format deepseek --chat-template-kwargs '{\"enable_thinking\":false}' --ctx-size 8192 --parallel 1 --batch-size 2048 --ubatch-size 4096 --no-cont-batching --media-path /benchmark/translation_benchmark_20260916_luna --metrics --n-gpu-layers all --flash-attn on\n/benchmark-user/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 translation_benchmark_20260916_luna/run_template_audit.py --base http://127.0.0.1:<port> --model <audit alias> --reasoning-format deepseek --user-instruction 'Translate the English text into Taiwan Traditional Chinese. Output only the translation; do not summarize, explain, or add safety advice.' --out <raw JSONL>\n```\n")

    report.append("## 七模型匿名重評與完整文字結果\n")
    report.append("評分先將七模型以固定 seed `20260917` 隨機排列並以 M-label 盲審，再在報告中揭露 mapping。文字 rubric：術語/領域準確 30、語義忠實 30、臺灣正體流暢 20、完整/格式 20。圖片 rubric：內容保留 40、藥名/劑量/數字 30、正體自然度 30；raw 沒有獨立 OCR channel，故圖片欄稱「圖片翻譯準確率」，不宣稱 OCR exact。\n")
    report.append("| blind label | model | raw source |\n|---|---|---|")
    for model in blind_order:
        report.append(f"| {blind_label[model]} | {DISPLAY_NAMES[model]} | `{stats[model]['source_raw']}` |")
    report.append("")
    report.append("下列每個模型均列出三案完整譯文、四項分數與具體扣分；沒有以舊 summary 取代 raw translation。\n")
    for model in ranking:
        st = stats[model]
        report.append(f"### {DISPLAY_NAMES[model]}（rank {ranks[model]}；文字平均 {st['text_average']:.2f}）\n")
        report.append("| case | 術語/領域 | 語義 | 流暢 | 完整/格式 | total | 扣分/觀察 |\n|---|---:|---:|---:|---:|---:|---|")
        for case in TEXT_CASES:
            s = st["text"][case]
            report.append(f"| {case} | {s['terminology']} | {s['semantic']} | {s['fluency']} | {s['completeness']} | {s['total']} | {md_inline('; '.join(s['errors']))} |")
        report.append("")
        for case in TEXT_CASES:
            report.append(f"**{case} 完整譯文**\n\n```text\n{response_text(rows_by_model[model][case])}\n```\n")

    report.append("## 圖片翻譯結果\n")
    report.append("合成圖片的 renderer source 是：`PSYCHIATRY FOLLOW-UP`、`AFTER FOUR WEEKS, PANIC FREQUENCY FELL FROM 5 EPISODES TO 2 PER WEEK.`、`CONTINUE SERTRALINE 50 MG DAILY; MONITOR NAUSEA AND SUICIDAL IDEATION.`。這是已知的合成輸入文字，不是模型獨立 OCR ground truth；因此只報 translation accuracy。\n")
    report.append("| rank | model | content / 40 | medication-dose-numbers / 30 | zh-TW naturalness / 30 | image translation accuracy | errors |\n|---:|---|---:|---:|---:|---:|---|")
    for model in ranking:
        s = stats[model]["image"]
        report.append(f"| {ranks[model]} | {DISPLAY_NAMES[model]} | {s['content']} | {s['medication']} | {s['naturalness']} | {s['total']} | {md_inline('; '.join(s['errors']))} |")
    report.append("")
    for model in ranking:
        report.append(f"**{DISPLAY_NAMES[model]}（{stats[model]['image']['total']}）**\n\n```text\n{response_text(rows_by_model[model]['image_clinical_note'])}\n```\n")

    report.append("## 速度與 raw evidence\n")
    report.append("所有數字來自各 raw JSONL 的 `elapsed_seconds` 與 `response.__verbose.timings`。model 已在 request 前載入；cold startup/load 未測。每模型四案在一個 sequential slot，prefix/KV cache 會讓 prompt tok/s 受影響，因此 generation tok/s 更可比。\n")
    report.append("| rank | model | text API avg s | image API s | text prompt tok/s | image prompt tok/s | text generation tok/s | image generation tok/s | cache/prompt note |\n|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for model in ranking:
        st = stats[model]
        ta = st["text_timing_avg"]
        im = st["image_timing"]
        report.append(f"| {ranks[model]} | {DISPLAY_NAMES[model]} | {ta['api_elapsed_s']:.3f} | {im['api_elapsed_s']:.3f} | {ta['prompt_tok_s']:.3f} | {im['prompt_tok_s']:.3f} | {ta['generation_tok_s']:.3f} | {im['generation_tok_s']:.3f} | text prompt_n avg {ta['prompt_tokens']:.1f}; image prompt_n {im['prompt_tokens']:.0f}; warm only |")
    report.append("")
    report.append("新模型 raw prompt checks：兩模型三個文字案例都是恰好 1 個 `<|turn>user` + 1 個 `<|turn>model` + 1 個空 thought channel；圖片都是恰好 1 個 `<|image>` 與 1 個 `<image|>`，輸出 content 無 `<|turn>`、`<|channel>`、`<start_of_turn>` 或 `<start_of_image>`。四個 request 各為 HTTP 200、`finish_reason=stop`。\n")

    report.append("## Template 狀態、部署條件與 owner chain\n")
    report.append("| model | template status | deployable | candidate alias | conditions |\n|---|---|---|---|---|")
    for model in ranking:
        report.append(f"| {DISPLAY_NAMES[model]} | {STATUS[model]['template_status']} | {STATUS[model]['deployable']} | `{STATUS[model]['alias']}` | {STATUS[model]['requirements']} |")
    report.append("")
    report.append("Gemma4 26B/31B QAT 的可採用 preset 片段（只放在本報告，沒有寫入正式 `models.ini`）：\n\n```ini\n[gemma-4-26b]\nmodel = /benchmark/Gemma4 26b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf\nmmproj = /benchmark/Gemma4 26b/mmproj-F16.gguf\nchat-template-kwargs = {\"enable_thinking\": false}\nreasoning-format = deepseek\nparallel = 1\nbatch-size = 2048\nubatch-size = 4096\n# no model-draft / no spec-type for the fair baseline\n\n[gemma-4-31b-qat]\nmodel = /benchmark/gemma-4-31B-it-qat-UD-Q4_K_XL/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf\nmmproj = /benchmark/gemma-4-31B-it-qat-UD-Q4_K_XL/mmproj-F16.gguf\nchat-template-kwargs = {\"enable_thinking\": false}\nreasoning-format = deepseek\nparallel = 1\nbatch-size = 2048\nubatch-size = 4096\n# no model-draft / no spec-type for the fair baseline\n```\n")
    report.append("Owner chain：Open WebUI 只送標準 OpenAI `messages` 到 LiteLLM；LiteLLM 以 `openai/<llama-server-id>` 轉發 `/v1/chat/completions`；llama.cpp 是唯一 template owner，負責 embedded/custom Jinja、control tokens、mmproj 與圖片 marker。任何上游都不可預渲染 `<|turn>`、`<|channel>`、`<start_of_turn>` 或 Gemma/TranslateGemma marker。多模態 request 使用 `content=[{\"type\":\"text\",\"text\":...},{\"type\":\"image_url\",\"image_url\":{\"url\":...}}]`；container 間使用 service DNS（例如 LiteLLM→`http://llama-server:8080/v1`），不可把 container 內的 `127.0.0.1` 當 upstream。既有候選鏈路見 [litellm.audit.yaml](litellm.audit.yaml:1) 與 [models.audit.ini](models.audit.ini:1)。\n")

    report.append("## 舊結果作廢與新增模型後判斷\n")
    report.append("舊 `llama-completion` 分數使用手工 `<start_of_turn>`/`<|turn>`/`<|channel>final`，不能證明 server 對 OAI messages 只套一次 template；舊 TranslateGemma 圖片曾強制 generic Gemma template，舊 Gemma4 圖片走 CLI/mtmd 且 12B 曾有 `ubatch=512` assertion 風險。故舊 CLI、generic-template 與非本次 OAI corrected raw 的品質/速度分數均 `VOID_FOR_DEPLOYMENT`，本報告只使用七份列出的 raw。\n")
    report.append("* **整體首選：Gemma 4 26B**。本樣本 combined 97.00，勝過 E4B 95.34、TG27 95.00、31B QAT 96.66。\n* **文字首選：Gemma 4 12B**（96.67），26B 96.00、31B QAT 95.33；新增模型沒有取代文字品質首選。\n* **圖片首選：Gemma 4 26B 與 Gemma 4 31B QAT 並列 98**，均保留 5→2 episodes、Sertraline 50 mg 與自殺意念等關鍵內容。\n* **速度/品質折衷：Gemma 4 26B**。generation 約 100.8 tok/s、text 96、image 98；31B QAT 約 32.0 tok/s，並未帶來相稱品質增益。\n* **31B QAT 是否值得：** 相對 TG27，31B 由 95.00 升至 96.66 且 generation 22.6→約32.0 tok/s、GGUF 17.3 GB 也小於 TG27 約28.7 GB，若需要較高品質且能接受 Gemma4 31B 記憶體仍可考慮；相對 E4B/12B/26B，31B 的品質沒有增益，卻明顯慢於 E4B/26B，通常不值得容量/速度成本。\n")

    report.append("## 限制與檔案\n")
    report.append("- 只有三段心理/精神醫學文字與一張本機合成圖片；不能代表廣泛翻譯能力。\n- 分數是人工 rubric review；圖片沒有獨立 OCR channel，不能宣稱 OCR exact。\n- 速度是 warm server、單 slot、MTP 關閉；未測 cold load、streaming、tool calls 或 LiteLLM/Open WebUI 實際 container hop。\n- b10964 將 `enable_thinking` chat-template kwarg 標為 deprecated；本次同時用 `reasoning=off` 驗證輸出乾淨，正式部署應固定記錄 b10964 版本並在升級後重跑。\n- 正式 `models.ini`、LiteLLM、launchd 與 8080 service 均未修改、未重啟。\n\n正式產物：\n\n- [deployment_translation_results_7models.csv](deployment_translation_results_7models.csv:1)（UTF-8、RFC4180、28 rows = 7×4，含完整譯文）\n- [report_deployment_7models.md](report_deployment_7models.md:1)\n\n原始證據：五份既有 corrected raw 加上 [raw_template_audit_gemma4_26b.jsonl](raw_template_audit_gemma4_26b.jsonl:1) 與 [raw_template_audit_gemma4_31b_qat.jsonl](raw_template_audit_gemma4_31b_qat.jsonl:1)。\n")

    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"wrote {CSV_OUT}")
    print("rows", len(csv_rows), "ranking", [(ranks[m], m, stats[m]["text_average"], stats[m]["image_accuracy"], stats[m]["combined"]) for m in ranking])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
