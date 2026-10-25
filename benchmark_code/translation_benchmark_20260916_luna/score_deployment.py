#!/usr/bin/env python3
"""Score only the corrected llama.cpp OpenAI/template runs.

The scoring notes are keyed by anonymous IDs.  The fixed mapping is written
only to the final JSON after the blinded score matrix is built.  This keeps
the review worksheet independent of model names while retaining a reproducible
mapping for later audit.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_FILES = {
    "translategemma-4b": "raw_template_audit_tg4_v2.jsonl",
    "translategemma-12b": "raw_template_audit_tg12.jsonl",
    "translategemma-27b": "raw_template_audit_tg27.jsonl",
    "gemma-4-e4b": "raw_template_audit_gemma4_e4b.jsonl",
    "gemma-4-12b": "raw_template_audit_gemma4_12b_v2.jsonl",
}

# Generated with Random(20260916).shuffle(list(RAW_FILES)); this order was used
# for the human score pass before the model-name mapping was reopened.
BLIND_ORDER = [
    "translategemma-12b",
    "translategemma-27b",
    "gemma-4-12b",
    "gemma-4-e4b",
    "translategemma-4b",
]
BLIND_LABELS = {name: f"M{i:02d}" for i, name in enumerate(BLIND_ORDER, 1)}

TEXT_SCORES = {
    "M01": {
        "panic_disorder": {
            "terminology": 27,
            "semantic_fidelity": 28,
            "traditional_fluency": 18,
            "completeness_format": 20,
            "errors": [
                "驚恐症 is understandable but 恐慌症 is the preferred Taiwan term.",
                "藥物 narrows substance, while the source is intentionally broader.",
            ],
        },
        "depression_behavioral_activation": {
            "terminology": 27,
            "semantic_fidelity": 28,
            "traditional_fluency": 17,
            "completeness_format": 20,
            "errors": [
                "藥物濫用 narrows substance use.",
                "干預措施 is understandable but less natural in Taiwan clinical prose.",
            ],
        },
        "ssri_pharmacology": {
            "terminology": 26,
            "semantic_fidelity": 27,
            "traditional_fluency": 17,
            "completeness_format": 19,
            "errors": [
                "焦慮 is not the best rendering of agitation.",
                "反射過度 and 肌肉抽搐 are less specific than hyperreflexia and clonus.",
                "Missing spaces around SSRI/Chinese text slightly hurts format.",
            ],
        },
    },
    "M02": {
        "panic_disorder": {
            "terminology": 29,
            "semantic_fidelity": 29,
            "traditional_fluency": 19,
            "completeness_format": 20,
            "errors": [
                "藥物 is narrower than the source's substance.",
                "持續的對再次發作 is slightly awkward word order.",
            ],
        },
        "depression_behavioral_activation": {
            "terminology": 25,
            "semantic_fidelity": 27,
            "traditional_fluency": 17,
            "completeness_format": 19,
            "errors": [
                "行為激活 and 日程 are less preferred Taiwan clinical wording than 行為啟動 and 日程安排.",
                "藥物濫用 narrows substance use.",
                "惡性循環 is a small addition to the source's neutral cycle wording.",
            ],
        },
        "ssri_pharmacology": {
            "terminology": 29,
            "semantic_fidelity": 29,
            "traditional_fluency": 19,
            "completeness_format": 20,
            "errors": [
                "多種 adds a mild quantifier not required by interacting serotonergic drugs.",
            ],
        },
    },
    "M03": {
        "panic_disorder": {
            "terminology": 29,
            "semantic_fidelity": 30,
            "traditional_fluency": 19,
            "completeness_format": 20,
            "errors": [
                "內在感覺暴露 is understandable but 內感受暴露 is the more standard technical term.",
            ],
        },
        "depression_behavioral_activation": {
            "terminology": 29,
            "semantic_fidelity": 29,
            "traditional_fluency": 18,
            "completeness_format": 20,
            "errors": [
                "日程 is a regional lexical choice; 日程安排 is more natural in Taiwan.",
                "精神病狀 is less precise than psychotic symptoms.",
            ],
        },
        "ssri_pharmacology": {
            "terminology": 29,
            "semantic_fidelity": 29,
            "traditional_fluency": 18,
            "completeness_format": 20,
            "errors": [
                "斷藥症狀 is understandable but less standard than discontinuation symptoms.",
                "肌陣攣 is close to, but not exactly, the source's clonus.",
            ],
        },
    },
    "M04": {
        "panic_disorder": {
            "terminology": 29,
            "semantic_fidelity": 29,
            "traditional_fluency": 18,
            "completeness_format": 20,
            "errors": [
                "規避的情境 is less idiomatic than 迴避的情境.",
            ],
        },
        "depression_behavioral_activation": {
            "terminology": 28,
            "semantic_fidelity": 29,
            "traditional_fluency": 19,
            "completeness_format": 19,
            "errors": [
                "減少強化維持低情緒 is compressed and slightly less clear than the source relationship.",
            ],
        },
        "ssri_pharmacology": {
            "terminology": 27,
            "semantic_fidelity": 27,
            "traditional_fluency": 17,
            "completeness_format": 19,
            "errors": [
                "焦慮 is not the best rendering of agitation.",
                "肌張性抽搐 is an approximate rendering of clonus.",
                "結合了血清素能藥物後 is grammatically awkward and compresses the interaction clause.",
            ],
        },
    },
    "M05": {
        "panic_disorder": {
            "terminology": 20,
            "semantic_fidelity": 25,
            "traditional_fluency": 17,
            "completeness_format": 19,
            "errors": [
                "焦慮症 and 焦慮發作 change panic disorder and panic attacks to anxiety terminology.",
                "藥物、疾病 are narrower than substance and medical condition.",
                "通過 and 避免的情境 are less natural Taiwan wording.",
            ],
        },
        "depression_behavioral_activation": {
            "terminology": 23,
            "semantic_fidelity": 27,
            "traditional_fluency": 16,
            "completeness_format": 19,
            "errors": [
                "行為激活 is a regional lexical choice rather than preferred Taiwan 行為啟動.",
                "藥物使用 narrows substance use.",
                "審查遇到的障礙 is awkward in this clinical context.",
            ],
        },
        "ssri_pharmacology": {
            "terminology": 28,
            "semantic_fidelity": 28,
            "traditional_fluency": 18,
            "completeness_format": 20,
            "errors": [
                "危及生命 is stronger than potentially serious.",
                "興奮 is acceptable but less precise than agitation in this context.",
            ],
        },
    },
}

# The new image raw responses contain translation only, not a separately
# transcribed OCR channel.  Therefore these are not OCR exactness scores.
IMAGE_SCORES = {
    "M01": {
        "content_retention_40": 37,
        "medication_dose_numbers_30": 28,
        "traditional_naturalness_30": 25,
        "errors": [
            "Content is retained; added 請 and 並注意 slightly changes the imperative.",
            "Missing spaces in 每週5次 and 50毫克 reduce presentation quality.",
        ],
    },
    "M02": {
        "content_retention_40": 39,
        "medication_dose_numbers_30": 30,
        "traditional_naturalness_30": 27,
        "errors": [
            "複診 is a close equivalent of follow-up, but 追蹤 is more natural here.",
            "四周 should preferably be 四週 in Taiwan usage.",
        ],
    },
    "M03": {
        "content_retention_40": 37,
        "medication_dose_numbers_30": 27,
        "traditional_naturalness_30": 24,
        "errors": [
            "經過四週後 is redundant.",
            "50 毫克 Sertraline reverses the natural Chinese medication order and leaves the English name capitalized.",
        ],
    },
    "M04": {
        "content_retention_40": 39,
        "medication_dose_numbers_30": 30,
        "traditional_naturalness_30": 28,
        "errors": [
            "四周 is less preferred than 四週.",
            "自殺念頭 is usable but less exact than 自殺意念.",
        ],
    },
    "M05": {
        "content_retention_40": 28,
        "medication_dose_numbers_30": 27,
        "traditional_naturalness_30": 23,
        "errors": [
            "焦慮發作 changes panic frequency to anxiety frequency.",
            "舍曲林 50 毫克每天 has awkward word order, though the medication and dose are retained.",
        ],
    },
}


def read_raw(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["id"] for row in rows] != ["panic_disorder", "depression_behavioral_activation", "ssri_pharmacology", "image_clinical_note"]:
        raise ValueError(f"unexpected case order in {path}")
    if any(row["http_status"] != 200 for row in rows):
        raise ValueError(f"non-200 response in {path}")
    return rows


def score_total(item: dict) -> int:
    return sum(item[key] for key in ("terminology", "semantic_fidelity", "traditional_fluency", "completeness_format"))


def image_total(item: dict) -> int:
    return sum(item[key] for key in ("content_retention_40", "medication_dose_numbers_30", "traditional_naturalness_30"))


def avg(values: list[float]) -> float:
    return round(statistics.mean(values), 3)


def main() -> int:
    raw = {model: read_raw(ROOT / filename) for model, filename in RAW_FILES.items()}
    audit = json.loads((ROOT / "template_audit.json").read_text(encoding="utf-8"))
    status_by_model = {item["model"]: item for item in audit["deployment_status"]}

    # Confirm the documented shuffle is reproducible.
    shuffled = list(RAW_FILES)
    random.Random(20260916).shuffle(shuffled)
    if shuffled != BLIND_ORDER:
        raise AssertionError((shuffled, BLIND_ORDER))

    blinded = {}
    model_results = {}
    csv_rows = []
    for model in BLIND_ORDER:
        label = BLIND_LABELS[model]
        rows = raw[model]
        text_rows = [row for row in rows if row["id"] != "image_clinical_note"]
        image_row = next(row for row in rows if row["id"] == "image_clinical_note")
        cases = {}
        for case_id in ("panic_disorder", "depression_behavioral_activation", "ssri_pharmacology"):
            score = dict(TEXT_SCORES[label][case_id])
            score["total"] = score_total(score)
            score["output"] = next(row for row in text_rows if row["id"] == case_id)["response"]["choices"][0]["message"]["content"]
            cases[case_id] = score
        image_score = dict(IMAGE_SCORES[label])
        image_score["total"] = image_total(image_score)
        image_score["output"] = image_row["response"]["choices"][0]["message"]["content"]

        timing_rows = []
        for row in rows:
            verbose = row["response"].get("__verbose", {})
            timing = verbose.get("timings", {})
            timing_rows.append({
                "id": row["id"],
                "api_elapsed_seconds": row["elapsed_seconds"],
                "prompt_tokens": timing.get("prompt_n"),
                "completion_tokens": timing.get("predicted_n"),
                "prompt_tok_s": timing.get("prompt_per_second"),
                "generation_tok_s": timing.get("predicted_per_second"),
                "prompt_ms": timing.get("prompt_ms"),
                "generation_ms": timing.get("predicted_ms"),
                "cache_tokens": verbose.get("tokens_cached", timing.get("cache_n")),
            })
        text_timing = [row for row in timing_rows if row["id"] != "image_clinical_note"]
        image_timing = next(row for row in timing_rows if row["id"] == "image_clinical_note")
        text_average = round(statistics.mean(cases[key]["total"] for key in cases), 2)
        image_average = image_score["total"]
        combined = round((text_average + image_average) / 2, 2)
        blinded[label] = {
            "text": cases,
            "image": image_score,
            "text_average": text_average,
            "image_translation_accuracy": image_average,
            "combined_mean_for_ranking_only": combined,
            "timings": {
                "text_cases": text_timing,
                "image_case": image_timing,
                "text_average": {
                    "api_elapsed_seconds": avg([row["api_elapsed_seconds"] for row in text_timing]),
                    "prompt_tok_s": avg([row["prompt_tok_s"] for row in text_timing]),
                    "generation_tok_s": avg([row["generation_tok_s"] for row in text_timing]),
                    "prompt_tokens": avg([row["prompt_tokens"] for row in text_timing]),
                    "completion_tokens": avg([row["completion_tokens"] for row in text_timing]),
                },
                "image": {
                    "api_elapsed_seconds": round(image_timing["api_elapsed_seconds"], 3),
                    "prompt_tok_s": round(image_timing["prompt_tok_s"], 3),
                    "generation_tok_s": round(image_timing["generation_tok_s"], 3),
                    "prompt_tokens": round(image_timing["prompt_tokens"], 3),
                    "completion_tokens": round(image_timing["completion_tokens"], 3),
                },
            },
            "raw_file": str(ROOT / RAW_FILES[model]),
        }
        status = status_by_model[model]
        model_results[model] = {
            "blind_id": label,
            "template_status": status["template_status"],
            "candidate_deployable": status["candidate_deployable"],
            "requires_custom_template": status["requires_custom_template"],
            "recommended_alias": status["recommended_alias"],
            "text_average": text_average,
            "image_translation_accuracy": image_average,
            "combined_mean_for_ranking_only": combined,
            "source_raw": str(ROOT / RAW_FILES[model]),
        }

    ranking = sorted(model_results, key=lambda model: (-model_results[model]["combined_mean_for_ranking_only"], model))
    for rank, model in enumerate(ranking, 1):
        model_results[model]["overall_rank_equal_weight_descriptive"] = rank

    output = {
        "evaluation": {
            "date": "2026-09-16",
            "source_policy": "Only the five corrected OpenAI/template raw JSONL files listed below; no historical scores were used for scoring or ranking.",
            "review_method": "Outputs were anonymized under a fixed random order before manual review. Scores were then mapped back using the recorded blind mapping.",
            "blind_seed": 20260916,
            "text_rubric": {
                "terminology_domain_accuracy": 30,
                "semantic_fidelity": 30,
                "traditional_chinese_fluency": 20,
                "completeness_and_format": 20,
                "total": 100,
            },
            "image_rubric": {
                "name": "image_translation_accuracy",
                "reason": "The corrected raw responses contain one image translation and no independent OCR transcription channel; therefore no OCR exactness score is claimed.",
                "content_retention": 40,
                "medication_dose_numbers": 30,
                "traditional_chinese_naturalness": 30,
                "total": 100,
            },
            "ranking_basis": "The descriptive combined mean gives text_average and image_translation_accuracy equal weight. It is not a replacement for the text rubric or an OCR score; image has one synthetic case.",
            "timing_regime": {
                "request_state": "warm_model_loaded",
                "cold_startup_load_measured": False,
                "sequential_slots": 1,
                "cache_note": "Each model's four requests ran in one sequential slot; prefix/KV cache state affects prompt timing. Generation tok/s is more comparable across cases than prompt tok/s.",
            },
        },
        "blind_review": {
            "label_order": [BLIND_LABELS[model] for model in BLIND_ORDER],
            "mapping": {
                BLIND_LABELS[model]: {
                    "model": model,
                    "raw_file": str(ROOT / RAW_FILES[model]),
                }
                for model in BLIND_ORDER
            },
            "blinded_score_matrix": blinded,
        },
        "models": model_results,
        "final_ranking": [
            {
                "rank": model_results[model]["overall_rank_equal_weight_descriptive"],
                "model": model,
                "text_average": model_results[model]["text_average"],
                "image_translation_accuracy": model_results[model]["image_translation_accuracy"],
                "combined_mean_for_ranking_only": model_results[model]["combined_mean_for_ranking_only"],
                "template_status": model_results[model]["template_status"],
            }
            for model in ranking
        ],
        "limitations": [
            "Three text cases and one synthetic image are a small quality sample, not a general benchmark.",
            "All raw requests used temperature=0, top_p=1, stream=false, one sequential slot. These are warm-request figures: servers were ready and models loaded before requests; cold startup/load time was not measured.",
            "Within each file, prefix/KV cache state and prior loaded context affect prompt timing; generation tok/s is more comparable than prompt tok/s.",
            "Image scores assess translation content, medication/dose/numbers, and zh-TW naturalness. They do not claim OCR exactness.",
            "The Gemma4 runs include an explicit translation instruction because Gemma4 is a general IT model; TranslateGemma uses the fixed en->zh-TW adapter instruction.",
        ],
    }
    (ROOT / "deployment_scores.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "rank", "model", "blind_id", "template_status", "candidate_deployable",
        "requires_custom_template", "text_average", "image_translation_accuracy",
        "combined_mean_for_ranking_only", "panic_disorder", "depression_behavioral_activation",
        "ssri_pharmacology", "text_api_elapsed_avg_s", "image_api_elapsed_s",
        "text_prompt_tok_s_avg", "image_prompt_tok_s", "text_generation_tok_s_avg",
        "image_generation_tok_s", "text_prompt_tokens_avg", "image_prompt_tokens",
        "source_raw",
    ]
    for model in ranking:
        label = BLIND_LABELS[model]
        b = blinded[label]
        row = {
            "rank": model_results[model]["overall_rank_equal_weight_descriptive"],
            "model": model,
            "blind_id": label,
            "template_status": model_results[model]["template_status"],
            "candidate_deployable": model_results[model]["candidate_deployable"],
            "requires_custom_template": model_results[model]["requires_custom_template"],
            "text_average": model_results[model]["text_average"],
            "image_translation_accuracy": model_results[model]["image_translation_accuracy"],
            "combined_mean_for_ranking_only": model_results[model]["combined_mean_for_ranking_only"],
            "panic_disorder": b["text"]["panic_disorder"]["total"],
            "depression_behavioral_activation": b["text"]["depression_behavioral_activation"]["total"],
            "ssri_pharmacology": b["text"]["ssri_pharmacology"]["total"],
            "text_api_elapsed_avg_s": b["timings"]["text_average"]["api_elapsed_seconds"],
            "image_api_elapsed_s": b["timings"]["image"]["api_elapsed_seconds"],
            "text_prompt_tok_s_avg": b["timings"]["text_average"]["prompt_tok_s"],
            "image_prompt_tok_s": b["timings"]["image"]["prompt_tok_s"],
            "text_generation_tok_s_avg": b["timings"]["text_average"]["generation_tok_s"],
            "image_generation_tok_s": b["timings"]["image"]["generation_tok_s"],
            "text_prompt_tokens_avg": b["timings"]["text_average"]["prompt_tokens"],
            "image_prompt_tokens": b["timings"]["image"]["prompt_tokens"],
            "source_raw": b["raw_file"],
        }
        csv_rows.append(row)
    with (ROOT / "deployment_results_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"wrote {ROOT / 'deployment_scores.json'}")
    print(f"wrote {ROOT / 'deployment_results_summary.csv'}")
    for row in csv_rows:
        print(row["rank"], row["model"], row["text_average"], row["image_translation_accuracy"], row["combined_mean_for_ranking_only"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
