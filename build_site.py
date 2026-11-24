#!/usr/bin/env python3
"""Build the privacy-safe Traditional Chinese static benchmark site.

Only Python's standard library is required.  The input CSVs are read from the
committed public ``docs/data`` snapshot by default and an allow-list of public
fields is emitted.  Raw
attempts, absolute paths, logs and model/server bindings are never copied.

Typical use from this repository::

    python build_site.py

Inputs can be overridden explicitly when a future frozen review is published::

    python build_site.py --results <external-results> \
        --materials <external-materials>
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


MODEL_ORDER = [
    "gemma4-12b",
    "gemma4-26b",
    "gemma4-31b-qat",
    "gemma4-e4b",
    "translategemma-12b",
    "translategemma-27b",
    "translategemma-4b",
]

MODEL_SHORT = {
    "gemma4-12b": "Gemma 4 12B",
    "gemma4-26b": "Gemma 4 26B",
    "gemma4-31b-qat": "Gemma 4 31B QAT",
    "gemma4-e4b": "Gemma 4 E4B",
    "translategemma-12b": "TranslateGemma 12B",
    "translategemma-27b": "TranslateGemma 27B",
    "translategemma-4b": "TranslateGemma 4B",
}

JUDGE_LABELS = {
    "judge-01": "Qwen（本機）",
    "judge-02": "Nemotron（本機）",
    "judge-03": "Mistral（本機）",
    "judge-04": "GPT（雲端）",
    "judge-05": "Gemini（雲端）",
}

PUBLIC_SITE_BASE = "https://gregorytsai.github.io/psych-translation-benchmark/"

CRITERION_DIMENSIONS = (
    ("terminology_locale", "術語與台灣用語", "Terminology & Taiwan usage", 30),
    ("semantic_safety", "語意與安全", "Meaning & safety", 30),
    ("fluency", "流暢度", "Fluency", 20),
    ("completeness_format", "完整性與格式", "Completeness & format", 20),
)
CRITERION_KEYS = tuple(item[0] for item in CRITERION_DIMENSIONS)


def esc(value: Any) -> str:
    """HTML-escape every value that enters markup."""

    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any, default: int = 0) -> int:
    number = as_float(value)
    return default if number is None else int(number)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def display_number(value: Any, digits: int = 2, missing: str = "—") -> str:
    number = as_float(value)
    if number is None:
        return missing
    rendered = f"{number:.{digits}f}"
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def display_integer(value: Any, missing: str = "—") -> str:
    number = as_float(value)
    return missing if number is None else f"{int(number):,}"


def material_number(material_id: str) -> int:
    match = re.search(r"(\d+)$", material_id or "")
    return int(match.group(1)) if match else 10_000


def safe_filename(value: Any) -> str:
    """Return only a filename, never a source path."""

    text = str(value or "").replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def safe_relative_material_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").lstrip("/")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if ".." in parts:
        return ""
    return "/".join(parts)


def model_index(model_id: str) -> int:
    try:
        return MODEL_ORDER.index(model_id)
    except ValueError:
        return len(MODEL_ORDER)


def bool_json(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    return as_bool(value)


def public_translation_row(row: dict[str, str]) -> dict[str, Any]:
    """Project one source row to the explicit public allow-list.

    In particular, no ``*_raw_path`` or other provenance path field is ever
    copied.  Public source text is retained because this site is an auditable
    translation benchmark; image aliases and hashes are intentionally omitted.
    """

    scores: list[float | None] = []
    valid: list[bool | None] = []
    for judge in ("judge-01", "judge-02", "judge-03", "judge-04", "judge-05"):
        scores.append(as_float(row.get(f"{judge}_score")))
        valid.append(bool_json(row.get(f"{judge}_valid")))
    item: dict[str, Any] = {
        "material_id": row.get("material_id", ""),
        "source_id": row.get("source_id", ""),
        "source_kind": row.get("source_kind", ""),
        "source_en": row.get("source_en", "") if row.get("source_kind") == "text" else "",
        "target_zh_tw": row.get("target_zh_tw", ""),
        "image_description_en": row.get("image_ground_truth_en", "")
        if row.get("source_kind") == "image"
        else "",
        "image_role": row.get("image_role", "") if row.get("source_kind") == "image" else "",
        "image_layout": row.get("image_layout_stratum", "") if row.get("source_kind") == "image" else "",
        "model_id": row.get("model_id", ""),
        "model_name": row.get("model_name", ""),
        "translation_status": row.get("translation_status", ""),
        "translation_elapsed_seconds": as_float(row.get("translation_elapsed_seconds")),
        "panel_mean": as_float(row.get("panel_mean")),
        "panel_median": as_float(row.get("panel_median")),
        "panel_n_valid": as_int(row.get("panel_n_valid")),
        "panel_required_k": as_int(row.get("panel_required_k")),
        "common_case_eligible": as_bool(row.get("common_case_eligible")),
        "common_case_status": row.get("common_case_status", ""),
        "untranslated_visible_english_any_judge": as_bool(
            row.get("untranslated_visible_english_any_judge")
        ),
        "judge_scores": scores,
        "judge_valid": valid,
    }
    return item


def normalize_curated_examples(raw: Any) -> dict[str, list[dict[str, Any]]]:
    """Read only safe, editorial example fields from an optional JSON file.

    The publisher may provide either ``{"examples": [...]}``, a model-keyed
    mapping, or a plain list.  Paths, raw-attempt identifiers, and unknown
    fields are deliberately ignored; this keeps the model pages useful when
    the larger analysis review remains private.
    """

    candidates: list[dict[str, Any]] = []
    if isinstance(raw, list):
        candidates = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        if isinstance(raw.get("examples"), list):
            candidates = [x for x in raw["examples"] if isinstance(x, dict)]
        else:
            for model_id, value in raw.items():
                if isinstance(value, list):
                    candidates.extend({"model_id": model_id, **x} for x in value if isinstance(x, dict))
                elif isinstance(value, dict):
                    if any(key in value for key in ("better", "risk", "examples")):
                        nested = value.get("examples", []) if isinstance(value.get("examples"), list) else []
                        for kind in ("better", "risk"):
                            entry = value.get(kind)
                            if isinstance(entry, dict):
                                nested.append({"kind": kind, **entry})
                            elif isinstance(entry, list):
                                nested.extend({"kind": kind, **x} for x in entry if isinstance(x, dict))
                        candidates.extend({"model_id": model_id, **x} for x in nested if isinstance(x, dict))
                    else:
                        candidates.append({"model_id": model_id, **value})

    out: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        model_id = str(item.get("model_id") or item.get("model") or "").strip()
        if not model_id:
            continue
        source = item.get("source_excerpt") or item.get("source_en") or item.get("source") or ""
        target = item.get("translation_excerpt") or item.get("target_zh_tw") or item.get("translation") or item.get("target") or ""
        if not source and not target:
            continue
        why_zh = item.get("why_zh") or item.get("note_zh") or "評審提出的疑義，需人工複核"
        why_en = item.get("why_en") or item.get("note_en") or "A reviewer raised a question that needs human review."
        normalized = {
            "model_id": model_id,
            "material_id": str(item.get("material_id") or item.get("case_id") or "") or "curated-example",
            "kind": str(item.get("kind") or item.get("label") or "example"),
            "source_en": str(source),
            "target_zh_tw": str(target),
            "panel_mean": as_float(item.get("panel_mean") or item.get("score") or item.get("panel_score")),
            "why_zh": str(why_zh),
            "why_en": str(why_en),
        }
        out.setdefault(model_id, []).append(normalized)
    return out


def load_runtime_inventory(results_dir: Path) -> dict[str, Any]:
    """Load public model-file facts without exposing source paths."""

    inventory: dict[str, Any] = {}
    server_version = ""
    config: dict[str, Any] = {}
    inventory_path = results_dir / "model_inventory.json"
    config_path = results_dir / "translation_config.json"
    if inventory_path.is_file():
        loaded = read_json(inventory_path)
        if isinstance(loaded, dict) and isinstance(loaded.get("models"), dict):
            inventory = loaded["models"]
            server_version = str(loaded.get("server_version") or "")
    if config_path.is_file():
        loaded = read_json(config_path)
        if isinstance(loaded, dict):
            config = loaded
    config_models = {
        str(item.get("id")): item
        for item in config.get("models", [])
        if isinstance(item, dict) and item.get("id")
    }
    by_model: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        item = inventory.get(model_id, {}) if isinstance(inventory, dict) else {}
        cfg = config_models.get(model_id, {})
        model_bytes = as_float(item.get("model_size_bytes"))
        mmproj_bytes = as_float(item.get("mmproj_size_bytes"))
        by_model[model_id] = {
            "model_filename": safe_filename(item.get("model_path") or cfg.get("model")),
            "mmproj_filename": safe_filename(item.get("mmproj_path") or cfg.get("mmproj")),
            "model_size_gib": model_bytes / (1024**3) if model_bytes is not None else None,
            "mmproj_size_gib": mmproj_bytes / (1024**3) if mmproj_bytes is not None else None,
            "total_size_gib": (model_bytes + mmproj_bytes) / (1024**3) if model_bytes is not None and mmproj_bytes is not None else None,
            "model_sha256": str(item.get("model_sha256") or ""),
            "mmproj_sha256": str(item.get("mmproj_sha256") or ""),
            "template_mode": str(cfg.get("template_mode") or ""),
            "alias": str(cfg.get("alias") or ""),
        }
    fixed = config.get("fixed_generation", {}) if isinstance(config.get("fixed_generation"), dict) else {}
    fixed_public = {key: fixed.get(key) for key in ("ctx_size", "max_tokens", "temperature", "top_p", "seed", "reasoning", "speculative")}
    build_match = re.search(r"\bbuild\s+(\d+)\b", server_version, flags=re.IGNORECASE)
    platform = "Darwin arm64" if "Darwin arm64" in server_version else ""
    return {
        "by_model": by_model,
        "fixed_generation": fixed_public,
        "server_build": f"llama.cpp build {build_match.group(1)}" if build_match else "",
        "platform": platform,
    }


def load_image_sources(materials_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the four public image-source records once for source panels."""

    fallback = {
        "img-01": {"relative_path": "images/img01_synthetic_depression_note.png", "kind": "synthetic", "label": "Synthetic evaluation image — not a clinical record", "pdf_url": "", "license_note": "Project-authored synthetic render; not a patient record."},
        "img-04": {"relative_path": "images/img04_synthetic_medication_label.png", "kind": "synthetic", "label": "Synthetic evaluation image — not a prescription", "pdf_url": "", "license_note": "Project-authored synthetic render; not a prescription."},
        "img-09": {"relative_path": "", "sha256": "7c51577226a100e72c96c4d649d3b53e7e2e9bfd2457899750d556b2288d2371", "kind": "who", "label": "WHO PDF page-1 render; reference text is not human gold", "pdf_url": "https://cdn.who.int/media/docs/default-source/mental-health/9789240078796-eng.pdf?download=true", "license_note": "WHO source: CC BY-NC-SA 3.0 IGO; attribution, non-commercial use and ShareAlike apply."},
        "img-10": {"relative_path": "", "sha256": "f21d424d79b8d9ffa7ea5b250e40bd0b81a040d4443301bfd9d595089e54e742", "kind": "who", "label": "WHO PDF page-1 render; reference text is not human gold", "pdf_url": "https://cdn.who.int/media/docs/default-source/searo/mhs/depression-what-you-should-know-what-you-can-do.pdf?sfvrsn=293cb15b_5", "license_note": "WHO source: CC BY-NC-SA 3.0 IGO; attribution, non-commercial use and ShareAlike apply."},
    }
    path = materials_dir / "image_selection.json"
    if not path.is_file():
        return fallback
    try:
        loaded = read_json(path)
    except (OSError, json.JSONDecodeError):
        return fallback
    selected = loaded.get("selected_images", []) if isinstance(loaded, dict) else []
    out: dict[str, dict[str, Any]] = {}
    for item in selected:
        if not isinstance(item, dict) or not item.get("image_id"):
            continue
        image_id = str(item["image_id"])
        relative_path = safe_relative_material_path(item.get("source_relative_path"))
        source_pdf_url = str(item.get("source_pdf_url") or "")
        kind = "who" if source_pdf_url else "synthetic"
        out[image_id] = {
            "relative_path": relative_path if kind == "synthetic" else "",
            "sha256": str(item.get("sha256") or ""),
            "kind": kind,
            "label": str(item.get("synthetic_label") or item.get("ground_truth_status") or "Image source record"),
            "pdf_url": source_pdf_url,
            "license_note": "WHO source: CC BY-NC-SA 3.0 IGO; attribution, non-commercial use and ShareAlike apply." if kind == "who" else "Project-authored synthetic render; not a patient record or prescription.",
        }
    return {**fallback, **out}


def _criterion_value(value: float, maximum: int) -> dict[str, float | int]:
    raw_mean = round(float(value), 6)
    return {
        "raw_mean": raw_mean,
        "normalized": round(raw_mean / maximum * 100.0, 6),
        "max_raw": maximum,
    }


def calculate_criterion_scores(results_dir: Path, models: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate safe numeric criterion summaries from the public long CSV.

    Each complete item is first averaged across its valid reviewer cells, then
    item means are averaged within model and track.  Text excludes text-15;
    image excludes the deliberately unsupported Mistral image cells.  The
    returned object contains no source text, paths, hashes, prompts or logs.
    """

    path = results_dir / "pointwise_results_long.csv"
    if not path.is_file():
        return {
            "schema_version": "criterion-scores-v1",
            "available": False,
            "reason": "pointwise_results_long.csv is not present in the selected public snapshot",
            "models": {},
        }

    rows = read_csv(path)
    expected_judges = {"text": 5, "image": 4}
    item_rows: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    observed_items: dict[str, set[str]] = {"text": set(), "image": set()}
    invalid_sum_rows: list[tuple[str, str, str]] = []

    for row in rows:
        material_id = str(row.get("material_id") or "")
        if material_id.startswith("text-"):
            track = "text"
        elif material_id.startswith("img-"):
            track = "image"
        else:
            continue
        if material_id == "text-15":
            continue
        if track == "image" and row.get("judge_slot") == "judge-03":
            continue
        if row.get("status") != "ok" or str(row.get("scalar_valid") or "").lower() != "true":
            continue
        model_id = str(row.get("model_id") or "")
        if model_id not in MODEL_ORDER:
            continue
        values: dict[str, float] = {}
        for key in CRITERION_KEYS:
            value = as_float(row.get(key))
            if value is None:
                raise ValueError(f"Missing public criterion value: {material_id}/{model_id}/{row.get('judge_slot')}/{key}")
            values[key] = value
        total = as_float(row.get("score_0_100"))
        if total is None or abs(sum(values.values()) - total) > 1e-9:
            invalid_sum_rows.append((material_id, model_id, str(row.get("judge_slot") or "")))
        values["total"] = total if total is not None else sum(values.values())
        item_rows.setdefault((track, material_id, model_id), []).append(values)
        observed_items[track].add(material_id)

    expected_items = {
        "text": sorted(observed_items["text"], key=material_number),
        "image": sorted(observed_items["image"], key=material_number),
    }
    if len(expected_items["text"]) != 29 or len(expected_items["image"]) != 4:
        raise ValueError(
            "Unexpected complete criterion set: "
            f"text={len(expected_items['text'])}, image={len(expected_items['image'])}"
        )
    if invalid_sum_rows:
        raise ValueError(
            "Criterion subscores do not sum to score_0_100: "
            + ", ".join("/".join(item) for item in invalid_sum_rows[:5])
        )

    models_by_id = {str(model["model_id"]): model for model in models}
    result_models: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        model_result: dict[str, Any] = {}
        for track in ("text", "image"):
            item_means: dict[str, list[float]] = {key: [] for key in (*CRITERION_KEYS, "total")}
            for material_id in expected_items[track]:
                cells = item_rows.get((track, material_id, model_id), [])
                if len(cells) != expected_judges[track]:
                    raise ValueError(
                        f"Incomplete criterion panel: {track}/{material_id}/{model_id} "
                        f"has {len(cells)} cells, expected {expected_judges[track]}"
                    )
                for key in item_means:
                    item_means[key].append(statistics.fmean(cell[key] for cell in cells))
            dimensions = {
                key: _criterion_value(statistics.fmean(item_means[key]), maximum)
                for key, _zh, _en, maximum in CRITERION_DIMENSIONS
            }
            total_raw = statistics.fmean(item_means["total"])
            expected_total = as_float(models_by_id.get(model_id, {}).get(f"{track}_score"))
            if expected_total is not None and abs(total_raw - expected_total) > 0.0011:
                raise ValueError(
                    f"Criterion total differs from published {track} mean for {model_id}: "
                    f"{total_raw:.6f} vs {expected_total:.6f}"
                )
            model_result[track] = {
                "eligible_items": len(expected_items[track]),
                "judge_count": expected_judges[track],
                "total": _criterion_value(total_raw, 100),
                "dimensions": dimensions,
            }
        result_models[model_id] = model_result

    return {
        "schema_version": "criterion-scores-v1",
        "available": True,
        "scale": {
            "normalized_min": 0,
            "normalized_max": 100,
            "note": "Criterion means are converted from their 30/30/20/20 raw scales; normalized values are not accuracy percentages.",
        },
        "tracks": {
            "text": {"eligible_items": 29, "judge_count": 5, "excluded_items": ["text-15"]},
            "image": {"eligible_items": 4, "judge_count": 4, "excluded_judges": ["judge-03"]},
        },
        "models": result_models,
    }


def load_dataset(results_dir: Path) -> dict[str, Any]:
    model_summary = read_csv(results_dir / "model_summary.csv")
    common_summary = read_csv(results_dir / "common_case_model_summary.csv")
    generation_summary = read_csv(results_dir / "translation_generation_summary.csv")
    judge_summary = read_csv(results_dir / "judge_summary.csv")
    comparison = read_csv(results_dir / "translation_comparison.csv")

    common_by_key = {(r.get("track", ""), r.get("model_id", "")): r for r in common_summary}
    generation_by_key = {(r.get("track", ""), r.get("model_id", "")): r for r in generation_summary}
    model_summary_by_key = {(r.get("track", ""), r.get("model_id", "")): r for r in model_summary}

    public_rows = [public_translation_row(row) for row in comparison]
    public_rows.sort(
        key=lambda r: (
            model_index(str(r.get("model_id", ""))),
            0 if r.get("source_kind") == "text" else 1,
            material_number(str(r.get("material_id", ""))),
        )
    )

    curated_examples: dict[str, list[dict[str, Any]]] = {}
    curated_path = results_dir / "curated_examples.json"
    if curated_path.is_file():
        curated_examples = normalize_curated_examples(read_json(curated_path))
    independent_review: dict[str, Any] = {}
    independent_review_path = results_dir / "independent_review.json"
    if independent_review_path.is_file():
        loaded_review = read_json(independent_review_path)
        if isinstance(loaded_review, dict):
            independent_review = loaded_review
    runtime_inventory = load_runtime_inventory(results_dir)

    models: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        model_name = MODEL_SHORT[model_id]
        text_common = common_by_key.get(("text", model_id), {})
        image_common = common_by_key.get(("image", model_id), {})
        text_generation = generation_by_key.get(("text", model_id), {})
        image_generation = generation_by_key.get(("image", model_id), {})
        text_model = model_summary_by_key.get(("text", model_id), {})
        image_model = model_summary_by_key.get(("image", model_id), {})
        models.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "text_score": as_float(text_common.get("common_case_panel_mean")),
                "text_median": as_float(text_common.get("common_case_panel_median")),
                "text_items": as_int(text_common.get("common_case_items")),
                "text_excluded": as_int(text_common.get("excluded_case_count")),
                "text_critical": as_int(text_common.get("critical_flag_cells")),
                "text_warm_median": as_float(text_generation.get("translation_elapsed_median_s")),
                "text_p95": as_float(text_generation.get("translation_elapsed_p95_s")),
                "text_cases": as_int(text_generation.get("translation_cases")),
                "image_score": as_float(image_common.get("common_case_panel_mean")),
                "image_median": as_float(image_common.get("common_case_panel_median")),
                "image_items": as_int(image_common.get("common_case_items")),
                "image_critical": as_int(image_common.get("critical_flag_cells")),
                "image_warm_median": as_float(image_generation.get("translation_elapsed_median_s")),
                "image_p95": as_float(image_generation.get("translation_elapsed_p95_s")),
                "image_cases": as_int(image_generation.get("translation_cases")),
                "text_coverage": text_model.get("coverage_fraction", ""),
                "image_coverage": image_model.get("coverage_fraction", ""),
                "runtime": runtime_inventory.get("by_model", {}).get(model_id, {}),
            }
        )

    criterion_scores = calculate_criterion_scores(results_dir, models)
    for model in models:
        model["criteria"] = criterion_scores.get("models", {}).get(model["model_id"], {})

    return {
        "models": models,
        "rows": public_rows,
        "judges": judge_summary,
        "common_summary": common_summary,
        "generation_summary": generation_summary,
        "model_summary": model_summary,
        "curated_examples": curated_examples,
        "independent_review": independent_review,
        "runtime_inventory": runtime_inventory,
        "criterion_scores": criterion_scores,
    }


def nav_html(current: str, depth: int = 0, lang: str = "zh") -> str:
    prefix = "../" * depth
    if lang == "en":
        links = [
            ("home", "Home", f"{prefix}index-en.html"),
            ("models", "Compare models", f"{prefix}index-en.html#overview-chart"),
            ("results", "238 translations", f"{prefix}results-en.html"),
            ("methods", "Methods & limits", f"{prefix}methods-en.html"),
            ("downloads", "Downloads", f"{prefix}downloads-en.html"),
            ("licenses", "Licensing & sources", f"{prefix}licenses-en.html"),
        ]
    else:
        links = [
            ("home", "首頁", f"{prefix}index.html"),
            ("models", "模型比較", f"{prefix}index.html#overview-chart"),
            ("results", "238 筆翻譯", f"{prefix}results.html"),
            ("methods", "方法與限制", f"{prefix}methods.html"),
            ("downloads", "下載", f"{prefix}downloads.html"),
            ("licenses", "授權與來源", f"{prefix}licenses.html"),
        ]
    nav_links = []
    for key, label, href in links:
        active = ' aria-current="page"' if key == current else ""
        nav_links.append(f'<a class="nav-link{(" active" if key == current else "")}" href="{href}"{active}>{label}</a>')
    return "\n".join(nav_links)


def footer_html(depth: int = 0, lang: str = "zh") -> str:
    prefix = "../" * depth
    english = lang == "en"
    title = "Mental Health Translation Benchmark" if english else "心理健康翻譯基準"
    status = "Evaluation completed 25 October 2026" if english else "評測完成於 2026 年 10 月 25 日"
    methods_label = "Methods & limits" if english else "方法與限制"
    downloads_label = "Downloads" if english else "下載"
    licenses_label = "Licensing & sources" if english else "授權與來源"
    methods_href = f"{prefix}methods-en.html" if english else f"{prefix}methods.html"
    downloads_href = f"{prefix}downloads-en.html" if english else f"{prefix}downloads.html"
    licenses_href = f"{prefix}licenses-en.html" if english else f"{prefix}licenses.html"
    note = (
        "Review the source text, machine translations and judge records together when evaluating a model for your own deployment."
        if english
        else "評估部署時，可以逐筆對照英文原文、模型譯文與評審紀錄。"
    )
    return f"""
    <footer class="site-footer">
      <div class="footer-inner">
        <div><strong>{title}</strong><br><span class="muted">{status}</span></div>
        <div class="footer-links"><a href="{methods_href}">{methods_label}</a><a href="{downloads_href}">{downloads_label}</a><a href="{licenses_href}">{licenses_label}</a></div>
      </div>
      <p class="footer-note">{note}</p>
    </footer>
    """


def page_html(
    title: str,
    current: str,
    body: str,
    depth: int = 0,
    lang: str = "zh",
    switch_href: str | None = None,
    self_href: str | None = None,
    verification_meta: str = "",
) -> str:
    prefix = "../" * depth
    english = lang == "en"
    html_lang = "en" if english else "zh-Hant-TW"
    language_label = "中文" if english else "English"
    description = (
        "Compare seven local models translating mental health materials from English into Taiwan Traditional Chinese. Review scores, response times, model files and the full translations."
        if english and current == "home"
        else "比較七款地端模型將心理健康英文翻成台灣繁體中文的表現。可比較評分、回應時間與模型檔案大小，並檢閱完整譯文。"
        if not english and current == "home"
        else "Scores, response times and source materials for seven local models translating mental health content into Taiwan Traditional Chinese."
        if english
        else "七款地端模型的心理健康英翻中測試，包含評分方法、模型設定、回應時間與逐筆翻譯。"
    )
    if switch_href is None:
        switch_href = f"{prefix}index.html" if english else f"{prefix}index-en.html"
    home_href = f"{prefix}index.html" if not english else f"{prefix}index-en.html"
    current_href = home_href
    if self_href is None:
        self_href = home_href
    page_dir = "models/" if depth else ""
    absolute_url = lambda href: f"{PUBLIC_SITE_BASE}{page_dir}{str(href).lstrip('./')}"
    canonical_url = absolute_url(self_href)
    if english:
        zh_href = switch_href
        en_href = self_href
        brand = '<span class="brand-mark">譯</span><span class="brand-label">Local Model Translation Quality<br>Mental Health · Traditional Chinese<em>Benchmark 2026</em></span>'
    else:
        zh_href = self_href
        en_href = switch_href
        brand = '<span class="brand-mark">譯</span><span class="brand-label">地端模型心理領域中文翻譯品質測量<em>Benchmark 2026</em></span>'
    zh_url = absolute_url(zh_href)
    en_url = absolute_url(en_href)
    verification_html = f'\n  <meta name="google-site-verification" content="{esc(verification_meta)}">' if verification_meta else ""
    comparison_script = f'\n  <script src="{prefix}assets/report_ui.js?v=20261124-2" defer></script>' if current == "home" else ""
    return f"""<!doctype html>
<html lang="{html_lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{esc(description)}">
  <link rel="canonical" href="{esc(canonical_url)}">
  <link rel="alternate" hreflang="zh-Hant" href="{esc(zh_url)}">
  <link rel="alternate" hreflang="en" href="{esc(en_url)}">
  <link rel="alternate" hreflang="x-default" href="{esc(zh_url)}">{verification_html}
  <title>{esc(title if current == 'home' else title + (' · Mental Health Translation Benchmark' if english else ' · 心理健康翻譯基準'))}</title>
  <link rel="stylesheet" href="{prefix}assets/style.css?v=20261124-2">
</head>
<body>
  <a class="skip-link" href="#main">{'Skip to main content' if english else '跳到主要內容'}</a>
  <header class="site-header">
    <div class="header-inner">
      <a class="brand" href="{current_href}">{brand}</a>
      <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="site-nav">{'Menu' if english else '選單'}</button>
      <nav id="site-nav" class="site-nav" aria-label="{'Primary navigation' if english else '主要導覽'}">{nav_html(current, depth, lang)}<a class="language-switch" href="{esc(switch_href)}" lang="{'zh-Hant-TW' if english else 'en'}">{language_label}</a></nav>
    </div>
  </header>
  <main id="main">{body}</main>
  {footer_html(depth, lang)}
  <script src="{prefix}assets/site.js?v=20261124-2" defer></script>{comparison_script}
</body>
</html>
"""


def metric_card(label: str, value: str, note: str = "", tone: str = "teal") -> str:
    return f'''<div class="metric-card {esc(tone)}"><div class="metric-label">{esc(label)}</div><div class="metric-value">{esc(value)}</div><div class="metric-note">{esc(note)}</div></div>'''


def annual_findings(dataset: dict[str, Any], lang: str = "zh") -> str:
    """Render three data-backed, plain-language selection views."""

    english = lang == "en"
    models = dataset.get("models", [])
    if not models:
        return ""
    by_id = {str(model.get("model_id")): model for model in models}
    best_26 = by_id.get("gemma4-26b", models[0])
    best_31 = by_id.get("gemma4-31b-qat", models[0])
    twelve = by_id.get("gemma4-12b", models[0])
    best_image = max(models, key=lambda model: as_float(model.get("image_score")) or float("-inf"))
    text_gap = abs((as_float(best_26.get("text_score")) or 0) - (as_float(best_31.get("text_score")) or 0))
    text_time_gap = abs((as_float(best_26.get("text_warm_median")) or 0) - (as_float(best_31.get("text_warm_median")) or 0))
    image_gap = abs((as_float(best_image.get("image_score")) or 0) - (as_float(twelve.get("image_score")) or 0))
    image_time_gap = abs((as_float(best_image.get("image_warm_median")) or 0) - (as_float(twelve.get("image_warm_median")) or 0))
    twelve_runtime = twelve.get("runtime", {}).get("total_size_gib")
    best_image_runtime = best_image.get("runtime", {}).get("total_size_gib")
    def model_href(model: dict[str, Any]) -> str:
        return f"models/{esc(model['model_id'])}{'-en.html' if english else '.html'}"

    if english:
        return f"""
    <section class="section-shell annual-findings" aria-labelledby="annual-findings-title">
      <div class="section-heading"><div><h2 id="annual-findings-title">Model selection for text and image translation</h2></div></div>
      <div class="finding-grid annual-finding-grid">
        <article class="finding"><h3><span class="finding-use">Text quality and wait</span><span class="finding-model">{esc(best_26['model_name'])}</span></h3><div class="finding-metrics"><div><span>Shared text score</span><strong>{display_number(best_26['text_score'])}<small> points</small></strong></div><div><span>Median response</span><strong>{display_number(best_26['text_warm_median'])}<small> s</small></strong></div></div><p>{esc(best_31['model_name'])} averaged {display_number(best_31['text_score'])} points in {display_number(best_31['text_warm_median'])} s. The 26B led by {display_number(text_gap)} points and took {display_number(text_time_gap)} s less per request.</p><small>29 shared text cases · <a class="text-link" href="{model_href(best_26)}">{esc(best_26['model_name'])} page</a> · <a class="text-link" href="{model_href(best_31)}">{esc(best_31['model_name'])} page</a></small></article>
        <article class="finding"><h3><span class="finding-use">Image quality</span><span class="finding-model">{esc(best_image['model_name'])}</span></h3><div class="finding-metrics"><div><span>Four-image score</span><strong>{display_number(best_image['image_score'])}<small> points</small></strong></div><div><span>Median response</span><strong>{display_number(best_image['image_warm_median'])}<small> s</small></strong></div></div><p>This model had the highest mean on the four images. Its model and vision-projector files occupied {display_number(best_image_runtime, 1)} GiB.</p><small>Four images only · <a class="text-link" href="{model_href(best_image)}">View model details</a></small></article>
        <article class="finding"><h3><span class="finding-use">Shorter image wait</span><span class="finding-model">{esc(twelve['model_name'])}</span></h3><div class="finding-metrics"><div><span>Four-image score</span><strong>{display_number(twelve['image_score'])}<small> points</small></strong></div><div><span>Median response</span><strong>{display_number(twelve['image_warm_median'])}<small> s</small></strong></div></div><p>Compared with {esc(best_image['model_name'])}, this is {display_number(image_gap)} points lower but {display_number(image_time_gap)} s faster. Its files occupy {display_number(twelve_runtime, 1)} GiB.</p><small>Four images only · <a class="text-link" href="{model_href(twelve)}">View model details</a></small></article>
      </div>
    </section>
    """
    return f"""
    <section class="section-shell annual-findings" aria-labelledby="annual-findings-title">
      <div class="section-heading"><div><h2 id="annual-findings-title">文字與圖片翻譯的模型選擇</h2></div></div>
      <div class="finding-grid annual-finding-grid">
        <article class="finding"><h3><span class="finding-use">文字品質與等待時間</span><span class="finding-model">{esc(best_26['model_name'])}</span></h3><div class="finding-metrics"><div><span>共同文字均分</span><strong>{display_number(best_26['text_score'])}<small> 分</small></strong></div><div><span>回應時間中位數</span><strong>{display_number(best_26['text_warm_median'])}<small> 秒</small></strong></div></div><p>{esc(best_31['model_name'])} 平均 {display_number(best_31['text_score'])} 分，回應時間中位數 {display_number(best_31['text_warm_median'])} 秒。26B 高 {display_number(text_gap)} 分，每篇快 {display_number(text_time_gap)} 秒。</p><small>29 篇共同文字題目 · <a class="text-link" href="{model_href(best_26)}">{esc(best_26['model_name'])} 模型頁</a> · <a class="text-link" href="{model_href(best_31)}">{esc(best_31['model_name'])} 模型頁</a></small></article>
        <article class="finding"><h3><span class="finding-use">圖片品質</span><span class="finding-model">{esc(best_image['model_name'])}</span></h3><div class="finding-metrics"><div><span>四張圖片均分</span><strong>{display_number(best_image['image_score'])}<small> 分</small></strong></div><div><span>回應時間中位數</span><strong>{display_number(best_image['image_warm_median'])}<small> 秒</small></strong></div></div><p>在本次四張圖片中平均分數最高。模型與視覺投影器檔案合計 {display_number(best_image_runtime, 1)} GiB。</p><small>只有 4 張圖片 · <a class="text-link" href="{model_href(best_image)}">查看模型頁</a></small></article>
        <article class="finding"><h3><span class="finding-use">圖片等待時間較短</span><span class="finding-model">{esc(twelve['model_name'])}</span></h3><div class="finding-metrics"><div><span>四張圖片均分</span><strong>{display_number(twelve['image_score'])}<small> 分</small></strong></div><div><span>回應時間中位數</span><strong>{display_number(twelve['image_warm_median'])}<small> 秒</small></strong></div></div><p>相較 {esc(best_image['model_name'])} 少 {display_number(image_gap)} 分、快 {display_number(image_time_gap)} 秒。模型與視覺投影器檔案合計 {display_number(twelve_runtime, 1)} GiB。</p><small>只有 4 張圖片 · <a class="text-link" href="{model_href(twelve)}">查看模型頁</a></small></article>
      </div>
    </section>
    """


def hero_condition_line(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    models = dataset.get("models", [])
    text_cases = max((as_int(model.get("text_cases")) for model in models), default=30)
    shared_text = max((as_int(model.get("text_items")) for model in models), default=29)
    image_cases = max((as_int(model.get("image_cases")) for model in models), default=4)
    fixed = dataset.get("runtime_inventory", {}).get("fixed_generation", {})
    build = dataset.get("runtime_inventory", {}).get("server_build") or "—"
    platform = dataset.get("runtime_inventory", {}).get("platform") or "—"
    if english:
        return f'<p class="hero-condition-line">2026/10 · English → Taiwan Traditional Chinese · {len(models)} local models · {shared_text} of {text_cases} text cases shared · {image_cases} images · temperature {esc(fixed.get("temperature", "—"))} · context {as_int(fixed.get("ctx_size")):,} · {esc(build)} / {esc(platform)}</p>'
    return f'<p class="hero-condition-line">2026/10 · 心理健康英文 → 台灣繁中 · {len(models)} 款地端模型 · {shared_text}/{text_cases} 篇文字為共同題目 · {image_cases} 張圖片 · temperature {esc(fixed.get("temperature", "—"))} · context {as_int(fixed.get("ctx_size")):,} · {esc(build)} / {esc(platform)}</p>'


def reliability_metric(dataset: dict[str, Any], track: str) -> dict[str, Any]:
    metrics = dataset.get("independent_review", {}).get("reliability_metrics", [])
    if not isinstance(metrics, list):
        return {}
    return next((item for item in metrics if isinstance(item, dict) and item.get("track") == track), {})


def judge_explainer(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    suffix = "-en.html" if english else ".html"
    reliability = reliability_metric(dataset, "text")
    single = display_number(reliability.get("icc_2_1_absolute_agreement"))
    panel = display_number(reliability.get("icc_2_k_absolute_agreement"))
    if english:
        title = "How LLMs scored the translations"
        summary = "Five LLM judges (Qwen, Nemotron, Mistral, GPT and Gemini) compared each translation with the English source without seeing the model name. The text score averages the five judges for each passage, then averages the 29 shared passages. Four judges scored all four images. Mistral did not support images in this setup."
        dimensions = [("30 · Terminology and Taiwan usage", "Clinical terms, character forms and local usage."), ("30 · Meaning and safety", "Preservation of negation, drug names, doses, numbers and conditions."), ("20 · Fluency", "Clarity and natural wording in Taiwan Traditional Chinese."), ("20 · Completeness and format", "Omissions, additions and changes to lists or visible structure.")]
        agreement = f"On text, the intraclass correlation coefficient (ICC) was {single} for one judge and {panel} for the average of five. A higher ICC indicates more consistent ratings. The 100-point total summarizes these four criteria. A score of 98 does not imply that 98% of a translation is correct."
        human = "The translations did not undergo item-by-item human review or scoring."
        links = ("Inspect the sources and translations", "Full rubric and ICC estimates")
    else:
        title = "本實測如何用LLM協助評分"
        summary = "Qwen、Nemotron、Mistral、GPT 與 Gemini 五個大型語言模型（LLM）擔任評審。它們對照英文原文與匿名譯文，依四項標準給分。文字先算每篇的五位評審平均，再對 29 篇共同題目取平均。圖片由四位評審評完四張，Mistral 在本次設定下不支援圖片。"
        dimensions = [("30 分 · 術語與台灣用語", "心理與醫療術語是否貼切，字形與用詞是否符合台灣習慣。"), ("30 分 · 語意與安全", "否定、藥名、劑量、數字與適用條件有沒有譯錯。"), ("20 分 · 流暢度", "繁體中文是否自然、清楚，讀者能否順暢理解。"), ("20 分 · 完整性與格式", "有無漏譯、增譯，清單與可見結構是否保留。")]
        agreement = f"文字評分的組內相關係數（ICC）為單一評審 {single}、五位評審平均 {panel}。ICC 越高，表示評審給分越一致。總分由四項加總，流暢度占 20 分。例如 98 分，代表在這套標準上平均得到 98 分，無法換算成譯文正確的百分比。"
        human = "本實測未經人工逐筆審閱與評分，判斷實際翻譯品質時可以直接對照原文與譯文。"
        links = ("逐筆檢閱原文與譯文", "完整評分標準與 ICC 估計")
    items = "".join(f'<div class="rubric-mini-item"><b>{a}</b><span>{b}</span></div>' for a, b in dimensions)
    return f"""<section class="section-shell judge-explainer" id="judge-rubric"><div class="judge-explainer-panel">
      <h2>{title}</h2><p>{summary}</p><div class="rubric-mini-grid">{items}</div>
      <div class="judge-icc-mini"><span>{agreement}</span><p>{human}</p><a class="text-link" href="results{suffix}">{links[0]}</a><a class="text-link" href="methods{suffix}#rubric-icc">{links[1]}</a></div>
    </div></section>"""


def rubric_reliability_section(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    text_metric = reliability_metric(dataset, "text")
    image_metric = reliability_metric(dataset, "image")
    text_one = display_number(text_metric.get("icc_2_1_absolute_agreement"), 2)
    text_k = display_number(text_metric.get("icc_2_k_absolute_agreement"), 2)
    image_one = display_number(image_metric.get("icc_2_1_absolute_agreement"), 2)
    image_k = display_number(image_metric.get("icc_2_k_absolute_agreement"), 2)
    text_units = display_integer(text_metric.get("complete_balanced_units"))
    image_units = display_integer(image_metric.get("complete_balanced_units"))
    if english:
        return f"""
    <section class="section-shell rubric-detail" id="rubric-icc"><div class="section-heading"><div><div class="eyebrow">Scoring rubric and reviewer agreement</div><h2>Four score dimensions and the ICC units</h2><p class="section-intro">The public instructions define four integer subscores for text and image jobs. They sum to the 0–100 score; the rubric is not a fixed error-penalty table.</p></div></div><div class="rubric-table-wrap"><table class="rubric-table"><thead><tr><th>Scoring dimension</th><th>Points</th><th>What reviewers inspect</th></tr></thead><tbody><tr><td>Terminology &amp; Taiwan usage</td><td>0–30</td><td>Medical/psychological terms and Taiwan Traditional Chinese, not Simplified Chinese.</td></tr><tr><td>Meaning &amp; safety facts</td><td>0–30</td><td>Meaning, negation, medication, dose/number/unit, frequency, timing, conditions and suicide/self-harm meaning.</td></tr><tr><td>Fluency</td><td>0–20</td><td>Natural, clear Taiwan Traditional Chinese register.</td></tr><tr><td>Completeness &amp; format</td><td>0–20</td><td>No omission/addition; preserve visible content, paragraph/list structure and relevant layout cues.</td></tr></tbody></table></div><div class="icc-detail-grid"><article><h3>Text reviewer agreement · {text_units} translation units · k=5</h3><p><b>ICC(2,1) = {text_one}</b><br><span>Absolute-agreement point estimate for one reviewer.</span></p><p><b>ICC(2,5) = {text_k}</b><br><span>Point estimate after averaging the actual five-reviewer panel.</span></p></article><article><h3>Image reviewer agreement · {image_units} translation units · k=4</h3><p><b>ICC(2,1) = {image_one}</b><br><span>Absolute-agreement point estimate for one reviewer.</span></p><p><b>ICC(2,4) = {image_k}</b><br><span>Point estimate after averaging four reviewers; Mistral is excluded.</span></p></article></div><p class="rubric-note">ICC is computed across {text_units} text translation units (29 cases × 7 models), each with five reviewer scores, and {image_units} image units (4 cases × 7 models), each with four scores—not across seven model means. ICC values describe agreement in reviewer scoring, not translation accuracy, an 87.4% correctness rate, or clinical safety. No confidence intervals are reported here; these are point estimates.</p><div class="source-links rubric-sources"><a class="text-link" href="https://github.com/gregorytsai/psych-translation-benchmark/blob/main/benchmark_code/translation_benchmark_20261023_new_materials/judge_evaluation/public/instructions.md" target="_blank" rel="noopener">Open the public repository rubric ↗</a><a class="text-link" href="data/independent_review.json">Open the public ICC/audit data ↗</a><a class="text-link" href="https://pubmed.ncbi.nlm.nih.gov/27330520/" target="_blank" rel="noopener">ICC reference (Koo &amp; Li) ↗</a></div></section>
        """
    return f"""
    <section class="section-shell rubric-detail" id="rubric-icc"><div class="section-heading"><div><div class="eyebrow">評分規準與評審一致性</div><h2>四項分數與 ICC 計算單位</h2><p class="section-intro">公開 instructions.md 對文字與圖片都定義四項整數分數；四項相加才是 0–100 總分，rubric 不是固定錯誤扣分表。</p></div></div><div class="rubric-table-wrap"><table class="rubric-table"><thead><tr><th>評分項目</th><th>原始配分</th><th>評審檢查內容</th></tr></thead><tbody><tr><td>術語與台灣用語</td><td>0–30</td><td>醫療/心理術語與台灣繁體中文，不使用簡體中文。</td></tr><tr><td>語意與安全事實</td><td>0–30</td><td>原意、否定、藥名、劑量/數字/單位、頻率、時間、條件，以及自殺/自傷意義。</td></tr><tr><td>流暢度</td><td>0–20</td><td>自然、清楚的台灣繁中語域。</td></tr><tr><td>完整與格式</td><td>0–20</td><td>不漏譯、不增譯，保留可見內容、段落/清單結構與版面線索。</td></tr></tbody></table></div><div class="icc-detail-grid"><article><h3>文字評審一致性｜{text_units} 份翻譯｜k=5</h3><p><b>ICC(2,1) = {text_one}</b><br><span>單一評審的絕對一致性點估計。</span></p><p><b>ICC(2,5) = {text_k}</b><br><span>平均當次 5 位評審後的點估計。</span></p></article><article><h3>圖片評審一致性｜{image_units} 份翻譯｜k=4</h3><p><b>ICC(2,1) = {image_one}</b><br><span>單一評審的絕對一致性點估計。</span></p><p><b>ICC(2,4) = {image_k}</b><br><span>平均 4 位評審後的點估計；不含 Mistral。</span></p></article></div><p class="rubric-note">ICC 是在 {text_units} 份文字翻譯（29 題 × 7 模型、每份 5 位評審）與 {image_units} 份圖片翻譯（4 題 × 7 模型、每份 4 位評審）上計算，不是對 7 個模型平均分計算。它衡量給分一致，不是翻譯正確率、87.4% 正確率或臨床安全；本頁未報告信賴區間，數字是點估計。</p><div class="source-links rubric-sources"><a class="text-link" href="https://github.com/gregorytsai/psych-translation-benchmark/blob/main/benchmark_code/translation_benchmark_20261023_new_materials/judge_evaluation/public/instructions.md" target="_blank" rel="noopener">開啟公開 repo 原始 rubric ↗</a><a class="text-link" href="data/independent_review.json">開啟公開 ICC／audit data ↗</a><a class="text-link" href="https://pubmed.ncbi.nlm.nih.gov/27330520/" target="_blank" rel="noopener">ICC 參考文獻（Koo &amp; Li）↗</a></div></section>
    """


def comparison_tool(dataset: dict[str, Any], lang: str = "zh") -> str:
    """Render the chart and exact table from one shared, filterable data state."""

    english = lang == "en"
    metric_specs = (
        ("total", "總分", "Total score"),
        ("terminology_locale", "術語與台灣用語", "Terminology & Taiwan usage"),
        ("semantic_safety", "語意與安全", "Meaning & safety"),
        ("fluency", "流暢度", "Fluency"),
        ("completeness_format", "完整性與格式", "Completeness & format"),
    )
    track_specs = (("both", "文字＋圖片", "Text + image"), ("text", "只文字", "Text only"), ("image", "只圖片", "Image only"))
    sort_specs = (("score_desc", "分數高→低", "Quality high → low"), ("time_asc", "回應時間快→慢", "Response time fast → slow"))

    def metric_value(model: dict[str, Any], track: str, metric: str) -> float | None:
        if metric == "total":
            return as_float(model.get(f"{track}_score"))
        return as_float(model.get("criteria", {}).get(track, {}).get("dimensions", {}).get(metric, {}).get("normalized"))

    def display_digits_for(metric: str) -> int:
        """Use the fewest decimals that keep each track's raw values distinct."""

        groups: list[list[float]] = []
        for track in ("text", "image"):
            values = [metric_value(model, track, metric) for model in dataset.get("models", [])]
            present = [value for value in values if value is not None]
            if present:
                groups.append(present)
        for digits in range(1, 7):
            if all(
                len({display_number(value, digits) for value in values})
                == len(set(values))
                for values in groups
            ):
                return digits
        return 6

    metric_digits = {metric: display_digits_for(metric) for metric, _zh, _en in metric_specs}

    def raw_value(model: dict[str, Any], track: str, metric: str) -> float | None:
        if metric == "total":
            return as_float(model.get("criteria", {}).get(track, {}).get("total", {}).get("raw_mean")) or as_float(model.get(f"{track}_score"))
        return as_float(model.get("criteria", {}).get(track, {}).get("dimensions", {}).get(metric, {}).get("raw_mean"))

    def metric_display(value: Any, metric: str) -> str:
        return display_number(value, metric_digits.get(metric, 2))

    def model_attrs(model: dict[str, Any], order: int) -> str:
        attrs = [f'data-model-id="{esc(model["model_id"])}"', f'data-default-order="{order}"']
        for track in ("text", "image"):
            for metric, _zh, _en in metric_specs:
                key = metric
                value = metric_value(model, track, metric)
                attrs.append(f'data-{track}-{key}="{esc("" if value is None else value)}"')
                attrs.append(f'data-{track}-{key}-display="{esc("" if value is None else metric_display(value, metric))}"')
                maximum = 100 if metric == "total" else next(item[3] for item in CRITERION_DIMENSIONS if item[0] == metric)
                raw = raw_value(model, track, metric)
                attrs.append(f'data-{track}-{key}-raw="{esc("" if raw is None else raw)}"')
                attrs.append(f'data-{track}-{key}-max="{maximum}"')
            warm = as_float(model.get(f"{track}_warm_median"))
            attrs.append(f'data-{track}-warm="{esc("" if warm is None else warm)}"')
            attrs.append(f'data-{track}-warm-display="{esc("" if warm is None else display_number(warm))}"')
        return " ".join(attrs)

    def metric_button(metric: str, zh_label: str, en_label: str) -> str:
        label = en_label if english else zh_label
        return f'<button type="button" class="overview-tab" data-overview-metric="{metric}" aria-pressed="{"true" if metric == "total" else "false"}">{esc(label)}</button>'

    def track_button(track: str, zh_label: str, en_label: str) -> str:
        label = en_label if english else zh_label
        return f'<button type="button" class="overview-track-tab" data-overview-track="{track}" aria-pressed="{"true" if track == "both" else "false"}">{esc(label)}</button>'

    def sort_button(sort: str, zh_label: str, en_label: str) -> str:
        label = en_label if english else zh_label
        return f'<button type="button" class="overview-sort-tab" data-overview-sort="{sort}" aria-pressed="{"true" if sort == "score_desc" else "false"}">{esc(label)}</button>'

    chart_rows: list[str] = []
    table_rows: list[str] = []
    models = sorted(dataset["models"], key=lambda item: (-(metric_value(item, "text", "total") or -1), item["model_name"]))
    for order, model in enumerate(models, 1):
        text_value = metric_value(model, "text", "total") or 0
        image_value = metric_value(model, "image", "total") or 0
        text_display = metric_display(text_value, "total")
        image_display = metric_display(image_value, "total")
        text_time = max(0.0, min(6.0, float(model["text_warm_median"] or 0)))
        image_time = max(0.0, min(6.0, float(model["image_warm_median"] or 0)))
        text_label = "Text" if english else "文字"
        image_label = "Image" if english else "圖片"
        model_href = f"models/{esc(model['model_id'])}{'-en.html' if english else '.html'}"
        attrs = model_attrs(model, order)
        chart_rows.append(
            f'<div class="overview-row" role="listitem" {attrs}>'
            f'<div class="overview-name"><span class="overview-rank" title="{esc("Descriptive order for this run" if english else "本次條件下的描述性排序")}">{order}</span><a href="{model_href}">{esc(model["model_name"])}</a></div>'
            f'<div class="overview-scores">'
            f'<div class="overview-line text-line" data-overview-line-track="text"><span class="overview-key text-key">{text_label}</span><div class="overview-track" role="img" aria-label="{esc(text_label)} score {text_display} (0–100)"><i class="overview-fill text-fill" style="width: {float(text_value):.3f}%"></i></div><b><span class="overview-value">{text_display}</span><small class="overview-raw"></small></b></div>'
            f'<div class="overview-line image-line" data-overview-line-track="image"><span class="overview-key image-key">{image_label}</span><div class="overview-track" role="img" aria-label="{esc(image_label)} score {image_display} (0–100)"><i class="overview-fill image-fill" style="width: {float(image_value):.3f}%"></i></div><b><span class="overview-value">{image_display}</span><small class="overview-raw"></small></b></div>'
            f'</div><div class="overview-speed">'
            f'<div class="speed-line" data-overview-speed-track="text"><span>{text_label} {display_number(model["text_warm_median"])}s</span><div class="speed-track" aria-label="{esc("Text response time, 0 to 6 seconds" if english else "文字回應時間，0 到 6 秒")}"><i style="left:{(text_time / 6) * 100:.2f}%"></i></div></div>'
            f'<div class="speed-line" data-overview-speed-track="image"><span>{image_label} {display_number(model["image_warm_median"])}s</span><div class="speed-track" aria-label="{esc("Image response time, 0 to 6 seconds" if english else "圖片回應時間，0 到 6 秒")}"><i style="left:{(image_time / 6) * 100:.2f}%"></i></div></div>'
            f'</div></div>'
        )
        text_runtime = display_number(model.get("runtime", {}).get("total_size_gib"), 1)
        table_rows.append(
            f'<tr class="overview-table-row" {attrs}>'
            f'<th scope="row"><span class="overview-table-rank">{order}</span><a href="{model_href}">{esc(model["model_name"])}</a></th>'
            f'<td class="overview-table-cell overview-table-score-cell" data-table-track="text"><span class="overview-table-score">{text_display}</span><small class="overview-table-raw"></small></td>'
            f'<td class="overview-table-cell overview-table-score-cell" data-table-track="image"><span class="overview-table-score">{image_display}</span><small class="overview-table-raw"></small></td>'
            f'<td class="overview-table-cell overview-table-time-cell" data-table-track="text"><span class="overview-table-time">{display_number(model["text_warm_median"])}</span> s</td>'
            f'<td class="overview-table-cell overview-table-time-cell" data-table-track="image"><span class="overview-table-time">{display_number(model["image_warm_median"])}</span> s</td>'
            f'<td class="overview-table-files">{text_runtime} GiB</td></tr>'
        )

    if english:
        heading = "Seven-model comparison · Text + image · Total score"
        intro = "The bars show reviewer scores from 0–100. Dots show median response time after model loading, on a common 0–6 s scale. When both input types are shown, the order follows the text results."
        default_note = "The total is the mean of the four-part rubric score, out of 100."
        noscript = "The chart and table below show the total score for both input types. Enable JavaScript to change the comparison."
        table_label = "Model comparison table · Text + image · Total score"
    else:
        heading = "七款模型視覺總覽 · 文字＋圖片 · 總分"
        intro = "長條表示評審分數（0–100 分），圓點表示模型載入後的回應時間中位數（0–6 秒）。同時顯示文字和圖片時，依文字結果排序。"
        default_note = "總分是四項評分加總後的平均，滿分 100 分。"
        noscript = "以下圖表列出七款模型的文字與圖片總分。啟用 JavaScript 可切換素材、指標與排序。"
        table_label = "模型比較表 · 文字＋圖片 · 總分"
    metric_buttons = "".join(metric_button(*spec) for spec in metric_specs)
    track_buttons = "".join(track_button(*spec) for spec in track_specs)
    sort_buttons = "".join(sort_button(*spec) for spec in sort_specs)
    file_header = "Model + vision projector (GiB)" if english else "模型＋視覺投影器檔案（GiB）"
    return f"""
    <section class="section-shell overview-section" id="overview-chart" aria-labelledby="overview-heading">
      <div class="section-heading"><div><div class="eyebrow">{'Scores and response times' if english else '分數與回應時間'}</div><h2 id="overview-heading">{heading}</h2><p class="section-intro">{intro}</p></div><a class="text-link" href="{'methods-en.html#rubric-icc' if english else 'methods.html#rubric-icc'}">{'Read score and ICC notes →' if english else '看分數與 ICC 說明 →'}</a></div>
      <div class="overview-control-panel">
        <fieldset class="overview-control-group"><legend>{'Score dimension' if english else '分數指標'}</legend><div class="overview-controls">{metric_buttons}</div></fieldset>
        <fieldset class="overview-control-group"><legend>{'Input type' if english else '素材類型'}</legend><div class="overview-track-controls">{track_buttons}</div></fieldset>
        <fieldset class="overview-control-group"><legend>{'Sort' if english else '排序'}</legend><div class="overview-sort-controls">{sort_buttons}</div></fieldset>
      </div>
      <p class="overview-scale-note" id="overview-scale-note" aria-live="polite">{default_note}</p><p class="overview-sort-note" id="overview-sort-note" aria-live="polite">{'Ordered by text total score, highest first. Equal values share a rank.' if english else '依文字總分由高到低排列。原始數值相同時並列。'}</p><noscript><p class="overview-noscript">{noscript}</p></noscript>
      <div class="overview-legend"><span data-overview-legend-track="text"><i class="legend-dot teal"></i>{'Text score' if english else '文字分數'}</span><span data-overview-legend-track="image"><i class="legend-dot coral"></i>{'Image score' if english else '圖片分數'}</span><span class="speed-legend"><i></i>{'Response time, 0–6 s' if english else '回應時間，0–6 秒'}</span></div>
      <div class="overview-chart" role="list">{"".join(chart_rows)}</div>
      <div class="overview-table-wrap"><table class="overview-table" id="overview-table"><caption id="overview-table-heading">{table_label}</caption><thead><tr><th scope="col">{'Rank / model' if english else '排名／模型'}</th><th scope="col" data-table-track="text" data-table-measure="score">{'Text score' if english else '文字分數'}</th><th scope="col" data-table-track="image" data-table-measure="score">{'Image score' if english else '圖片分數'}</th><th scope="col" data-table-track="text" data-table-measure="time">{'Text time (s)' if english else '文字回應時間（秒）'}</th><th scope="col" data-table-track="image" data-table-measure="time">{'Image time (s)' if english else '圖片回應時間（秒）'}</th><th scope="col">{file_header}</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
      <div class="source-links rubric-sources"><a class="text-link" href="{'methods-en.html#criterion-ranks' if english else 'methods.html#criterion-ranks'}">{'How to interpret changes in rank →' if english else '如何解讀不同指標的排名 →'}</a><span class="source-condition">{'Values are rounded for display. Ranks use the underlying scores or times.' if english else '表中的數字經過四捨五入，排名以完整精度計算。'}</span></div>
    </section>
    """


def image_case_highlight(lang: str = "zh") -> str:
    english = lang == "en"
    image = "materials/images/img04_synthetic_medication_label.png"
    href = "models/gemma4-26b" + ("-en.html" if english else ".html") + "#img-04-example"
    title = "Drug-name transcription on a synthetic label" if english else "合成藥品標籤的藥名辨識"
    text = (
        "In img-04, Gemma 4 26B transcribed <code>SERTRALINE</code> as <code>SETRALINE</code>, dropping the first R. All four image judges flagged the mismatch. Their mean score for this case was 72 out of 100."
        if english else "在 img-04 中，原圖的藥名是 <code>SERTRALINE</code>，Gemma 4 26B 輸出成 <code>SETRALINE</code>，漏了第一個 R。四位圖片評審都標記了這個錯誤，這張圖的平均分為 72 分。"
    )
    alt = "Test image img-04, a synthetic medication label" if english else "測試圖片 img-04，自製模擬用藥標籤"
    return f"""<section class="section-shell image-case-highlight"><article>
      <div><h2>{title}</h2><p>{text}</p><a class="text-link" href="{href}">{"Read the full translation and judge comments" if english else "對照完整譯文與評語"}</a></div>
      <a class="image-case-thumbnail" href="{image}"><img src="{image}" alt="{alt}" width="1600" height="900" loading="lazy"></a>
    </article></section>"""


def home_body(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    suffix = "-en.html" if english else ".html"
    methods = "methods" + suffix
    results = "results" + suffix
    title = "Seven local models tested" if english else "七款地端模型實測"
    lead = (
        "For mental health text translated into Taiwan Traditional Chinese, Gemma 4 26B and 31B QAT had close mean scores, with shorter waits on 26B. The 31B QAT model scored highest on the four images. The comparisons below pair each rubric score with the median time to return a translation after the model was loaded."
        if english else
        "將心理健康英文翻成台灣繁中時，Gemma 4 26B 與 31B QAT 的文字平均分數接近，26B 的回應較快。四張圖片則以 31B QAT 的平均分最高。下列比較將評審分數與模型載入後完成翻譯的時間並列，方便依工作量與等待時間選擇模型。"
    )
    limits = (
        "The text comparison covers 29 passages from 12 source pages. The image comparison covers four layouts. Timing uses all 30 text requests and all four image requests per model, on one hardware and software configuration. The file sizes are disk space occupied by the model and its vision projector. Peak RAM was not measured."
        if english else
        "文字評分涵蓋 12 個來源頁面的 29 篇文字，圖片評分涵蓋四張不同版面的素材。回應時間以每款模型全部 30 筆文字與 4 筆圖片請求計算，適用於本次硬體、量化與推論設定。檔案大小是模型與視覺投影器的磁碟用量，本次未量測記憶體峰值。"
    )
    links = [
        (results, "All 238 source texts and translations" if english else "238 筆原文與譯文"),
        ("downloads" + suffix, "Code, results and test materials" if english else "程式碼、結果與測試素材"),
        ("licenses" + suffix, "Sources and licensing" if english else "來源與授權"),
    ]
    link_html = "".join(f'<a class="link-tile" href="{url}">{label}</a>' for url, label in links)
    hero = f"""<section class="hero section-shell"><div class="hero-copy">
      {hero_condition_line(dataset, lang)}<h1>{title}</h1><p class="hero-lede">{lead}</p>
      <div class="hero-actions"><a class="button primary" href="#overview-chart">{"Compare all seven models" if english else "比較七款模型"}</a><a class="button ghost" href="#judge-rubric">{"How the scores were calculated" if english else "評分方式與信度"}</a></div>
    </div></section>"""
    limitation = f'<section class="section-shell callout caution"><div><h2>{"Limits of this run" if english else "本測驗的限制"}</h2><p>{limits}</p><a class="text-link" href="{methods}">{"Methods and test settings" if english else "完整方法與測試設定"}</a></div></section>'
    data_links = f'<section class="section-shell"><h2>{"Translations, code and materials" if english else "譯文、程式碼與素材"}</h2><div class="link-stack">{link_html}</div></section>'
    return "\n".join(section.strip() for section in (hero, annual_findings(dataset, lang), comparison_tool(dataset, lang), judge_explainer(dataset, lang), limitation, image_case_highlight(lang), data_links))


def runtime_specs_html(model: dict[str, Any], dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    runtime = model.get("runtime", {})
    fixed = dataset.get("runtime_inventory", {}).get("fixed_generation", {})
    if not runtime:
        return ""
    model_size = display_number(runtime.get("model_size_gib"), 1)
    mmproj_size = display_number(runtime.get("mmproj_size_gib"), 1)
    total_size = display_number(runtime.get("total_size_gib"), 1)
    file_label = "GGUF model file" if english else "GGUF 模型檔"
    projector_label = "Vision projector (mmproj)" if english else "視覺投影器（mmproj）"
    total_label = "Model + mmproj on-disk total" if english else "模型＋mmproj 磁碟檔案合計"
    mode_label = "Chat/template mode" if english else "對話／模板模式"
    alias_label = "Runtime alias" if english else "執行別名"
    model_card_label = "Official model documentation" if english else "官方模型文件"
    model_doc = "https://ai.google.dev/gemma/docs/releases" if model["model_id"].startswith("translate") else "https://ai.google.dev/gemma/docs/core/model_card_4"
    model_doc_text = "TranslateGemma release page" if english and model["model_id"].startswith("translate") else "TranslateGemma 官方發布頁" if model["model_id"].startswith("translate") else "Gemma 4 model card" if english else "Gemma 4 官方 model card"
    fixed_text = (
        f"context {fixed.get('ctx_size', '—')} · max output {fixed.get('max_tokens', '—')} · temperature {fixed.get('temperature', '—')} · seed {fixed.get('seed', '—')} · thinking {fixed.get('reasoning', '—')}"
        if english
        else f"context {fixed.get('ctx_size', '—')} · 輸出上限 {fixed.get('max_tokens', '—')} · temperature {fixed.get('temperature', '—')} · seed {fixed.get('seed', '—')} · thinking {fixed.get('reasoning', '—')}"
    )
    note = "These are on-disk GGUF file sizes, not a RAM requirement; runtime memory also depends on context, projector, buffers and settings." if english else "磁碟合計由原始位元組相加後換算。執行時的記憶體用量也受上下文長度、視覺投影器與推論緩衝區影響。"
    sha_label = "SHA-256 checksums" if english else "SHA-256 雜湊"
    settings_label = "Measured translation inference settings" if english else "本次實測翻譯推論設定"
    return f"""
    <section class="section-shell runtime-section"><div class="section-heading"><div><div class="eyebrow">{'Measured files and settings' if english else '實測檔案與推論設定'}</div><h2>{'Model files, vision projector, and inference settings' if english else '模型檔、視覺投影器與推論設定'}</h2><p class="section-intro">{'The run used the GGUF weights, vision projector and chat template listed here. Match the filenames and SHA-256 checksums when reproducing this configuration.' if english else '本次實測使用以下 GGUF 權重、視覺投影器與對話模板。要在自己的硬體上重測，可用檔名及下方 SHA-256 核對相同版本。'}</p></div></div><div class="runtime-grid"><div class="runtime-facts"><div><b>{file_label}</b><code>{esc(runtime.get('model_filename'))}</code><span>{model_size} GiB</span></div><div><b>{projector_label}</b><code>{esc(runtime.get('mmproj_filename'))}</code><span>{mmproj_size} GiB</span></div><div><b>{total_label}</b><strong>{total_size} GiB</strong></div><div><b>{mode_label}</b><span>{esc(runtime.get('template_mode'))}</span></div><div><b>{alias_label}</b><span>{esc(runtime.get('alias'))}</span></div></div><div class="runtime-settings"><b>{settings_label}</b><p>{esc(fixed_text)}</p><p class="runtime-note">{esc(note)}</p><p><a class="text-link" href="{model_doc}">{esc(model_card_label)}：{esc(model_doc_text)} ↗</a></p></div></div><details class="sha-details"><summary>{sha_label}</summary><div><code>model: {esc(runtime.get('model_sha256'))}</code><code>mmproj: {esc(runtime.get('mmproj_sha256'))}</code></div></details></section>
    """


def image_sources_panel(dataset: dict[str, Any], lang: str = "zh", depth: int = 0) -> str:
    english = lang == "en"
    names = {
        "img-01": ("合成的心理健康紀錄", "Synthetic mental health note"),
        "img-04": ("合成的藥品標籤", "Synthetic medication label"),
        "img-09": ("WHO 自殺除罪化政策簡報，第 1 頁", "WHO policy brief on decriminalization of suicide, page 1"),
        "img-10": ("WHO 憂鬱症手冊封面", "WHO depression brochure, front cover"),
    }
    cards = []
    for image_id, labels in names.items():
        source = dataset["image_sources"].get(image_id, {})
        title = labels[1] if english else labels[0]
        digest = esc(source.get("sha256") or "")
        checksum = ""
        if digest:
            checksum = f'<details class="sha-details"><summary>{"Tested PNG SHA-256" if english else "實測 PNG 的 SHA-256"}</summary><code>{digest}</code></details>'
        cards.append(f'<article class="source-card"><div class="case-label">{image_id}</div><h3>{title}</h3>{image_source_links(dataset, image_id, lang, depth)}{checksum}</article>')
    heading = "Source materials for the four image cases" if english else "四張測試圖片的來源"
    detail = (
        "The two synthetic images are available as tested. For the two WHO publications, the links open the source page in the official PDF. A checksum identifies each PNG used in the run."
        if english else
        "兩張自製圖提供當次測試的 PNG。另兩張取自 WHO 出版品，連結可開啟官方 PDF 的實測頁。每項的 SHA-256 可用來核對當時使用的 PNG。"
    )
    return f'<section class="section-shell image-sources-section"><h2>{heading}</h2><p>{detail}</p><div class="source-card-grid">{"".join(cards)}</div></section>'


def image_source_links(dataset: dict[str, Any], image_id: str, lang: str = "zh", depth: int = 1) -> str:
    english = lang == "en"
    prefix = "../" * depth
    source = dataset.get("image_sources", {}).get(image_id, {})
    if source.get("kind") == "who":
        pdf = esc(source["pdf_url"]) + "#page=1"
        rights = prefix + ("licenses-en.html" if english else "licenses.html")
        links = f'<a class="text-link" href="{pdf}">{"Official WHO PDF, page 1" if english else "WHO 官方 PDF 第 1 頁"}</a> <a class="text-link" href="{rights}">{"Attribution and translation terms" if english else "出處與譯文使用條件"}</a>'
    elif source.get("relative_path"):
        href = prefix + "materials/" + esc(source["relative_path"])
        links = f'<a class="text-link" href="{href}">{"Open the tested PNG" if english else "開啟測試原圖（PNG）"}</a>'
    else:
        return ""
    return f'<div class="source-links">{links}</div>'


def home_body_en(dataset: dict[str, Any]) -> str:
    return home_body(dataset, "en")


def detail_example_card(example: dict[str, Any], index: int) -> str:
    return f"""
    <article class="example-card">
      <div class="example-header"><span class="case-label">範例 {index:02d} · {esc(example['material_id'])}</span><span class="pill">{display_number(example['panel_mean'], 2)} 分</span></div>
      <div class="example-source"><div class="label">英文原文（精確摘錄）</div><p>{esc(example['source_en'])}</p></div>
      <div class="example-arrow" aria-hidden="true">↓</div>
      <div class="example-target"><div class="label">模型繁中輸出</div><p>{esc(example['target_zh_tw'])}</p><small>翻譯時間 {display_number(example['translation_elapsed_seconds'], 2)} 秒 · {esc(example['translation_status'])}</small></div>
    </article>
    """


def curated_example_card(example: dict[str, Any], index: int, lang: str = "zh") -> str:
    english = lang == "en"
    kind = "Better example" if english and example.get("kind") == "better" else "Risk example" if english and example.get("kind") == "risk" else "Curated example" if english else "較佳案例" if example.get("kind") == "better" else "風險案例" if example.get("kind") == "risk" else "精選案例"
    source_label = "English source excerpt" if english else "英文原文摘錄"
    target_label = "Traditional Chinese output excerpt" if english else "繁中輸出摘錄"
    reason_label = "Why it matters" if english else "為什麼值得看"
    evidence = example.get("why_en" if english else "why_zh") or ("A reviewer raised a question that needs human review." if english else "評審提出的疑義，需人工複核")
    evidence_html = f"<div class=\"curated-evidence\"><div class=\"label\">{reason_label}</div><p>{esc(evidence)}</p></div>"
    score = display_number(example.get("panel_mean"), 2)
    score_html = f'<span class="pill">{score} score</span>' if english and score != "—" else f'<span class="pill">{score} 分</span>' if score != "—" else ""
    return f"""
    <article class="example-card curated-example"><div class="example-header"><span class="case-label">{kind} {index:02d} · {esc(example.get('material_id'))}</span>{score_html}</div><div class="example-source"><div class="label">{source_label}</div><p>{esc(example.get('source_en'))}</p></div><div class="example-arrow" aria-hidden="true">↓</div><div class="example-target"><div class="label">{target_label}</div><p>{esc(example.get('target_zh_tw'))}</p></div>{evidence_html}</article>
    """


def model_detail_body(dataset: dict[str, Any], model_id: str) -> str:
    model = next((m for m in dataset["models"] if m["model_id"] == model_id), None)
    if model is None:
        model = dataset["models"][0]
        model_id = model["model_id"]
    rows = [r for r in dataset["rows"] if r["model_id"] == model_id]
    examples = [r for r in rows if r["source_kind"] == "text" and r["common_case_eligible"]][:2]
    curated = dataset.get("curated_examples", {}).get(model_id, [])
    image_example = next((r for r in rows if r["source_kind"] == "image" and r["material_id"] == "img-04"), None) or next((r for r in rows if r["source_kind"] == "image"), None)
    detail_rows = "\n".join(
        f"<tr><td>{esc(r['material_id'])}</td><td>{'文字' if r['source_kind'] == 'text' else '圖片'}</td><td>{display_number(r['panel_mean'])}</td><td>{display_number(r['translation_elapsed_seconds'])}</td><td>{display_integer(r['panel_n_valid'])}/{display_integer(r['panel_required_k'])}</td></tr>"
        for r in rows
    )
    other_models = " ".join(
        f'<a class="model-chip{(" active" if m["model_id"] == model_id else "")}" href="{esc(m["model_id"])}.html">{esc(m["model_name"])}</a>'
        for m in dataset["models"]
    ) + ' <a class="model-chip compare-chip" href="../index.html#overview-chart">回到共用比較</a>'
    image_block = ""
    if image_example:
        image_block = f"""
        <article id="img-04-example" class="image-example-card"><div class="case-label">圖片案例 {esc(image_example['material_id'])}</div><h3>圖片案例的模型譯文</h3><p>{esc(image_example['target_zh_tw'])}</p><div class="mini-facts"><span>4 題圖片模型平均 <b>{display_number(model['image_score'])}</b></span><span>回應時間中位數 <b>{display_number(model['image_warm_median'])} 秒</b></span></div><small>圖片分數不是 OCR 字元準確率；原圖與來源連結列在下方。</small>{image_source_links(dataset, image_example['material_id'])}</article>
        """
    examples_html = "\n".join(curated_example_card(example, i + 1) for i, example in enumerate(curated[:3])) if curated else "\n".join(detail_example_card(example, i + 1) for i, example in enumerate(examples))
    return f"""
    <section class="section-shell page-hero compact">
      <div class="eyebrow">模型實測摘要 · {esc(model_id)}</div><h1>{esc(model['model_name'])}</h1><p>本頁列出這個模型在固定素材與推論設定下的分數、回應時間、檔案資訊，以及可逐筆核對的模型譯文。</p>
      <div class="model-chips" aria-label="切換模型">{other_models}</div><p class="model-compare-entry"><a class="text-link" href="../index.html#overview-chart">在分數與回應時間查看七款模型 →</a></p>
    </section>

    <section class="section-shell metrics-row">
      {metric_card("文字共同題目模型平均分", display_number(model['text_score']), f"{model['text_items']} 題；每題先取評審平均，題目分數中位數 {display_number(model['text_median'])}", "teal")}
      {metric_card("圖片共同題目模型平均分", display_number(model['image_score']), f"{model['image_items']} 張；每題先取評審平均，題目分數中位數 {display_number(model['image_median'])}", "coral")}
      {metric_card("文字回應時間中位數", f"{display_number(model['text_warm_median'])} 秒", f"樣本 n={model['text_cases']}；p95 {display_number(model['text_p95'])} 秒", "purple")}
      {metric_card("圖片回應時間中位數", f"{display_number(model['image_warm_median'])} 秒", f"樣本 n={model['image_cases']}；p95 {display_number(model['image_p95'])} 秒", "amber")}
    </section>
    {runtime_specs_html(model, dataset)}

    <section class="section-shell split-panel model-reading">
      <div><div class="eyebrow">分數、時間與樣本</div><h2>如何解讀這個模型的實測數字</h2><p>文字模型平均分使用 {model['text_items']} 個完整共同題目；每題先平均 5 位 AI 評審，再對題目平均。圖片模型平均分使用 {model['image_items']} 張圖片，每題先平均 4 位 AI 評審。這些是描述性觀察，不是人工真值準確率；不完整評審欄位不補成 0。</p><p>回應時間中位數是固定翻譯測試中、模型已載入後的回應時間；它描述這次設定的樣本，不代表其他硬體一定有相同速度。p95 與樣本數列在上方。</p></div>
      <div class="accent-note"><strong>評審診斷欄位</strong><span>文字評審警示欄位：{model['text_critical']}</span><span>文字共同題目排除：{model['text_excluded']}</span><small>警示與排除是獨立診斷，不能直接換算成臨床風險。</small></div>
    </section>

    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">逐案例輸出</div><h2>原文、模型譯文與評審註記</h2><p class="section-intro">下列內容直接取自公開 allow-list 結果；精選案例的註記說明為什麼值得人工回看，沒有重新生成或臆造。</p></div><a class="text-link" href="../results.html">看全部 238 筆 →</a></div><div class="examples-grid">{examples_html}</div></section>

    <section class="section-shell">{image_block}</section>

    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">逐案例摘要</div><h2>所有翻譯案例的分數與時間</h2></div></div><div class="table-wrap"><table><thead><tr><th>案例 ID</th><th>類型</th><th>題內評審平均</th><th>翻譯時間（秒）</th><th>有效評審 / 要求</th></tr></thead><tbody>{detail_rows}</tbody></table></div></section>

    <section class="section-shell callout caution"><div class="callout-icon">!</div><div><strong>分析限制</strong><p>翻譯樣本是 30 篇文字與 4 張圖片；共同文字平均實際使用 29 題，因為 1 題的評審欄位無效。5 位文字、4 位圖片 AI 評審不是人工金標，圖片分數也不是逐字 OCR 錯誤率。模型頁提供檢查用資料，不構成通用排名或臨床安全結論。</p></div></section>
    """


def detail_example_card_en(example: dict[str, Any], index: int) -> str:
    return f"""
    <article class="example-card"><div class="example-header"><span class="case-label">Example {index:02d} · {esc(example['material_id'])}</span><span class="pill">{display_number(example['panel_mean'], 2)} score</span></div><div class="example-source"><div class="label">English source (exact excerpt)</div><p>{esc(example['source_en'])}</p></div><div class="example-arrow" aria-hidden="true">↓</div><div class="example-target"><div class="label">Model output in Traditional Chinese</div><p>{esc(example['target_zh_tw'])}</p><small>translation time {display_number(example['translation_elapsed_seconds'], 2)} s · {esc(example['translation_status'])}</small></div></article>
    """


def model_detail_body_en(dataset: dict[str, Any], model_id: str) -> str:
    model = next((m for m in dataset["models"] if m["model_id"] == model_id), None)
    if model is None:
        model = dataset["models"][0]
        model_id = model["model_id"]
    rows = [r for r in dataset["rows"] if r["model_id"] == model_id]
    examples = [r for r in rows if r["source_kind"] == "text" and r["common_case_eligible"]][:2]
    curated = dataset.get("curated_examples", {}).get(model_id, [])
    image_example = next((r for r in rows if r["source_kind"] == "image" and r["material_id"] == "img-04"), None) or next((r for r in rows if r["source_kind"] == "image"), None)
    detail_rows = "\n".join(f"<tr><td>{esc(r['material_id'])}</td><td>{'Text' if r['source_kind'] == 'text' else 'Image'}</td><td>{display_number(r['panel_mean'], 2)}</td><td>{display_number(r['translation_elapsed_seconds'], 2)}</td><td>{display_integer(r['panel_n_valid'])}/{display_integer(r['panel_required_k'])}</td></tr>" for r in rows)
    other_models = " ".join(f'<a class="model-chip{(" active" if m["model_id"] == model_id else "")}" href="{esc(m["model_id"])}-en.html">{esc(m["model_name"])}</a>' for m in dataset["models"]) + ' <a class="model-chip compare-chip" href="../index-en.html#overview-chart">Back to shared comparison</a>'
    image_block = ""
    if image_example:
        image_block = f"<article id=\"img-04-example\" class=\"image-example-card\"><div class=\"case-label\">Image case {esc(image_example['material_id'])}</div><h3>Model output for the image case</h3><p>{esc(image_example['target_zh_tw'])}</p><div class=\"mini-facts\"><span>Four-image model mean <b>{display_number(model['image_score'], 2)}</b></span><span>Warm-response median <b>{display_number(model['image_warm_median'], 2)} s</b></span></div><small>Image scores are not OCR character accuracy; the frozen image and source links are shown below.</small>{image_source_links(dataset, image_example['material_id'], 'en')}</article>"
    examples_html = "\n".join(curated_example_card(example, i + 1, "en") for i, example in enumerate(curated[:3])) if curated else "\n".join(detail_example_card_en(example, i + 1) for i, example in enumerate(examples))
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">Measured model summary · {esc(model_id)}</div><h1>{esc(model['model_name'])}</h1><p>This page lists the model's scores, response times, measured files, and case-level outputs under the fixed materials and inference settings.</p><div class="model-chips" aria-label="Switch model">{other_models}</div><p class="model-compare-entry"><a class="text-link" href="../index-en.html#overview-chart">Compare all seven models in one tool →</a></p></section>
    <section class="section-shell metrics-row">{metric_card("Text shared-case model mean", display_number(model['text_score'], 2), f"{model['text_items']} complete cases; reviewer scores averaged per case; median case score {display_number(model['text_median'], 1)}", "teal")}{metric_card("Image shared-case model mean", display_number(model['image_score'], 2), f"{model['image_items']} images; reviewer scores averaged per case; median case score {display_number(model['image_median'], 1)}", "coral")}{metric_card("Text warm-response median", f"{display_number(model['text_warm_median'], 2)} s", f"sample n={model['text_cases']}; p95 {display_number(model['text_p95'], 2)} s", "purple")}{metric_card("Image warm-response median", f"{display_number(model['image_warm_median'], 2)} s", f"sample n={model['image_cases']}; p95 {display_number(model['image_p95'], 2)} s", "amber")}</section>
    {runtime_specs_html(model, dataset, "en")}
    <section class="section-shell split-panel model-reading"><div><div class="eyebrow">Scores, time, and sample size</div><h2>How to read this model's measured numbers</h2><p>The text model mean uses {model['text_items']} complete shared cases: five AI reviewer scores are averaged within each case, then the case means are averaged. The image model mean uses {model['image_items']} images with four reviewers per case. These are descriptive observations, not a human gold accuracy score; incomplete reviewer cells are not filled with zeros.</p><p>Warm-response medians measure response time after the model was loaded in the frozen translation run. They describe this sample and configuration, not the speed of every machine. P95 and sample size are shown above.</p></div><div class="accent-note"><strong>Reviewer diagnostics</strong><span>Text critical flag cells: {model['text_critical']}</span><span>Text shared-case exclusions: {model['text_excluded']}</span><small>Critical flags and exclusions are separate diagnostics, not clinical-risk estimates.</small></div></section>
    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">Case-level outputs</div><h2>Source, model output, and reviewer notes</h2><p class="section-intro">These fields come directly from the public allow-listed results. Curated notes explain why a case merits a human check; nothing was regenerated or invented for this page.</p></div><a class="text-link" href="../results-en.html">See all 238 →</a></div><div class="examples-grid">{examples_html}</div></section>
    <section class="section-shell">{image_block}</section>
    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">Case-level summary</div><h2>Scores and translation time for every case</h2></div></div><div class="table-wrap"><table><thead><tr><th>Case ID</th><th>Type</th><th>Within-case reviewer mean</th><th>Translation time (s)</th><th>Valid reviewers / required</th></tr></thead><tbody>{detail_rows}</tbody></table></div></section>
    <section class="section-shell callout caution"><div class="callout-icon">!</div><div><strong>Analysis limits</strong><p>The translation sample has 30 text cases and 4 images; text model means use 29 shared cases because one review cell was invalid. Five text reviewers and four image reviewers are not a human gold standard, and image scores are not character-level OCR error rates. This model page supports inspection; it is not a universal ranking or a clinical-safety conclusion.</p></div></section>
    """


def result_row_html(row: dict[str, Any]) -> str:
    source_kind = row["source_kind"]
    track_label = "文字" if source_kind == "text" else "圖片"
    source = row["source_en"] if source_kind == "text" else f"圖片案例 {row['material_id']} · {row['image_role'] or 'synthetic layout'}"
    target = row["target_zh_tw"] or "（沒有公開輸出文字）"
    search = " ".join(
        str(row.get(key, "")) for key in ("material_id", "source_id", "model_name", "source_en", "target_zh_tw", "image_description_en")
    )
    score_details = ", ".join(
        f"{JUDGE_LABELS.get(f'judge-0{i}', f'judge-0{i}')}：{display_number(score)}"
        for i, score in enumerate(row["judge_scores"], 1)
        if score is not None
    )
    excluded = " excluded-row" if source_kind == "text" and not row["common_case_eligible"] else ""
    common_note = "共同案例" if row["common_case_eligible"] else "排除於共同案例"
    return f"""
    <tr class="result-row{excluded}" data-track="{esc(source_kind)}" data-model="{esc(row['model_id'])}" data-search="{esc(search)}">
      <td class="case-cell"><strong>{esc(row['material_id'])}</strong><small>{esc(row['source_id'])}</small></td>
      <td><span class="track-badge {('image' if source_kind == 'image' else 'text')}">{track_label}</span></td>
      <td class="source-cell"><details><summary>{esc(source[:120])}{'…' if len(source) > 120 else ''}</summary><p>{esc(source)}</p></details></td>
      <td class="target-cell"><details open><summary>{esc(target[:120])}{'…' if len(target) > 120 else ''}</summary><p>{esc(target)}</p></details></td>
      <td class="score-cell"><strong>{display_number(row['panel_mean'])}</strong><small>{esc(common_note)} · {display_integer(row['panel_n_valid'])}/{display_integer(row['panel_required_k'])}</small><details><summary>AI 評審分數</summary><p>{esc(score_details) or '無可用 AI 評審分數'}</p></details></td>
      <td class="time-cell">{display_number(row['translation_elapsed_seconds'])}<small>秒</small></td>
    </tr>
    """


def results_body(dataset: dict[str, Any]) -> str:
    rows_html = "\n".join(result_row_html(row) for row in dataset["rows"])
    model_options = "".join(f'<option value="{esc(m["model_id"])}">{esc(m["model_name"])}</option>' for m in dataset["models"])
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">可搜尋資料頁</div><h1>逐案例翻譯結果與評審摘要</h1><p>用下方欄位依案例 ID、模型、英文原文或繁中輸出篩選。每列保留可公開的原文、模型輸出、評審摘要與翻譯時間；原始 JSONL、路徑與 logs 不在公開站內。</p></section>
    <section class="section-shell results-toolbar" aria-label="結果篩選器"><label class="search-field"><span>搜尋</span><input id="result-search" type="search" placeholder="例如：text-28、PTSD、Gemma 26B" autocomplete="off"></label><label><span>資料類型</span><select id="result-track"><option value="all">全部</option><option value="text">文字</option><option value="image">圖片</option></select></label><label><span>模型</span><select id="result-model"><option value="all">全部模型</option>{model_options}</select></label><div id="result-count" class="result-count" data-template="顯示 {{visible}} / {{total}} 筆" aria-live="polite">顯示 {len(dataset['rows'])} / {len(dataset['rows'])} 筆</div></section>
    <section class="section-shell table-section results-table-section"><div class="table-wrap"><table id="results-table"><caption class="sr-only">七個模型的 238 筆逐案例翻譯結果</caption><thead><tr><th>案例 ID</th><th>類型</th><th>英文輸入／圖片</th><th>模型繁中輸出</th><th>評審摘要</th><th>翻譯時間（秒）</th></tr></thead><tbody>{rows_html}</tbody></table></div></section>
    <section class="section-shell callout"><div class="callout-icon">↓</div><div><strong>公開資料格式</strong><p>相同的安全 allow-list 也提供為 <a class="text-link" href="data/translation_results.json">translation_results.json</a>，方便重建篩選與檢查。JSON 不含 raw path、secret、PID 或未整理的 AI 評審回覆。</p></div></section>
    """


def result_row_html_en(row: dict[str, Any]) -> str:
    source_kind = row["source_kind"]
    track_label = "Text" if source_kind == "text" else "Image"
    source = row["source_en"] if source_kind == "text" else f"Image case {row['material_id']} · {row['image_role'] or 'synthetic layout'}"
    target = row["target_zh_tw"] or "(no public output text)"
    search = " ".join(str(row.get(key, "")) for key in ("material_id", "source_id", "model_name", "source_en", "target_zh_tw", "image_description_en"))
    score_details = ", ".join(
        f"{JUDGE_LABELS.get(f'judge-0{i}', f'judge-0{i}')}:{display_number(score)}"
        for i, score in enumerate(row["judge_scores"], 1)
        if score is not None
    )
    excluded = " excluded-row" if source_kind == "text" and not row["common_case_eligible"] else ""
    common_note = "shared case" if row["common_case_eligible"] else "excluded from shared cases"
    return f"""
    <tr class="result-row{excluded}" data-track="{esc(source_kind)}" data-model="{esc(row['model_id'])}" data-search="{esc(search)}">
      <td class="case-cell"><strong>{esc(row['material_id'])}</strong><small>{esc(row['source_id'])}</small></td>
      <td><span class="track-badge {('image' if source_kind == 'image' else 'text')}">{track_label}</span></td>
      <td class="source-cell"><details><summary>{esc(source[:120])}{'…' if len(source) > 120 else ''}</summary><p>{esc(source)}</p></details></td>
      <td class="target-cell"><details open><summary>{esc(target[:120])}{'…' if len(target) > 120 else ''}</summary><p>{esc(target)}</p></details></td>
      <td class="score-cell"><strong>{display_number(row['panel_mean'])}</strong><small>{esc(common_note)} · {display_integer(row['panel_n_valid'])}/{display_integer(row['panel_required_k'])}</small><details><summary>Reviewer scores</summary><p>{esc(score_details) or 'No reviewer score available'}</p></details></td>
      <td class="time-cell">{display_number(row['translation_elapsed_seconds'])}<small>s</small></td>
    </tr>
    """


def results_body_en(dataset: dict[str, Any]) -> str:
    rows_html = "\n".join(result_row_html_en(row) for row in dataset["rows"])
    model_options = "".join(f'<option value="{esc(m["model_id"])}">{esc(m["model_name"])}</option>' for m in dataset["models"])
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">Searchable data page</div><h1>Case-level translation results and reviewer summaries</h1><p>Use the controls below to filter by case ID, model, English source, or Traditional Chinese output. Each row keeps the public source, model output, reviewer summary, and translation time; raw JSONL, paths, and logs are excluded.</p></section>
    <section class="section-shell results-toolbar" aria-label="Result filters"><label class="search-field"><span>Search</span><input id="result-search" type="search" placeholder="For example: text-28, PTSD, Gemma 26B" autocomplete="off"></label><label><span>Data type</span><select id="result-track"><option value="all">All</option><option value="text">Text</option><option value="image">Image</option></select></label><label><span>Model</span><select id="result-model"><option value="all">All models</option>{model_options}</select></label><div id="result-count" class="result-count" data-template="Showing {{visible}} / {{total}} rows" aria-live="polite">Showing {len(dataset['rows'])} / {len(dataset['rows'])} rows</div></section>
    <section class="section-shell table-section results-table-section"><div class="table-wrap"><table id="results-table"><caption class="sr-only">238 case-level translation results from seven models</caption><thead><tr><th>Case ID</th><th>Type</th><th>English input / image</th><th>Model output (Traditional Chinese)</th><th>Reviewer summary</th><th>Translation time (s)</th></tr></thead><tbody>{rows_html}</tbody></table></div></section>
    <section class="section-shell callout"><div class="callout-icon">↓</div><div><strong>Public data formats</strong><p>The same safe allow-list is available as <a class="text-link" href="data/translation_results.json">translation_results.json</a> and the complete public <a class="text-link" href="data/translation_comparison.csv">translation_comparison.csv</a>. Neither contains raw paths, secrets, PIDs, or unprocessed reviewer responses.</p></div></section>
    """


def methods_body(dataset: dict[str, Any]) -> str:
    judge_rows = "\n".join(
        f"<tr><td>{esc(r.get('judge_slot'))}</td><td>{esc(r.get('judge_label'))}</td><td>{display_integer(r.get('pointwise_scalar_valid'))} / {display_integer(r.get('pointwise_expected'))}</td><td>{display_number(r.get('latency_median_s'), 2)}</td></tr>"
        for r in dataset["judges"]
    )
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">方法與限制</div><h1>測試設計、分數計算與限制</h1><p>本頁說明測試素材、推論設定、評審面板、分數計算與回應時間。共同題目、AI 評審回覆時間、翻譯暖請求時間是不同的測量。</p></section>

    <section class="section-shell method-flow"><article><span>01</span><h2>測試素材與樣本</h2><p>30 篇英文文字、4 張圖片；7 個候選模型各產生一筆翻譯，公開比較共 238 個模型－題目欄位。</p></article><article><span>02</span><h2>推論設定</h2><p>翻譯測試使用固定 context、temperature 0、關閉 thinking，並逐一發出請求；完整設定留在可重建來源。</p></article><article><span>03</span><h2>AI 評審設計</h2><p>文字使用 5 位 AI 評審；圖片使用 4 位固定 AI 評審。這是評審面板觀察，不是人工金標。</p></article></section>

    <section class="section-shell split-panel"><div><div class="eyebrow">共同題目篩選</div><h2>1 個無效評審欄位，不當成 0 分。</h2><p>文字共同題目必須有完整的 5 位 AI 評審結果。text-15 有 1 個評審欄位無效，因此模型平均使用 29 個共同題目；該案例的候選翻譯與有效公開資訊仍保留。</p><p>圖片共同題目為 4 張、每張使用 4 位固定 AI 評審。Mistral 的圖片路徑設計上不支援，因此不補成低分。</p></div><div class="big-number"><strong>29</strong><span>共同文字案例</span><small>12 個來源頁面</small></div></section>

    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">模型分數的計算方式</div><h2>先算每題，再算模型平均</h2></div></div><div class="definition-grid"><div class="definition"><h3>共同題目範圍</h3><p>完整共同題目為文字 29 題、圖片 4 題；原始測試仍有 30 篇文字與 4 張圖片。</p></div><div class="definition"><h3>題內與模型平均</h3><p>每題先取有效 AI 評審分數的算術平均，再對共同題目平均。四項 rubric 原始配分相加成 0–100。</p></div><div class="definition"><h3>不代表</h3><p>不是翻譯正確率、不是臨床安全分數，也不是跨模型通用排行榜。</p></div></div></section>
    {rubric_reliability_section(dataset)}
    {image_sources_panel(dataset)}

    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">AI 評審面板摘要</div><h2>評審回覆時間，與翻譯時間分開</h2><p class="section-intro">下表的 latency 是 AI 評審回覆時間；首頁與模型頁的回應時間只來自翻譯測試。兩者不能混用。</p></div></div><div class="table-wrap"><table><thead><tr><th>欄位</th><th>評審標籤</th><th>有效 / 預期</th><th>評審回覆中位（秒）</th></tr></thead><tbody>{judge_rows}</tbody></table></div></section>

    <section class="section-shell limitations"><div class="eyebrow">分析限制</div><h2>這份資料能回答與不能回答的問題</h2><div class="limit-list"><div><b>樣本小</b><span>原始文字 30 篇、圖片 4 張；圖片最高分不能當作普遍視覺能力。</span></div><div><b>評審不是人工金標</b><span>5 位 AI 評審的一致性不等於翻譯真值或臨床安全。</span></div><div><b>圖片不是逐字辨識</b><span>圖片結果是端到端翻譯品質觀察，不是逐字 OCR CER。</span></div><div><b>速度不可外推</b><span>翻譯時間中位數受硬體、runtime、設定和請求內容影響。</span></div><div><b>不做完整勝率</b><span>兩兩比較不是完整 round-robin，不從不完整比較圖推導排名。</span></div><div><b>原始檔不公開</b><span>網站排除 raw attempt、絕對路徑、PID、private binding 與未整理 logs。</span></div></div></section>

    <section class="section-shell reproducibility"><div><div class="eyebrow">重建方式與公開範圍</div><h2>標準庫重建，欄位先經 allow-list。</h2><p>網站由 <code>build_site.py</code> 使用 Python 標準庫重建，預設讀取已發布的公開 CSV，先套用 allow-list，再寫入 docs。若結果或方法檔更新，請重新核對公開欄位與授權。</p></div><pre><code>python build_site.py
python build_site.py --results &lt;latest_results&gt; --materials &lt;materials&gt;</code></pre></section>
    """


def downloads_body() -> str:
    return '''<section class="section-shell editorial-section"><h1>程式碼、完整翻譯與測試素材</h1><p>可下載 238 筆翻譯、評審紀錄，以及重建本測試所用的程式與素材。</p><ul><li><a href="downloads/complete-results.zip">完整結果（ZIP）</a></li><li><a href="downloads/benchmark-code.zip">測試與分析程式（ZIP）</a></li><li><a href="downloads/materials.zip">翻譯素材（ZIP）</a> · <a href="licenses.html">素材使用條件</a></li></ul></section>'''


def methods_body_en(dataset: dict[str, Any]) -> str:
    judge_rows = "\n".join(
        f"<tr><td>{esc(r.get('judge_slot'))}</td><td>{esc(r.get('judge_label'))}</td><td>{display_integer(r.get('pointwise_scalar_valid'))} / {display_integer(r.get('pointwise_expected'))}</td><td>{display_number(r.get('latency_median_s'), 2)}</td></tr>"
        for r in dataset["judges"]
    )
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">Methods and limits</div><h1>Test design, score calculation, and limits</h1><p>This page documents the materials, inference settings, reviewer panels, score calculation, and response-time measures. Shared cases, reviewer-response latency, and translation warm-response time are different measurements.</p></section>
    <section class="section-shell method-flow"><article><span>01</span><h2>Materials and sample</h2><p>30 English text cases and 4 images; each of seven candidate models produced one translation per case, for 238 public model-case cells.</p></article><article><span>02</span><h2>Inference settings</h2><p>The translation run used fixed context, temperature 0, thinking off, and one request at a time. Full settings remain in the reproducible source.</p></article><article><span>03</span><h2>AI reviewer design</h2><p>Text used five reviewers; images used four fixed reviewers. This is an observed panel, not a human gold standard.</p></article></section>
    <section class="section-shell split-panel"><div><div class="eyebrow">Shared-case inclusion</div><h2>One invalid reviewer cell is excluded, not scored as zero.</h2><p>A text shared-case item requires a complete five-reviewer panel. text-15 had one invalid reviewer cell, so model means use 29 shared items. The candidate translation and valid public information remain available.</p><p>The image shared set contains four images and four fixed reviewers per image. Mistral's image route was unsupported by design, so it was not filled with a low score.</p></div><div class="big-number"><strong>29</strong><span>shared text cases</span><small>12 source pages</small></div></section>
    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">How model scores are calculated</div><h2>Average within each case, then across cases</h2></div></div><div class="definition-grid"><div class="definition"><h3>Shared-case set</h3><p>Complete shared cases: 29 text items and 4 image items; the original text sample still contains 30 cases.</p></div><div class="definition"><h3>Within-case and model mean</h3><p>Valid reviewer scores are averaged within each case, then case means are averaged. The four rubric dimensions sum to 0–100.</p></div><div class="definition"><h3>Not a claim of</h3><p>Translation accuracy, clinical safety, or a universal cross-model leaderboard.</p></div></div></section>
    {rubric_reliability_section(dataset, "en")}
    {image_sources_panel(dataset, "en")}
    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">AI reviewer panel summary</div><h2>Reviewer response latency is separate from translation time</h2><p class="section-intro">The latency below is reviewer-response time. Homepage and model-page warm-response seconds come from translation generation; do not combine the two measures.</p></div></div><div class="table-wrap"><table><thead><tr><th>Slot</th><th>Reviewer label</th><th>Valid / expected</th><th>Reviewer response median (s)</th></tr></thead><tbody>{judge_rows}</tbody></table></div></section>
    <section class="section-shell limitations"><div class="eyebrow">Analysis limits</div><h2>What this dataset can and cannot answer</h2><div class="limit-list"><div><b>Small sample</b><span>30 text cases and 4 images; a top image score is not a general visual claim.</span></div><div><b>Panel is not gold</b><span>Reviewer agreement does not establish translation truth or clinical safety.</span></div><div><b>Images are not OCR</b><span>Image results are end-to-end translation observations, not character-level OCR CER.</span></div><div><b>Speed is local</b><span>Generation medians depend on hardware, runtime, settings, and request content.</span></div><div><b>No full win rate</b><span>Pairwise comparisons are not a complete round-robin, so no full ranking is inferred.</span></div><div><b>Raw files stay private</b><span>The site excludes raw attempts, absolute paths, PIDs, private bindings, and unprocessed logs.</span></div></div></section>
    <section class="section-shell reproducibility"><div><div class="eyebrow">Rebuild process and public scope</div><h2>Standard-library rebuild with an explicit allow-list</h2><p>The site is rebuilt by <code>build_site.py</code> with Python's standard library. It reads the published public CSV snapshot by default, applies an explicit allow-list, and writes docs. If result or method files change, re-check public fields and licensing.</p></div><pre><code>python build_site.py
python build_site.py --results &lt;latest_results&gt; --materials &lt;materials&gt;</code></pre></section>
    """


def downloads_body_en() -> str:
    return '''<section class="section-shell editorial-section"><h1>Code, translations and source materials</h1><p>Download the 238 translations, judge records, test materials and code used for this run.</p><ul><li><a href="downloads/complete-results.zip">Complete results (ZIP)</a></li><li><a href="downloads/benchmark-code.zip">Benchmark and analysis code (ZIP)</a></li><li><a href="downloads/materials.zip">Test materials (ZIP)</a> · <a href="licenses-en.html">Terms of use</a></li></ul></section>'''


def licenses_body_en() -> str:
    return '''<section class="section-shell editorial-section"><h1>Source attribution and terms of use</h1><p>Each material is subject to its source publisher’s terms. The rights file lists attribution requirements, reuse conditions and the notice required for translations derived from WHO materials.</p><p><a href="materials/RIGHTS.md">Material rights</a> · <a href="materials/source_attribution.json">Sources by material</a></p></section>'''


def licenses_body() -> str:
    return '''<section class="section-shell editorial-section"><h1>素材出處與使用條件</h1><p>每項素材的授權依原出版者與來源而異。權利說明列出署名方式、使用條件，以及 WHO 衍生譯文所需的聲明。</p><p><a href="materials/RIGHTS.md">完整授權說明</a> · <a href="materials/source_attribution.json">素材與來源對照</a></p></section>'''


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def load_site_asset(name: str) -> str:
    return (Path(__file__).resolve().parent / "scripts" / name).read_text(encoding="utf-8")


def apply_editorial_sections(page: str, page_name: str, sections: dict[str, Any]) -> str:
    """Apply optional publisher-reviewed HTML snippets without copying sources.

    ``docs/data/editorial_sections.json`` is intentionally an opt-in extension
    point for the parent release audit.  Values are trusted, reviewed HTML
    fragments.  For downloads/licenses a supplied fragment replaces the
    placeholder body; for other pages it is appended immediately before
    ``</main>`` so the generated page remains useful when the file is absent.
    """

    value = sections.get(page_name)
    if not isinstance(value, str) or not value.strip():
        return page
    if page_name in {"downloads.html", "downloads-en.html", "licenses.html", "licenses-en.html"}:
        start_marker = '<main id="main">'
        start = page.find(start_marker)
        end = page.find("</main>", start + len(start_marker)) if start >= 0 else -1
        if start >= 0 and end >= 0:
            return page[: start + len(start_marker)] + "\n" + value + "\n" + page[end:]
    return page.replace("</main>", "\n" + value + "\n</main>", 1)


def load_google_verification(output_dir: Path) -> str:
    """Read an optional Search Console token without inventing one."""

    path = output_dir / "google-site-verification-meta.txt"
    if not path.is_file():
        return ""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not value:
        return ""
    match = re.search(r"content\s*=\s*[\"']([^\"']+)[\"']", value, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip()
    if re.fullmatch(r"[A-Za-z0-9._-]{8,256}", value):
        return value
    return ""


def write_discovery_files(docs: Path) -> None:
    """Write public canonical discovery files for GitHub Pages."""

    paths = [
        "index.html",
        "index-en.html",
        "methods.html",
        "methods-en.html",
        "results.html",
        "results-en.html",
        "downloads.html",
        "downloads-en.html",
        "licenses.html",
        "licenses-en.html",
    ] + [f"models/{model_id}{suffix}.html" for model_id in MODEL_ORDER for suffix in ("", "-en")]
    urls = [f"{PUBLIC_SITE_BASE}{path}" for path in paths]
    body = "".join(f"  <url><loc>{html.escape(url, quote=True)}</loc></url>\n" for url in urls)
    write_text(
        docs / "sitemap.xml",
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + body
        + "</urlset>\n",
    )
    write_text(
        docs / "robots.txt",
        f"User-agent: *\nAllow: /\nSitemap: {PUBLIC_SITE_BASE}sitemap.xml\n",
    )


SITE_JS = r"""
(function () {
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('#site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(open));
    });
  }

  const search = document.querySelector('#result-search');
  const track = document.querySelector('#result-track');
  const model = document.querySelector('#result-model');
  const count = document.querySelector('#result-count');
  const rows = Array.from(document.querySelectorAll('.result-row'));
  if (!search || !track || !model || !count || !rows.length) return;

  function applyFilters() {
    const q = search.value.trim().toLocaleLowerCase();
    const trackValue = track.value;
    const modelValue = model.value;
    let visible = 0;
    rows.forEach(function (row) {
      const matchesQuery = !q || (row.dataset.search || '').toLocaleLowerCase().includes(q);
      const matchesTrack = trackValue === 'all' || row.dataset.track === trackValue;
      const matchesModel = modelValue === 'all' || row.dataset.model === modelValue;
      const show = matchesQuery && matchesTrack && matchesModel;
      row.hidden = !show;
      if (show) visible += 1;
    });
    const template = count.dataset.template || '顯示 {visible} / {total} 筆';
    count.textContent = template.replace('{visible}', String(visible)).replace('{total}', String(rows.length));
  }
  search.addEventListener('input', applyFilters);
  track.addEventListener('change', applyFilters);
  model.addEventListener('change', applyFilters);
}());
"""


def build_site(results_dir: Path, materials_dir: Path, output_dir: Path) -> dict[str, Any]:
    dataset = load_dataset(results_dir)
    dataset["image_sources"] = load_image_sources(materials_dir)
    docs = output_dir
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "models").mkdir(parents=True, exist_ok=True)
    (docs / "assets").mkdir(parents=True, exist_ok=True)
    (docs / "data").mkdir(parents=True, exist_ok=True)

    editorial_sections: dict[str, Any] = {}
    editorial_path = docs / "data" / "editorial_sections.json"
    if not editorial_path.is_file():
        editorial_path = results_dir / "editorial_sections.json"
    if editorial_path.is_file():
        loaded_editorial = read_json(editorial_path)
        if isinstance(loaded_editorial, dict):
            editorial_sections = loaded_editorial
    verification_meta = load_google_verification(docs)

    def write_page(relative_path: str, rendered: str) -> None:
        key = relative_path.replace("\\", "/")
        write_text(docs / relative_path, apply_editorial_sections(rendered, key, editorial_sections))

    write_text(docs / "assets" / "style.css", load_site_asset("report.css").strip() + "\n")
    write_text(docs / "assets" / "site.js", SITE_JS.strip() + "\n")
    write_text(docs / "assets" / "report_ui.js", load_site_asset("report_ui.js").strip() + "\n")

    # Machine-readable public data is intentionally a small, explicit projection.
    public_manifest = {
        "schema_version": "1.0",
        "title": "心理健康翻譯基準：安全 allow-list 結果",
        "generated_by": "build_site.py",
        "rows": dataset["rows"],
    }
    write_text(docs / "data" / "translation_results.json", json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(docs / "data" / "criterion_scores.json", json.dumps(dataset["criterion_scores"], ensure_ascii=False, indent=2) + "\n")

    write_page("index.html", page_html("七款地端模型實測 · 心理健康英中翻譯", "home", home_body(dataset), self_href="index.html", switch_href="index-en.html", verification_meta=verification_meta))
    write_page("results.html", page_html("238 筆翻譯", "results", results_body(dataset), self_href="results.html", switch_href="results-en.html"))
    write_page("methods.html", page_html("方法與限制", "methods", methods_body(dataset), self_href="methods.html", switch_href="methods-en.html"))
    write_page("downloads.html", page_html("下載", "downloads", downloads_body(), self_href="downloads.html", switch_href="downloads-en.html"))
    write_page("licenses.html", page_html("授權與來源", "licenses", licenses_body(), self_href="licenses.html", switch_href="licenses-en.html"))

    for model in dataset["models"]:
        model_id = str(model["model_id"])
        write_page(
            f"models/{model_id}.html",
            page_html(model["model_name"], "models", model_detail_body(dataset, model_id), depth=1, self_href=f"{model_id}.html", switch_href=f"{model_id}-en.html"),
        )

    # English pages are generated from the same public data projection. Source
    # and target experiment text remains byte-for-byte as supplied in the CSV.
    write_page("index-en.html", page_html("Seven local models tested · Mental health translation into Traditional Chinese", "home", home_body_en(dataset), lang="en", self_href="index-en.html", switch_href="index.html", verification_meta=verification_meta))
    write_page("results-en.html", page_html("238 translations", "results", results_body_en(dataset), lang="en", self_href="results-en.html", switch_href="results.html"))
    write_page("methods-en.html", page_html("Methods & limits", "methods", methods_body_en(dataset), lang="en", self_href="methods-en.html", switch_href="methods.html"))
    write_page("downloads-en.html", page_html("Downloads", "downloads", downloads_body_en(), lang="en", self_href="downloads-en.html", switch_href="downloads.html"))
    write_page("licenses-en.html", page_html("Licensing & sources", "licenses", licenses_body_en(), lang="en", self_href="licenses-en.html", switch_href="licenses.html"))
    for model in dataset["models"]:
        model_id = str(model["model_id"])
        write_page(
            f"models/{model_id}-en.html",
            page_html(model["model_name"], "models", model_detail_body_en(dataset, model_id), depth=1, lang="en", self_href=f"{model_id}-en.html", switch_href=f"{model_id}.html"),
        )

    # GitHub Pages serves this repository as a static tree; .nojekyll avoids
    # accidental processing of generated JSON names.
    write_text(docs / ".nojekyll", "")
    write_discovery_files(docs)
    return dataset


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=here / "docs" / "data",
        help="Directory containing public/frozen result CSVs. The committed docs/data snapshot is the default; private review data requires an explicit path.",
    )
    parser.add_argument(
        "--materials",
        type=Path,
        default=here / "docs" / "materials",
        help="Public materials directory (reserved for manifest checks; never copied). Private materials require an explicit path.",
    )
    parser.add_argument("--out", type=Path, default=here / "docs", help="Static output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = [
        "model_summary.csv",
        "common_case_model_summary.csv",
        "translation_generation_summary.csv",
        "judge_summary.csv",
        "translation_comparison.csv",
    ]
    missing = [name for name in required if not (args.results / name).is_file()]
    if missing:
        raise SystemExit("Missing required result files: " + ", ".join(missing))
    dataset = build_site(args.results, args.materials, args.out)
    print(f"Built {len(dataset['rows'])} public translation rows and {len(dataset['models'])} model pages at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
