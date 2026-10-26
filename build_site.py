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


def display_number(value: Any, digits: int = 3, missing: str = "—") -> str:
    number = as_float(value)
    if number is None:
        return missing
    return f"{number:.{digits}f}"


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
    config: dict[str, Any] = {}
    inventory_path = results_dir / "model_inventory.json"
    config_path = results_dir / "translation_config.json"
    if inventory_path.is_file():
        loaded = read_json(inventory_path)
        if isinstance(loaded, dict) and isinstance(loaded.get("models"), dict):
            inventory = loaded["models"]
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
    return {"by_model": by_model, "fixed_generation": fixed_public}


def load_image_sources(materials_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the four public image-source records once for source panels."""

    fallback = {
        "img-01": {"relative_path": "images/img01_synthetic_depression_note.png", "kind": "synthetic", "label": "Synthetic evaluation image — not a clinical record", "pdf_url": "", "license_note": "Project-authored synthetic render; not a patient record."},
        "img-04": {"relative_path": "images/img04_synthetic_medication_label.png", "kind": "synthetic", "label": "Synthetic evaluation image — not a prescription", "pdf_url": "", "license_note": "Project-authored synthetic render; not a prescription."},
        "img-09": {"relative_path": "images/img09_original_who_suicide_policy_p1.png", "kind": "who", "label": "WHO PDF page-1 render; reference text is not human gold", "pdf_url": "https://cdn.who.int/media/docs/default-source/mental-health/9789240078796-eng.pdf?download=true", "license_note": "WHO source: CC BY-NC-SA 3.0 IGO; attribution, non-commercial use and ShareAlike apply."},
        "img-10": {"relative_path": "images/img10_original_who_depression_brochure_p1.png", "kind": "who", "label": "WHO PDF page-1 render; reference text is not human gold", "pdf_url": "https://cdn.who.int/media/docs/default-source/searo/mhs/depression-what-you-should-know-what-you-can-do.pdf?sfvrsn=293cb15b_5", "license_note": "WHO source: CC BY-NC-SA 3.0 IGO; attribution, non-commercial use and ShareAlike apply."},
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
            "relative_path": relative_path,
            "kind": kind,
            "label": str(item.get("synthetic_label") or item.get("ground_truth_status") or "Image source record"),
            "pdf_url": source_pdf_url,
            "license_note": "WHO source: CC BY-NC-SA 3.0 IGO; attribution, non-commercial use and ShareAlike apply." if kind == "who" else "Project-authored synthetic render; not a patient record or prescription.",
        }
    return {**fallback, **out}


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
    }


def nav_html(current: str, depth: int = 0, lang: str = "zh") -> str:
    prefix = "../" * depth
    if lang == "en":
        links = [
            ("home", "Home", f"{prefix}index-en.html"),
            ("models", "Models", f"{prefix}models/gemma4-26b-en.html"),
            ("results", "238 translations", f"{prefix}results-en.html"),
            ("methods", "Methods & limits", f"{prefix}methods-en.html"),
            ("downloads", "Downloads", f"{prefix}downloads-en.html"),
            ("licenses", "Licensing & sources", f"{prefix}licenses-en.html"),
        ]
    else:
        links = [
            ("home", "首頁", f"{prefix}index.html"),
            ("models", "模型比較", f"{prefix}models/gemma4-26b.html"),
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
    title = "Psychological Health Translation Benchmark" if english else "心理健康翻譯基準"
    status = "2026-10-25 frozen review · research use" if english else "2026-10-25 凍結檢視版本 · 研究用途"
    methods_label = "Methods & limits" if english else "方法與限制"
    downloads_label = "Downloads" if english else "下載"
    licenses_label = "Licensing & sources" if english else "授權與來源"
    methods_href = f"{prefix}methods-en.html" if english else f"{prefix}methods.html"
    downloads_href = f"{prefix}downloads-en.html" if english else f"{prefix}downloads.html"
    licenses_href = f"{prefix}licenses-en.html" if english else f"{prefix}licenses.html"
    note = (
        "This site provides descriptive, traceable translation observations; it is not medical advice or clinical-safety certification. Data pages contain only explicitly allow-listed public fields."
        if english
        else "本網站提供描述性、可追溯的翻譯觀察；不是醫療建議，也不是臨床安全認證。資料頁只包含明確允許公開的欄位。"
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
        "Seven local models tested on psychological-health English-to-Traditional-Chinese translation, with reviewer scoring, response times and traceable results."
        if english and current == "home"
        else "七款地端模型實測：心理健康英中翻譯品質、模型載入後回應時間、AI 評審方法與可追溯結果。"
        if not english and current == "home"
        else "Seven-model psychological health translation benchmark with descriptive quality observations, generation latency and traceable methods."
        if english
        else "七個模型的繁體中文心理健康翻譯基準：品質觀察、生成速度與可追溯方法。"
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
        brand = '<span class="brand-mark">譯</span><span>Psychological Health<br><em>Translation benchmark 2026</em></span>'
    else:
        zh_href = self_href
        en_href = switch_href
        brand = '<span class="brand-mark">譯</span><span>心理健康翻譯<br><em>Benchmark 2026</em></span>'
    zh_url = absolute_url(zh_href)
    en_url = absolute_url(en_href)
    verification_html = f'\n  <meta name="google-site-verification" content="{esc(verification_meta)}">' if verification_meta else ""
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
  <title>{esc(title if current == 'home' else title + (' · Psychological Health Translation Benchmark' if english else ' · 心理健康翻譯基準'))}</title>
  <link rel="stylesheet" href="{prefix}assets/style.css?v=20261026-2">
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
  <script src="{prefix}assets/site.js" defer></script>
</body>
</html>
"""


def score_bar(label: str, value: float | None, tone: str = "teal") -> str:
    safe_value = 0 if value is None else max(0.0, min(100.0, float(value)))
    value_text = display_number(value)
    return f"""
    <div class="bar-row">
      <div class="bar-label"><span>{esc(label)}</span><strong>{value_text}</strong></div>
      <div class="bar-track" role="img" aria-label="{esc(label)}：{esc(value_text)}（0 到 100 固定刻度）"><span class="bar-fill {esc(tone)}" style="width: {safe_value:.3f}%"></span></div>
    </div>
    """


def metric_card(label: str, value: str, note: str = "", tone: str = "teal") -> str:
    return f'''<div class="metric-card {esc(tone)}"><div class="metric-label">{esc(label)}</div><div class="metric-value">{esc(value)}</div><div class="metric-note">{esc(note)}</div></div>'''


def model_card(model: dict[str, Any]) -> str:
    slug = esc(model["model_id"])
    return f"""
    <article class="model-card">
      <div class="card-kicker">候選模型</div>
      <h3><a href="models/{slug}.html">{esc(model['model_name'])}</a></h3>
      <div class="model-score-line"><span>共同文字題目均分</span><strong>{display_number(model['text_score'])}</strong></div>
      {score_bar("文字", model["text_score"], "teal")}
      <div class="model-score-line image-line"><span>共同圖片題目均分</span><strong>{display_number(model['image_score'])}</strong></div>
      {score_bar("圖片", model["image_score"], "coral")}
      <div class="mini-facts"><span>文字回應時間中位 <b>{display_number(model['text_warm_median'])} 秒</b></span><span>圖片回應時間中位 <b>{display_number(model['image_warm_median'])} 秒</b></span></div>
      <a class="text-link" href="models/{slug}.html">查看模型細頁 <span aria-hidden="true">→</span></a>
    </article>
    """


def quick_compare_table(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    heading = "Seven-model quick comparison" if english else "七個模型快速比較"
    intro = (
        "Scores are the descriptive common-case panel means; times are the median response after the model was loaded."
        if english
        else "分數是共同題目的描述性評審均分；時間是模型載入後的回應時間中位數。"
    )
    headers = ["Model", "Text score", "Image score", "Text time", "Image time"] if english else ["模型", "文字分數", "圖片分數", "文字時間", "圖片時間"]
    rows = "\n".join(
        f"<tr><td><a href=\"{'models/' + esc(m['model_id']) + ('-en.html' if english else '.html')}\">{esc(m['model_name'])}</a></td><td><strong>{display_number(m['text_score'])}</strong></td><td><strong>{display_number(m['image_score'])}</strong></td><td>{display_number(m['text_warm_median'])} s</td><td>{display_number(m['image_warm_median'])} s</td></tr>"
        for m in dataset["models"]
    )
    return f"""
    <section class="section-shell quick-compare-section"><div class="section-heading"><div><div class="eyebrow">{heading}</div><h2>{'一眼看全七款模型。' if not english else 'All seven models at a glance.'}</h2><p class="section-intro">{intro}</p></div></div><div class="table-wrap quick-compare-wrap"><table class="quick-compare"><thead><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr></thead><tbody>{rows}</tbody></table></div></section>
    """


def hero_result_panel(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    best_text = next(m for m in dataset["models"] if m["model_id"] == "gemma4-26b")
    best_image = next(m for m in dataset["models"] if m["model_id"] == "gemma4-31b-qat")
    return f"""
      <div class="hero-result-panel" aria-label="{'Key results' if english else '快速結果'}"><div class="eyebrow">{'Key results' if english else '快速結果'}</div><div class="hero-result-row"><span>{'Text | 26B' if english else '文字｜26B'}</span><strong>{display_number(best_text['text_score'])}</strong></div><div class="hero-result-row"><span>{'Image | 31B QAT' if english else '圖片｜31B QAT'}</span><strong>{display_number(best_image['image_score'])}</strong></div><div class="hero-result-row"><span>{'Fastest shared text | 26B' if english else '共同文字最快｜26B'}</span><strong>{display_number(best_text['text_warm_median'])} s</strong></div><div class="hero-result-note"><a href="#judge-rubric">{'A 98 score is not 98% accuracy; see how it is scored ↓' if english else '98 分不等於 98% 正確；看評分方法 ↓'}</a><span>{'Time is after model loading.' if english else '時間是模型載入後回應。'}</span></div></div>
    """


def reliability_metric(dataset: dict[str, Any], track: str) -> dict[str, Any]:
    metrics = dataset.get("independent_review", {}).get("reliability_metrics", [])
    if not isinstance(metrics, list):
        return {}
    return next((item for item in metrics if isinstance(item, dict) and item.get("track") == track), {})


def judge_explainer(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    text_metric = reliability_metric(dataset, "text")
    text_one = display_number(text_metric.get("icc_2_1_absolute_agreement"))
    text_k = display_number(text_metric.get("icc_2_k_absolute_agreement"))
    if english:
        return f"""
    <section class="section-shell judge-explainer" id="judge-rubric"><div class="judge-explainer-panel"><div class="eyebrow">How the scores are made</div><h2>What does a score in the 90s mean?</h2><p>The score is not just about fluency: 80 points cover terminology, meaning and completeness; 20 cover fluency. Five different AI reviewers — Qwen, Nemotron, Mistral, GPT and Gemini — scored the source and translation without knowing the model name. Each item first averages valid reviewer scores; text then averages the 29 shared cases. Images use four reviewers, excluding Mistral, across four cases.</p><div class="rubric-mini-grid"><div class="rubric-mini-item"><b>30 · Terminology &amp; Taiwan usage</b><span>Terms and Taiwan Traditional Chinese; catches Simplified/non-Taiwan terms.</span></div><div class="rubric-mini-item"><b>30 · Meaning &amp; safety facts</b><span>Negation, medication, dose, numbers and conditions.</span></div><div class="rubric-mini-item"><b>20 · Fluency</b><span>Natural, clear Taiwan Traditional Chinese.</span></div><div class="rubric-mini-item"><b>20 · Completeness &amp; format</b><span>Omissions, additions, format and visible structure.</span></div></div><div class="judge-icc-mini"><span>ICC (intraclass correlation) checks reviewer reliability: one reviewer {text_one}, five-reviewer mean {text_k}; averaging can make scores more stable. This is agreement in scoring, not accuracy or clinical safety.</span><small>The four subscores add to the total; errors do not use a fixed penalty schedule.</small><a class="text-link" href="models/gemma4-26b-en.html#img-04-example">High scores still need medication/dose checks: img-04 →</a><a class="text-link" href="methods-en.html#rubric-icc">Full rubric and ICC notes →</a></div></div></section>
        """
    return f"""
    <section class="section-shell judge-explainer" id="judge-rubric"><div class="judge-explainer-panel"><div class="eyebrow">評審與分數</div><h2>九十幾分，代表什麼？</h2><p>不只看語感：80 分看術語、原意與完整性，20 分看流暢度。五位不同的 AI 評審——Qwen、Nemotron、Mistral、GPT、Gemini——在不知道翻譯模型名稱的情況下，對照原文評分。每筆先取有效評審分數平均，再對 29 個共同文字題目取平均。圖片使用 4 位評審（不含 Mistral），共 4 張。</p><div class="rubric-mini-grid"><div class="rubric-mini-item"><b>30｜術語與台灣用語</b><span>檢查術語誤用、簡體字與不合台灣用語。</span></div><div class="rubric-mini-item"><b>30｜語意與安全事實</b><span>否定、藥名、劑量、數字與條件。</span></div><div class="rubric-mini-item"><b>20｜流暢度</b><span>自然、清楚的台灣繁中。</span></div><div class="rubric-mini-item"><b>20｜完整與格式</b><span>漏譯、增譯、格式與可見結構。</span></div></div><div class="judge-icc-mini"><span>再用 ICC（組內相關係數）檢查評審間信度：單一評審 {text_one}，五位平均 {text_k}，顯示取平均後的給分較穩定。這是給分一致性，不是正確率，也不能證明臨床安全。</span><small>四項分數相加，不是每個錯誤固定扣分。</small><a class="text-link" href="models/gemma4-26b.html#img-04-example">高分仍要檢查藥名、劑量等重大錯誤：看 img-04 →</a><a class="text-link" href="methods.html#rubric-icc">完整 rubric 與 ICC 說明 →</a></div></div></section>
    """


def rubric_reliability_section(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    text_metric = reliability_metric(dataset, "text")
    image_metric = reliability_metric(dataset, "image")
    text_one = display_number(text_metric.get("icc_2_1_absolute_agreement"))
    text_k = display_number(text_metric.get("icc_2_k_absolute_agreement"))
    image_one = display_number(image_metric.get("icc_2_1_absolute_agreement"))
    image_k = display_number(image_metric.get("icc_2_k_absolute_agreement"))
    text_units = display_integer(text_metric.get("complete_balanced_units"))
    image_units = display_integer(image_metric.get("complete_balanced_units"))
    if english:
        return f"""
    <section class="section-shell rubric-detail" id="rubric-icc"><div class="section-heading"><div><div class="eyebrow">Traceable rubric and reliability</div><h2>What each reviewer score means</h2><p class="section-intro">The public instructions define four integer subscores for text and image jobs. They must sum exactly to the 0–100 score; this is not a fixed error-penalty table.</p></div></div><div class="rubric-table-wrap"><table class="rubric-table"><thead><tr><th>Dimension</th><th>Points</th><th>Reviewers inspect</th></tr></thead><tbody><tr><td>Terminology &amp; Taiwan usage</td><td>0–30</td><td>Medical/psychological terms and Taiwan Traditional Chinese, not Simplified Chinese.</td></tr><tr><td>Meaning &amp; safety facts</td><td>0–30</td><td>Meaning, negation, medication, dose/number/unit, frequency, timing, conditions and suicide/self-harm meaning.</td></tr><tr><td>Fluency</td><td>0–20</td><td>Natural, clear Taiwan Traditional Chinese register.</td></tr><tr><td>Completeness &amp; format</td><td>0–20</td><td>No omission/addition; preserve visible content, paragraph/list structure and relevant layout cues.</td></tr></tbody></table></div><div class="icc-detail-grid"><article><h3>Text · {text_units} translations · 29 shared cases · k=5</h3><p><b>ICC(2,1) = {text_one}</b><br><span>Absolute-agreement point estimate for one reviewer.</span></p><p><b>ICC(2,5) = {text_k}</b><br><span>Point estimate after averaging the actual five-reviewer panel.</span></p></article><article><h3>Images · {image_units} translations · 4 cases · k=4</h3><p><b>ICC(2,1) = {image_one}</b><br><span>Absolute-agreement point estimate for one reviewer.</span></p><p><b>ICC(2,4) = {image_k}</b><br><span>Point estimate after averaging four reviewers; Mistral is excluded.</span></p></article></div><p class="rubric-note">ICC is computed across {text_units} text translation units (29 cases × 7 models), each with five reviewer scores, and {image_units} image units (4 cases × 7 models), each with four scores—not across seven model means. ICC values describe agreement in reviewer scoring, not translation accuracy, an 87.4% correctness rate, or clinical safety. No confidence intervals are reported here; these are point estimates.</p><div class="source-links rubric-sources"><a class="text-link" href="https://github.com/gregorytsai/psych-translation-benchmark/blob/main/benchmark_code/translation_benchmark_20261023_new_materials/judge_evaluation/public/instructions.md" target="_blank" rel="noopener">Open the public repository rubric ↗</a><a class="text-link" href="data/independent_review.json">Open the public ICC/audit data ↗</a><a class="text-link" href="https://pubmed.ncbi.nlm.nih.gov/27330520/" target="_blank" rel="noopener">ICC reference (Koo &amp; Li) ↗</a></div></section>
        """
    return f"""
    <section class="section-shell rubric-detail" id="rubric-icc"><div class="section-heading"><div><div class="eyebrow">可追溯 rubric 與信度</div><h2>每一項評審分數代表什麼？</h2><p class="section-intro">公開 instructions.md 對文字與圖片都定義四項整數分數，必須加總為 0–100；rubric 不是固定錯誤扣分表。</p></div></div><div class="rubric-table-wrap"><table class="rubric-table"><thead><tr><th>項目</th><th>配分</th><th>評審查看什麼</th></tr></thead><tbody><tr><td>術語與台灣用語</td><td>0–30</td><td>醫療/心理術語與台灣繁體中文，不使用簡體中文。</td></tr><tr><td>語意與安全事實</td><td>0–30</td><td>原意、否定、藥名、劑量/數字/單位、頻率、時間、條件，以及自殺/自傷意義。</td></tr><tr><td>流暢度</td><td>0–20</td><td>自然、清楚的台灣繁中語域。</td></tr><tr><td>完整與格式</td><td>0–20</td><td>不漏譯、不增譯，保留可見內容、段落/清單結構與版面線索。</td></tr></tbody></table></div><div class="icc-detail-grid"><article><h3>文字｜{text_units} 份翻譯｜29 個共同題目｜k=5</h3><p><b>ICC(2,1) = {text_one}</b><br><span>單一評審的絕對一致性點估計。</span></p><p><b>ICC(2,5) = {text_k}</b><br><span>平均當次 5 位評審後的點估計。</span></p></article><article><h3>圖片｜{image_units} 份翻譯｜4 張｜k=4</h3><p><b>ICC(2,1) = {image_one}</b><br><span>單一評審的絕對一致性點估計。</span></p><p><b>ICC(2,4) = {image_k}</b><br><span>平均 4 位評審後的點估計；不含 Mistral。</span></p></article></div><p class="rubric-note">ICC 是在 {text_units} 份文字翻譯（29 題 × 7 模型、每份 5 位評審）與 {image_units} 份圖片翻譯（4 題 × 7 模型、每份 4 位評審）上計算，不是對 7 個模型平均分計算。它衡量給分一致，不是翻譯正確率、87.4% 正確率或臨床安全；本頁未報告信賴區間，數字是點估計。</p><div class="source-links rubric-sources"><a class="text-link" href="https://github.com/gregorytsai/psych-translation-benchmark/blob/main/benchmark_code/translation_benchmark_20261023_new_materials/judge_evaluation/public/instructions.md" target="_blank" rel="noopener">開啟公開 repo 原始 rubric ↗</a><a class="text-link" href="data/independent_review.json">開啟公開 ICC／audit data ↗</a><a class="text-link" href="https://pubmed.ncbi.nlm.nih.gov/27330520/" target="_blank" rel="noopener">ICC 參考文獻（Koo &amp; Li）↗</a></div></section>
    """


def overview_chart(dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    models = sorted(dataset["models"], key=lambda item: (-(item["text_score"] or -1), item["model_name"]))
    rows: list[str] = []
    for model in models:
        text_score = max(0.0, min(100.0, float(model["text_score"] or 0)))
        image_score = max(0.0, min(100.0, float(model["image_score"] or 0)))
        text_time = max(0.0, min(6.0, float(model["text_warm_median"] or 0)))
        image_time = max(0.0, min(6.0, float(model["image_warm_median"] or 0)))
        rows.append(
            f"<div class=\"overview-row\"><div class=\"overview-name\"><a href=\"models/{esc(model['model_id'])}{'-en.html' if english else '.html'}\">{esc(model['model_name'])}</a></div><div class=\"overview-scores\"><div class=\"overview-line\"><span class=\"overview-key text-key\">{'Text' if english else '文字'}</span><div class=\"overview-track\"><i class=\"overview-fill text-fill\" style=\"width:{text_score:.3f}%\"></i></div><b>{display_number(model['text_score'])}</b></div><div class=\"overview-line\"><span class=\"overview-key image-key\">{'Image' if english else '圖片'}</span><div class=\"overview-track\"><i class=\"overview-fill image-fill\" style=\"width:{image_score:.3f}%\"></i></div><b>{display_number(model['image_score'])}</b></div></div><div class=\"overview-speed\"><div class=\"speed-line\"><span>{'Text' if english else '文字'} {display_number(model['text_warm_median'])}s</span><div class=\"speed-track\" aria-label=\"{'Text response time, 0 to 6 seconds' if english else '文字回應時間，0 到 6 秒'}\"><i style=\"left:{(text_time / 6) * 100:.2f}%\"></i></div></div><div class=\"speed-line\"><span>{'Image' if english else '圖片'} {display_number(model['image_warm_median'])}s</span><div class=\"speed-track\" aria-label=\"{'Image response time, 0 to 6 seconds' if english else '圖片回應時間，0 到 6 秒'}\"><i style=\"left:{(image_time / 6) * 100:.2f}%\"></i></div></div></div></div>"
        )
    return f"""
    <section class="section-shell overview-section"><div class="section-heading"><div><div class="eyebrow">{'Seven-model visual overview' if english else '七款模型視覺總覽'}</div><h2>{'Scores and response times in one view.' if english else '分數與回應時間，一張圖看懂。'}</h2><p class="section-intro">{'Rows are sorted by text score. Score bars use 0–100; response dots use a fixed 0–6 seconds scale, so a dot further left is faster.' if english else '依文字平均分數排序。分數條固定 0–100；回應時間點固定 0–6 秒，越靠左代表越快。'}</p></div><div class="overview-legend"><span><i class="legend-dot teal"></i>{'Text score' if english else '文字分數'}</span><span><i class="legend-dot coral"></i>{'Image score' if english else '圖片分數'}</span><span class="speed-legend"><i></i>{'Response time, 0–6s' if english else '回應時間，0–6 秒'}</span></div></div><div class="overview-chart">{''.join(rows)}</div></section>
    """


def image_case_highlight(lang: str = "zh") -> str:
    if lang == "en":
        return """
        <section class="section-shell image-case-highlight"><article><div class="eyebrow">A concrete image example</div><h2>High text scores do not prevent a medication-label copy error.</h2><p>In synthetic image case <b>img-04</b>, the source label is <code>SERTRALINE</code> while Gemma 4 26B's output shows <code>SETRALINE</code> (one R missing). All four applicable image reviewers flagged the medication-name mismatch; the image score was <b>72</b>.</p><p class="muted">This is a synthetic evaluation image, not a real prescription. It is an exact benchmark observation, not a clinical conclusion.</p><a class="button ghost" href="models/gemma4-26b-en.html#img-04-example">Open the exact model example →</a></article></section>
        """
    return """
    <section class="section-shell image-case-highlight"><article><div class="eyebrow">一個具體的圖片案例</div><h2>文字分數高，也可能抄錯藥名。</h2><p>在合成圖片案例 <b>img-04</b> 中，原圖標籤是 <code>SERTRALINE</code>，Gemma 4 26B 輸出為 <code>SETRALINE</code>（少一個 R）。適用的 4 位圖片 AI 評審都標記了藥名不一致；圖片分數為 <b>72</b>。</p><p class="muted">這是合成評估圖像，不是真實處方。它是精確的基準觀察，不延伸為臨床結論。</p><a class="button ghost" href="models/gemma4-26b.html#img-04-example">查看模型精確案例 →</a></article></section>
    """


def home_body(dataset: dict[str, Any]) -> str:
    models = dataset["models"]
    best_26 = next(m for m in models if m["model_id"] == "gemma4-26b")
    best_31 = next(m for m in models if m["model_id"] == "gemma4-31b-qat")
    diff = (best_26["text_score"] or 0) - (best_31["text_score"] or 0)
    cards = "\n".join(model_card(model) for model in models)
    return f"""
    <section class="hero section-shell">
      <div class="hero-copy">
        <div class="eyebrow">心理健康英中翻譯 · 2026-10-25 凍結檢視 · 繁體中文</div>
        <h1>七款地端模型實測</h1>
        <p class="hero-lede">一批 30 篇文字與 4 張圖片，整理出翻譯品質、模型載入後回應時間，以及每一筆可回看的翻譯。</p>
        <div class="hero-actions"><a class="button primary" href="results.html">瀏覽 238 筆翻譯 <span aria-hidden="true">→</span></a><a class="button ghost" href="methods.html">先看方法與限制</a></div>
      </div>
      <div class="hero-aside" aria-label="本輪資料範圍">{hero_result_panel(dataset)}</div>
    </section>

    <section class="section-shell stats-grid" aria-label="資料範圍摘要">
      {metric_card("共同文字題目", "29 / 30", "5 位 AI 評審；1 筆排除於共同比較", "teal")}
      {metric_card("共同圖片題目", "4 / 4", "4 位固定 AI 評審；非 OCR 字元準確率", "coral")}
      {metric_card("候選翻譯", "238 筆", "30×7 文字 + 4×7 圖片", "purple")}
      {metric_card("排除案例", "1 筆", "text-15：一個 AI 評審欄位無效", "amber")}
    </section>

    <section class="section-shell callout caution">
      <div class="callout-icon">i</div><div><strong>先讀懂這句話</strong><p>分數是本批資料的描述性評審觀察，只代表本次條件，不是通用排行榜或臨床安全證明。回應時間是模型已載入後的回應時間，只描述這次固定環境。</p></div>
    </section>

    {judge_explainer(dataset)}
    {overview_chart(dataset)}
    {image_case_highlight()}

    <section class="section-shell">
      <div class="section-heading"><div><div class="eyebrow">這次測試的三個重點</div><h2>差異有多大？要看案例數。</h2></div><a class="text-link" href="methods.html">完整解讀方法 →</a></div>
      <div class="finding-grid">
        <article class="finding"><span class="finding-no">01</span><h3>26B 與 31B 很接近</h3><p>共同文字題目均分為 <b>{display_number(best_26['text_score'])}</b> 與 <b>{display_number(best_31['text_score'])}</b>，差約 <b>{display_number(diff)}</b> 分；不宜宣稱有明顯勝負。</p></article>
        <article class="finding"><span class="finding-no">02</span><h3>圖片 31B 最高，但只有 4 張</h3><p>Gemma 4 31B QAT 圖片均分 <b>{display_number(best_31['image_score'])}</b>。這是小樣本的描述，不能直接延伸到所有圖片。</p></article>
        <article class="finding"><span class="finding-no">03</span><h3>26B 的文字回應最快</h3><p>本次模型載入後的文字回應中位 <b>{display_number(best_26['text_warm_median'])} 秒</b>；它不是跨硬體、跨設定的通用速度保證。</p></article>
      </div>
    </section>

    <section class="section-shell model-section">
      <div class="section-heading"><div><div class="eyebrow">七個候選模型</div><h2>文字、圖片，各自看。</h2><p class="section-intro">柱狀圖固定使用 0–100 刻度；均分只用共同題目，速度則是獨立的翻譯時間摘要。</p></div><span class="legend"><i class="legend-dot teal"></i>文字 <i class="legend-dot coral"></i>圖片</span></div>
      <div class="models-grid">{cards}</div>
    </section>
    {quick_compare_table(dataset)}

    <section class="section-shell split-panel">
      <div><div class="eyebrow">資料可以追到哪裡？</div><h2>從首頁，到每一筆輸出。</h2><p>結果頁可搜尋模型、案例編號或原文；模型頁提供精確範例、各類題目分數、回應時間與限制。方法頁則把共同題目、評審和速度分開說清楚。</p></div>
      <div class="link-stack"><a href="results.html" class="link-tile"><span>01</span><strong>搜尋 238 筆翻譯</strong><small>原文、繁中輸出、分數與秒數</small></a><a href="data/translation_comparison.csv" class="link-tile"><span>02</span><strong>下載完整比較 CSV</strong><small>可再分析的 238 筆公開結果</small></a><a href="methods.html" class="link-tile"><span>03</span><strong>讀方法與限制</strong><small>看懂 29、4、1 分別代表什麼</small></a></div>
    </section>
    """


def model_card_en(model: dict[str, Any]) -> str:
    slug = esc(model["model_id"])
    return f"""
    <article class="model-card">
      <div class="card-kicker">Candidate model</div>
      <h3><a href="models/{slug}-en.html">{esc(model['model_name'])}</a></h3>
      <div class="model-score-line"><span>Shared text score</span><strong>{display_number(model['text_score'])}</strong></div>
      {score_bar("Text", model["text_score"], "teal")}
      <div class="model-score-line image-line"><span>Shared image score</span><strong>{display_number(model['image_score'])}</strong></div>
      {score_bar("Image", model["image_score"], "coral")}
      <div class="mini-facts"><span>Text response median <b>{display_number(model['text_warm_median'])} s</b></span><span>Image response median <b>{display_number(model['image_warm_median'])} s</b></span></div>
      <a class="text-link" href="models/{slug}-en.html">Open model page <span aria-hidden="true">→</span></a>
    </article>
    """


def runtime_specs_html(model: dict[str, Any], dataset: dict[str, Any], lang: str = "zh") -> str:
    english = lang == "en"
    runtime = model.get("runtime", {})
    fixed = dataset.get("runtime_inventory", {}).get("fixed_generation", {})
    if not runtime:
        return ""
    model_size = display_number(runtime.get("model_size_gib"))
    mmproj_size = display_number(runtime.get("mmproj_size_gib"))
    total_size = display_number(runtime.get("total_size_gib"))
    file_label = "Model file" if english else "模型檔"
    projector_label = "mmproj file" if english else "mmproj 檔"
    total_label = "Model + mmproj on-disk size" if english else "模型＋mmproj 磁碟檔案合計"
    mode_label = "Template mode" if english else "模板模式"
    alias_label = "Runtime alias" if english else "執行別名"
    model_card_label = "Official model documentation" if english else "官方模型文件"
    model_doc = "https://ai.google.dev/gemma/docs/releases" if model["model_id"].startswith("translate") else "https://ai.google.dev/gemma/docs/core/model_card_4"
    model_doc_text = "TranslateGemma release page" if english and model["model_id"].startswith("translate") else "TranslateGemma 官方發布頁" if model["model_id"].startswith("translate") else "Gemma 4 model card" if english else "Gemma 4 官方 model card"
    fixed_text = (
        f"ctx {fixed.get('ctx_size', '—')} · max tokens {fixed.get('max_tokens', '—')} · temperature {fixed.get('temperature', '—')} · seed {fixed.get('seed', '—')} · reasoning {fixed.get('reasoning', '—')}"
        if english
        else f"上下文 {fixed.get('ctx_size', '—')} · 輸出上限 {fixed.get('max_tokens', '—')} · temperature {fixed.get('temperature', '—')} · seed {fixed.get('seed', '—')} · thinking {fixed.get('reasoning', '—')}"
    )
    note = "On-disk file size is not a RAM requirement; runtime memory also depends on context, projector, buffers and settings." if english else "這是磁碟檔案大小，不是 RAM 需求；實際記憶體還會受上下文、projector、buffer 與設定影響。"
    sha_label = "SHA-256 checksums" if english else "SHA-256 雜湊"
    settings_label = "Measured translation configuration" if english else "本次實測翻譯設定"
    return f"""
    <section class="section-shell runtime-section"><div class="section-heading"><div><div class="eyebrow">{'Measured files and settings' if english else '實測檔案與設定'}</div><h2>{'What was actually loaded?' if english else '這次實際載入了什麼？'}</h2><p class="section-intro">{'These are the public inventory facts for the measured local artifacts, not a claim about official full-precision weights.' if english else '以下是本次本機實測檔案的公開清單，不代表官方原始精度權重比較。'}</p></div></div><div class="runtime-grid"><div class="runtime-facts"><div><b>{file_label}</b><code>{esc(runtime.get('model_filename'))}</code><span>{model_size} GiB</span></div><div><b>{projector_label}</b><code>{esc(runtime.get('mmproj_filename'))}</code><span>{mmproj_size} GiB</span></div><div><b>{total_label}</b><strong>{total_size} GiB</strong></div><div><b>{mode_label}</b><span>{esc(runtime.get('template_mode'))}</span></div><div><b>{alias_label}</b><span>{esc(runtime.get('alias'))}</span></div></div><div class="runtime-settings"><b>{settings_label}</b><p>{esc(fixed_text)}</p><p class="runtime-note">{esc(note)}</p><p><a class="text-link" href="{model_doc}">{esc(model_card_label)}：{esc(model_doc_text)} ↗</a></p></div></div><details class="sha-details"><summary>{sha_label}</summary><div><code>model: {esc(runtime.get('model_sha256'))}</code><code>mmproj: {esc(runtime.get('mmproj_sha256'))}</code></div></details></section>
    """


def image_sources_panel(dataset: dict[str, Any], lang: str = "zh", depth: int = 0) -> str:
    english = lang == "en"
    prefix = "../" * depth
    cards: list[str] = []
    for image_id in ("img-01", "img-04", "img-09", "img-10"):
        source = dataset.get("image_sources", {}).get(image_id, {})
        local_path = source.get("relative_path", "")
        local_href = f"{prefix}materials/{local_path}" if local_path else ""
        local_link = f'<a class="text-link" href="{esc(local_href)}">{'Open frozen PNG' if english else '開啟凍結原圖'} ↗</a>' if local_href else ""
        pdf_link = f'<a class="text-link" href="{esc(source.get("pdf_url"))}" target="_blank" rel="noopener">{'Open original WHO PDF' if english else '開啟 WHO 原始 PDF'} ↗</a>' if source.get("pdf_url") else ""
        condition = source.get("license_note", "")
        cards.append(f"<article class=\"source-card\"><div class=\"case-label\">{esc(image_id)}</div><h3>{'WHO source render' if english and source.get('kind') == 'who' else 'Synthetic evaluation image' if english else 'WHO 來源頁面 render' if source.get('kind') == 'who' else '合成評估圖像'}</h3><p>{esc(source.get('label'))}</p><div class=\"source-links\">{local_link}{pdf_link}</div><small>{esc(condition)}</small></article>")
    return f"""
    <section class="section-shell image-sources-section"><div class="section-heading"><div><div class="eyebrow">{'Image source links' if english else '圖片原圖與來源連結'}</div><h2>{'The four image cases, linked once.' if english else '四張圖片，一次列出原圖來源。'}</h2><p class="section-intro">{'Image source links are collected here rather than repeated in all 238 rows. WHO renders carry the CC BY-NC-SA 3.0 IGO condition and translation disclaimer.' if english else '原圖連結集中在此，不在 238 筆結果逐列重複。WHO render 附 CC BY-NC-SA 3.0 IGO 條件與翻譯聲明提醒。'}</p></div></div><div class="source-card-grid">{''.join(cards)}</div></section>
    """


def image_source_links(dataset: dict[str, Any], image_id: str, lang: str = "zh", depth: int = 1) -> str:
    english = lang == "en"
    prefix = "../" * depth
    source = dataset.get("image_sources", {}).get(image_id, {})
    local_path = source.get("relative_path", "")
    local_href = f"{prefix}materials/{local_path}" if local_path else ""
    parts: list[str] = []
    if local_href:
        parts.append(f'<a class="text-link" href="{esc(local_href)}">{"Open frozen PNG" if english else "開啟凍結原圖"} ↗</a>')
    if source.get("pdf_url"):
        parts.append(f'<a class="text-link" href="{esc(source.get("pdf_url"))}" target="_blank" rel="noopener">{"Open original WHO PDF" if english else "開啟 WHO 原始 PDF"} ↗</a>')
    note = source.get("license_note", "")
    if note:
        parts.append(f'<small class="source-condition">{esc(note)}</small>')
    return '<div class="source-links">' + "".join(parts) + "</div>" if parts else ""


def home_body_en(dataset: dict[str, Any]) -> str:
    models = dataset["models"]
    best_26 = next(m for m in models if m["model_id"] == "gemma4-26b")
    best_31 = next(m for m in models if m["model_id"] == "gemma4-31b-qat")
    diff = (best_26["text_score"] or 0) - (best_31["text_score"] or 0)
    cards = "\n".join(model_card_en(model) for model in models)
    return f"""
    <section class="hero section-shell">
      <div class="hero-copy"><div class="eyebrow">Psychological health English → Traditional Chinese · 2026-10-25 frozen review</div><h1>Seven local models tested.</h1><p class="hero-lede">The same 30 text cases and 4 images, summarized with translation quality observations, response time after model loading, and every reviewable output.</p><div class="hero-actions"><a class="button primary" href="results-en.html">Browse 238 translations <span aria-hidden="true">→</span></a><a class="button ghost" href="methods-en.html">Read methods &amp; limits</a></div></div>
      <div class="hero-aside" aria-label="Key results">{hero_result_panel(dataset, "en")}</div>
    </section>

    <section class="section-shell stats-grid" aria-label="Scope summary">{metric_card("Shared text cases", "29 / 30", "5 AI reviewers; 1 excluded from the shared comparison", "teal")}{metric_card("Shared image cases", "4 / 4", "4 fixed AI reviewers; not OCR character accuracy", "coral")}{metric_card("Candidate translations", "238", "30×7 text + 4×7 image cells", "purple")}{metric_card("Excluded case", "1", "text-15: one reviewer cell was invalid", "amber")}</section>
    <section class="section-shell callout caution"><div class="callout-icon">i</div><div><strong>How to read this</strong><p>Scores are descriptive reviewer observations for this run. They describe these conditions, not a universal leaderboard or clinical-safety certification. Response time means the time after the model was loaded.</p></div></section>
    {judge_explainer(dataset, "en")}
    {overview_chart(dataset, "en")}
    {image_case_highlight("en")}
    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">Three takeaways from this test</div><h2>Sample size matters when differences are small.</h2></div><a class="text-link" href="methods-en.html">Read the full method →</a></div><div class="finding-grid"><article class="finding"><span class="finding-no">01</span><h3>26B and 31B are close</h3><p>The shared text scores are <b>{display_number(best_26['text_score'])}</b> and <b>{display_number(best_31['text_score'])}</b>, a difference of about <b>{display_number(diff)}</b> points. This run does not establish a clear winner.</p></article><article class="finding"><span class="finding-no">02</span><h3>31B leads on images, with only four images</h3><p>Gemma 4 31B QAT scores <b>{display_number(best_31['image_score'])}</b> on the image set. It is a small descriptive sample, not a general visual claim.</p></article><article class="finding"><span class="finding-no">03</span><h3>26B has the shortest text response median</h3><p>Its text response median is <b>{display_number(best_26['text_warm_median'])} seconds</b> after loading in this fixed environment; it is not a hardware-independent promise.</p></article></div></section>
    <section class="section-shell model-section"><div class="section-heading"><div><div class="eyebrow">Seven candidate models</div><h2>Read text and image tracks separately.</h2><p class="section-intro">Bars use a fixed 0–100 scale. Scores use shared cases; response times are a separate generation summary.</p></div><span class="legend"><i class="legend-dot teal"></i>Text <i class="legend-dot coral"></i>Image</span></div><div class="models-grid">{cards}</div></section>
    {quick_compare_table(dataset, "en")}
    <section class="section-shell split-panel"><div><div class="eyebrow">Trace the data</div><h2>From the homepage to every output.</h2><p>Search the results by model, case ID, or source text. Model pages show exact examples, track scores, response medians, and limits. The methods page keeps shared cases, reviewers, and speed distinct.</p></div><div class="link-stack"><a href="results-en.html" class="link-tile"><span>01</span><strong>Search 238 translations</strong><small>Source text, Traditional Chinese output, scores and seconds</small></a><a href="data/translation_comparison.csv" class="link-tile"><span>02</span><strong>Download complete comparison CSV</strong><small>Public 238-row comparison data</small></a><a href="methods-en.html" class="link-tile"><span>03</span><strong>Read methods and limits</strong><small>Why the counts are 29, 4 and 1</small></a></div></section>
    """


def detail_example_card(example: dict[str, Any], index: int) -> str:
    return f"""
    <article class="example-card">
      <div class="example-header"><span class="case-label">範例 {index:02d} · {esc(example['material_id'])}</span><span class="pill">{display_number(example['panel_mean'])} 分</span></div>
      <div class="example-source"><div class="label">英文原文（精確摘錄）</div><p>{esc(example['source_en'])}</p></div>
      <div class="example-arrow" aria-hidden="true">↓</div>
      <div class="example-target"><div class="label">模型繁中輸出</div><p>{esc(example['target_zh_tw'])}</p><small>翻譯時間 {display_number(example['translation_elapsed_seconds'])} 秒 · {esc(example['translation_status'])}</small></div>
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
    score = display_number(example.get("panel_mean"))
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
    )
    image_block = ""
    if image_example:
        image_block = f"""
        <article id="img-04-example" class="image-example-card"><div class="case-label">圖片案例 {esc(image_example['material_id'])}</div><h3>視覺輸入的繁中輸出</h3><p>{esc(image_example['target_zh_tw'])}</p><div class="mini-facts"><span>圖片均分 <b>{display_number(model['image_score'])}</b></span><span>載入後圖片回應中位 <b>{display_number(model['image_warm_median'])} 秒</b></span></div><small>圖片品質分數不是 OCR 字元準確率；此頁保留原圖來源連結。</small>{image_source_links(dataset, image_example['material_id'])}</article>
        """
    examples_html = "\n".join(curated_example_card(example, i + 1) for i, example in enumerate(curated[:3])) if curated else "\n".join(detail_example_card(example, i + 1) for i, example in enumerate(examples))
    return f"""
    <section class="section-shell page-hero compact">
      <div class="eyebrow">模型細頁 · {esc(model_id)}</div><h1>{esc(model['model_name'])}</h1><p>在固定翻譯設定、同一批素材與共同題目評審摘要下，這個頁面把可核對的輸出和限制放在一起。</p>
      <div class="model-chips" aria-label="切換模型">{other_models}</div>
    </section>

    <section class="section-shell metrics-row">
      {metric_card("共同文字題目均分", display_number(model['text_score']), f"{model['text_items']} 篇共同題目；中位數 {display_number(model['text_median'])}", "teal")}
      {metric_card("共同圖片題目均分", display_number(model['image_score']), f"{model['image_items']} 張；中位數 {display_number(model['image_median'])}", "coral")}
      {metric_card("載入後文字回應中位", f"{display_number(model['text_warm_median'])} 秒", f"p95 {display_number(model['text_p95'])} 秒；{model['text_cases']} 筆", "purple")}
      {metric_card("載入後圖片回應中位", f"{display_number(model['image_warm_median'])} 秒", f"p95 {display_number(model['image_p95'])} 秒；{model['image_cases']} 筆", "amber")}
    </section>
    {runtime_specs_html(model, dataset)}

    <section class="section-shell split-panel model-reading">
      <div><div class="eyebrow">怎麼讀這個模型？</div><h2>先看共同題目，再看輸出。</h2><p>文字均分來自 29 個共同題目和 5 位固定 AI 評審；圖片均分來自 4 張圖片和 4 位固定 AI 評審。分數是觀察值，不是人工真值準確率，也沒有把不完整評審欄位補成 0。</p><p>本頁的載入後回應中位數來自固定翻譯測試。它回答「這次設定中回應花多久」，不回答「你的硬體一定會多快」。</p></div>
      <div class="accent-note"><strong>警示數</strong><span>文字評審警示欄位：{model['text_critical']}</span><span>文字共同題目排除：{model['text_excluded']}</span><small>警示與排除是獨立診斷，不能直接換算成臨床風險。</small></div>
    </section>

    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">精確範例</div><h2>原文與輸出，並排看。</h2><p class="section-intro">下列文字直接來自已發布的 allow-list 結果欄位；沒有重新生成或臆造。</p></div><a class="text-link" href="../results.html">看全部 238 筆 →</a></div><div class="examples-grid">{examples_html}</div></section>

    <section class="section-shell">{image_block}</section>

    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">詳細數據</div><h2>這個模型的所有案例摘要</h2></div></div><div class="table-wrap"><table><thead><tr><th>案例</th><th>類型</th><th>評審均分</th><th>翻譯秒數</th><th>有效 / 要求</th></tr></thead><tbody>{detail_rows}</tbody></table></div></section>

    <section class="section-shell callout caution"><div class="callout-icon">!</div><div><strong>限制請一起讀</strong><p>文字只含 30 篇、圖片只含 4 張；5 位 AI 評審不是人工真值標準，圖片也不是逐字辨識錯誤率；模型細頁不構成排名或臨床安全結論。</p></div></section>
    """


def detail_example_card_en(example: dict[str, Any], index: int) -> str:
    return f"""
    <article class="example-card"><div class="example-header"><span class="case-label">Example {index:02d} · {esc(example['material_id'])}</span><span class="pill">{display_number(example['panel_mean'])} score</span></div><div class="example-source"><div class="label">English source (exact excerpt)</div><p>{esc(example['source_en'])}</p></div><div class="example-arrow" aria-hidden="true">↓</div><div class="example-target"><div class="label">Model output in Traditional Chinese</div><p>{esc(example['target_zh_tw'])}</p><small>generation {display_number(example['translation_elapsed_seconds'])} s · {esc(example['translation_status'])}</small></div></article>
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
    detail_rows = "\n".join(f"<tr><td>{esc(r['material_id'])}</td><td>{'Text' if r['source_kind'] == 'text' else 'Image'}</td><td>{display_number(r['panel_mean'])}</td><td>{display_number(r['translation_elapsed_seconds'])}</td><td>{display_integer(r['panel_n_valid'])}/{display_integer(r['panel_required_k'])}</td></tr>" for r in rows)
    other_models = " ".join(f'<a class="model-chip{(" active" if m["model_id"] == model_id else "")}" href="{esc(m["model_id"])}-en.html">{esc(m["model_name"])}</a>' for m in dataset["models"])
    image_block = ""
    if image_example:
        image_block = f"<article id=\"img-04-example\" class=\"image-example-card\"><div class=\"case-label\">Image case {esc(image_example['material_id'])}</div><h3>Traditional Chinese output for the visual input</h3><p>{esc(image_example['target_zh_tw'])}</p><div class=\"mini-facts\"><span>4-image mean <b>{display_number(model['image_score'])}</b></span><span>Response median <b>{display_number(model['image_warm_median'])} s</b></span></div><small>Image quality scores are not OCR character accuracy; the frozen source link is shown below.</small>{image_source_links(dataset, image_example['material_id'], 'en')}</article>"
    examples_html = "\n".join(curated_example_card(example, i + 1, "en") for i, example in enumerate(curated[:3])) if curated else "\n".join(detail_example_card_en(example, i + 1) for i, example in enumerate(examples))
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">Model page · {esc(model_id)}</div><h1>{esc(model['model_name'])}</h1><p>Under fixed translation settings, the same materials, and a shared-case panel, this page places auditable outputs beside the limits of the measurement.</p><div class="model-chips" aria-label="Switch model">{other_models}</div></section>
    <section class="section-shell metrics-row">{metric_card("Shared text score", display_number(model['text_score']), f"{model['text_items']} shared cases; median {display_number(model['text_median'])}", "teal")}{metric_card("Shared image score", display_number(model['image_score']), f"{model['image_items']} images; median {display_number(model['image_median'])}", "coral")}{metric_card("Text response median", f"{display_number(model['text_warm_median'])} s", f"p95 {display_number(model['text_p95'])} s; {model['text_cases']} cases", "purple")}{metric_card("Image response median", f"{display_number(model['image_warm_median'])} s", f"p95 {display_number(model['image_p95'])} s; {model['image_cases']} cases", "amber")}</section>
    {runtime_specs_html(model, dataset, "en")}
    <section class="section-shell split-panel model-reading"><div><div class="eyebrow">How to read this model</div><h2>Start with shared cases, then inspect outputs.</h2><p>The text score uses 29 shared cases and five fixed reviewers; the image score uses four images and four fixed reviewers. These are observations, not a gold accuracy score, and incomplete panels are not filled with zeros.</p><p>The response medians come from the frozen translation run. They answer how long this configuration took after loading, not how fast every machine will be.</p></div><div class="accent-note"><strong>Diagnostics</strong><span>Text critical flag cells: {model['text_critical']}</span><span>Text shared-case exclusions: {model['text_excluded']}</span><small>Critical flags and exclusions are separate diagnostics, not clinical-risk estimates.</small></div></section>
    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">Exact examples</div><h2>Source and output, side by side.</h2><p class="section-intro">These text fields are taken directly from the allow-listed results; they were not regenerated or invented for this page.</p></div><a class="text-link" href="../results-en.html">See all 238 →</a></div><div class="examples-grid">{examples_html}</div></section>
    <section class="section-shell">{image_block}</section>
    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">Detailed data</div><h2>All case summaries for this model</h2></div></div><div class="table-wrap"><table><thead><tr><th>Case</th><th>Track</th><th>Panel score</th><th>Generation seconds</th><th>Valid / required</th></tr></thead><tbody>{detail_rows}</tbody></table></div></section>
    <section class="section-shell callout caution"><div class="callout-icon">!</div><div><strong>Read the limits with the score</strong><p>There are only 30 text cases and 4 images; five reviewers are not a human gold standard, and images are not OCR CER. A model page is not a ranking or clinical-safety conclusion.</p></div></section>
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
    <section class="section-shell page-hero compact"><div class="eyebrow">可搜尋資料頁</div><h1>238 筆翻譯，逐筆瀏覽。</h1><p>搜尋案例編號、模型、英文原文或繁中輸出。表格只使用 allow-list 欄位；原始 JSONL、路徑與 logs 不在公開站內。</p></section>
    <section class="section-shell results-toolbar" aria-label="結果篩選"><label class="search-field"><span>搜尋</span><input id="result-search" type="search" placeholder="例如：text-28、PTSD、Gemma 26B" autocomplete="off"></label><label><span>軌道</span><select id="result-track"><option value="all">全部</option><option value="text">文字</option><option value="image">圖片</option></select></label><label><span>模型</span><select id="result-model"><option value="all">全部模型</option>{model_options}</select></label><div id="result-count" class="result-count" data-template="顯示 {{visible}} / {{total}} 筆" aria-live="polite">顯示 {len(dataset['rows'])} / {len(dataset['rows'])} 筆</div></section>
    <section class="section-shell table-section results-table-section"><div class="table-wrap"><table id="results-table"><caption class="sr-only">七個模型的 238 筆翻譯結果</caption><thead><tr><th>案例</th><th>類型</th><th>英文輸入 / 圖片</th><th>繁中輸出</th><th>評審摘要</th><th>翻譯時間</th></tr></thead><tbody>{rows_html}</tbody></table></div></section>
    <section class="section-shell callout"><div class="callout-icon">↓</div><div><strong>機器可讀資料</strong><p>相同的安全 allow-list 也提供為 <a class="text-link" href="data/translation_results.json">translation_results.json</a>，方便重現頁面篩選與檢查。JSON 不含 raw path、secret、PID 或未整理的 AI 評審回覆。</p></div></section>
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
    <section class="section-shell page-hero compact"><div class="eyebrow">Searchable data page</div><h1>238 translations, row by row.</h1><p>Search by case ID, model, English source, or Traditional Chinese output. The table uses allow-listed fields; raw JSONL, paths, and logs are not part of the public site.</p></section>
    <section class="section-shell results-toolbar" aria-label="Result filters"><label class="search-field"><span>Search</span><input id="result-search" type="search" placeholder="For example: text-28, PTSD, Gemma 26B" autocomplete="off"></label><label><span>Track</span><select id="result-track"><option value="all">All</option><option value="text">Text</option><option value="image">Image</option></select></label><label><span>Model</span><select id="result-model"><option value="all">All models</option>{model_options}</select></label><div id="result-count" class="result-count" data-template="Showing {{visible}} / {{total}} rows" aria-live="polite">Showing {len(dataset['rows'])} / {len(dataset['rows'])} rows</div></section>
    <section class="section-shell table-section results-table-section"><div class="table-wrap"><table id="results-table"><caption class="sr-only">238 translation results from seven models</caption><thead><tr><th>Case</th><th>Track</th><th>English source / image</th><th>Traditional Chinese output</th><th>Panel summary</th><th>Generation</th></tr></thead><tbody>{rows_html}</tbody></table></div></section>
    <section class="section-shell callout"><div class="callout-icon">↓</div><div><strong>Machine-readable data</strong><p>The same safe allow-list is available as <a class="text-link" href="data/translation_results.json">translation_results.json</a> and the complete public <a class="text-link" href="data/translation_comparison.csv">translation_comparison.csv</a>. Neither contains raw paths, secrets, PIDs, or unprocessed reviewer responses.</p></div></section>
    """


def methods_body(dataset: dict[str, Any]) -> str:
    judge_rows = "\n".join(
        f"<tr><td>{esc(r.get('judge_slot'))}</td><td>{esc(r.get('judge_label'))}</td><td>{display_integer(r.get('pointwise_scalar_valid'))} / {display_integer(r.get('pointwise_expected'))}</td><td>{display_number(r.get('latency_median_s'))}</td></tr>"
        for r in dataset["judges"]
    )
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">方法與限制</div><h1>把數字放回它的上下文。</h1><p>本頁保留需要的技術細節，但先把一般讀者最容易誤讀的地方說清楚：共同題目、AI 評審群組、翻譯回應時間是三件不同的事。</p></section>

    <section class="section-shell method-flow"><article><span>01</span><h2>固定素材</h2><p>30 篇英文文字、4 張圖片；每個候選模型各跑一筆，合計 238 個模型－題目翻譯欄位。</p></article><article><span>02</span><h2>固定設定</h2><p>翻譯測試使用固定上下文、temperature 0、關閉 thinking 與逐一請求；詳細設定留在可重建來源。</p></article><article><span>03</span><h2>固定評審群組</h2><p>文字使用 5 位 AI 評審；圖片使用 4 位固定 AI 評審。這是觀察結果，不是人工金標。</p></article></section>

    <section class="section-shell split-panel"><div><div class="eyebrow">為什麼是 29？</div><h2>1 筆無效，不補 0 分。</h2><p>文字共同題目要求同一案例有完整的 5 位 AI 評審結果。text-15 有一個評審欄位無效，因此共同比較集為 29 篇；仍保留候選翻譯與原始有效資訊。</p><p>圖片共同題目為 4 張、每張使用 4 位固定 AI 評審。Mistral 的圖片路徑屬於設計上不支援，因此不把它補成低分。</p></div><div class="big-number"><strong>29</strong><span>共同文字案例</span><small>12 個來源頁面</small></div></section>

    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">分數怎麼算？</div><h2>共同案例評審均分</h2></div></div><div class="definition-grid"><div class="definition"><h3>分母</h3><p>每類題目的完整共同題目；文字 29、圖片 4。</p></div><div class="definition"><h3>分數</h3><p>每筆有效 AI 評審分數先取題內平均，再對共同題目做描述性算術平均。</p></div><div class="definition"><h3>不代表</h3><p>不是翻譯正確率、不是臨床安全分數，也不是跨模型通用排行榜。</p></div></div></section>
    {rubric_reliability_section(dataset)}
    {image_sources_panel(dataset)}

    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">AI 評審摘要</div><h2>觀察來源與速度分開。</h2><p class="section-intro">下表的 latency 是 AI 評審回覆的摘要；首頁與模型頁的載入後回應秒數只來自翻譯測試，兩者不可混用。</p></div></div><div class="table-wrap"><table><thead><tr><th>欄位</th><th>標籤</th><th>有效 / 預期</th><th>評審回覆中位（秒）</th></tr></thead><tbody>{judge_rows}</tbody></table></div></section>

    <section class="section-shell limitations"><div class="eyebrow">限制清單</div><h2>哪些問題，這份資料回答不了？</h2><div class="limit-list"><div><b>樣本小</b><span>文字 30、圖片 4；圖片最高分不能當作普遍視覺能力。</span></div><div><b>評審結果不是人工真值</b><span>5 位 AI 評審的一致性不等於翻譯真值或臨床安全。</span></div><div><b>圖片不是逐字辨識</b><span>圖片結果是端到端翻譯品質觀察，不是逐字 OCR CER。</span></div><div><b>速度不可外推</b><span>翻譯時間中位數受硬體、runtime、設定和請求內容影響。</span></div><div><b>不做完整勝率</b><span>兩兩比較不是完整 round-robin，不從不完整比較圖推導排名。</span></div><div><b>原始檔不公開</b><span>網站排除 raw attempt、絕對路徑、PID、private binding 與未整理 logs。</span></div></div></section>

    <section class="section-shell reproducibility"><div><div class="eyebrow">重建</div><h2>可重跑，但邊界先說明。</h2><p>網站由根目錄的 <code>build_site.py</code> 用 Python 標準庫重建。它從 frozen review 的 CSV 讀取資料，先做 allow-list，再寫入 docs。若結果檔或方法檔更新，請重新核對公開欄位與授權。</p></div><pre><code>python build_site.py
python build_site.py --results &lt;latest_results&gt; --materials &lt;materials&gt;</code></pre></section>
    """


def downloads_body() -> str:
    return """
    <section class="section-shell page-hero compact"><div class="eyebrow">下載</div><h1>可讀資料先行，完整套件研究中。</h1><p>這個頁面先保留公開資料的清楚入口；完整結果包、素材與程式碼的發布格式正在獨立檢查。</p></section>
    <section class="section-shell download-grid"><article class="download-card ready"><span class="download-status">目前可用</span><h2>安全翻譯資料</h2><p>238 筆翻譯的 allow-list JSON；含來源文字、繁中輸出、評審摘要與翻譯秒數，不含 raw paths 或 logs。</p><a class="button primary" href="data/translation_results.json" download>下載 translation_results.json</a><a class="text-link" href="data/translation_comparison.csv">下載完整比較 CSV →</a></article><article class="download-card pending"><span class="download-status">研究中</span><h2>完整結果包</h2><p>CSV、方法報告、可重建程式與素材的分層下載仍待主發布者完成授權與隱私檢查。</p><span class="muted">尚未公開連結</span></article><article class="download-card pending"><span class="download-status">研究中</span><h2>原始輸出 / logs</h2><p>原始 AI 評審回覆、模型路徑、PID 與 request logs 不在公開站內；不會因為下載方便而直接打包。</p><span class="muted">不公開</span></article></section>
    <section class="section-shell callout caution"><div class="callout-icon">i</div><div><strong>下載前請看授權頁</strong><p>資料欄位可讀不等於第三方素材、模型權重或原始來源都可再散布。<a class="text-link" href="licenses.html">查看授權與來源狀態 →</a></p></div></section>
    """


def methods_body_en(dataset: dict[str, Any]) -> str:
    judge_rows = "\n".join(
        f"<tr><td>{esc(r.get('judge_slot'))}</td><td>{esc(r.get('judge_label'))}</td><td>{display_integer(r.get('pointwise_scalar_valid'))} / {display_integer(r.get('pointwise_expected'))}</td><td>{display_number(r.get('latency_median_s'))}</td></tr>"
        for r in dataset["judges"]
    )
    return f"""
    <section class="section-shell page-hero compact"><div class="eyebrow">Methods and limits</div><h1>Put each number back in context.</h1><p>This page keeps the technical details short while explaining the distinctions most likely to be missed: shared cases, reviewer panels, and translation generation time are different things.</p></section>
    <section class="section-shell method-flow"><article><span>01</span><h2>Fixed materials</h2><p>30 English text cases and 4 images; every candidate model produced one translation per case, for 238 model-case cells.</p></article><article><span>02</span><h2>Fixed settings</h2><p>The frozen translation run used fixed context, temperature 0, thinking off, and one request at a time. Full settings remain in the reproducible source bundle.</p></article><article><span>03</span><h2>Fixed panel</h2><p>Text used five reviewers; images used four fixed reviewers. This is an observed panel, not a human gold standard.</p></article></section>
    <section class="section-shell split-panel"><div><div class="eyebrow">Why 29?</div><h2>One invalid cell is not a zero.</h2><p>A text shared-case item requires a complete five-reviewer panel. text-15 had one invalid reviewer cell, so the shared text comparison contains 29 items. The candidate translation and valid information remain available.</p><p>The image shared set contains four images and four fixed reviewers per image. Mistral's image route was unsupported by design, so it was not filled with a low score.</p></div><div class="big-number"><strong>29</strong><span>shared text cases</span><small>12 source pages</small></div></section>
    <section class="section-shell"><div class="section-heading"><div><div class="eyebrow">How scores are formed</div><h2>Shared-case reviewer means</h2></div></div><div class="definition-grid"><div class="definition"><h3>Denominator</h3><p>Complete shared-case items per track: 29 text items and 4 image items.</p></div><div class="definition"><h3>Score</h3><p>Valid reviewer scores are averaged within each item, then averaged descriptively across shared items.</p></div><div class="definition"><h3>Not a claim of</h3><p>Translation accuracy, clinical safety, or a universal cross-model leaderboard.</p></div></div></section>
    {rubric_reliability_section(dataset, "en")}
    {image_sources_panel(dataset, "en")}
    <section class="section-shell table-section"><div class="section-heading"><div><div class="eyebrow">Reviewer summary</div><h2>Review and speed stay separate.</h2><p class="section-intro">The latency below is reviewer-response latency. Homepage and model-page response times come from translation generation; the two must not be mixed.</p></div></div><div class="table-wrap"><table><thead><tr><th>Slot</th><th>Label</th><th>Valid / expected</th><th>Reviewer latency median (s)</th></tr></thead><tbody>{judge_rows}</tbody></table></div></section>
    <section class="section-shell limitations"><div class="eyebrow">Limits</div><h2>What this dataset cannot answer</h2><div class="limit-list"><div><b>Small sample</b><span>30 text cases and 4 images; a top image score is not a general visual claim.</span></div><div><b>Panel is not gold</b><span>Reviewer agreement does not establish translation truth or clinical safety.</span></div><div><b>Images are not OCR</b><span>Image results are end-to-end translation observations, not character-level OCR CER.</span></div><div><b>Speed is local</b><span>Generation medians depend on hardware, runtime, settings, and request content.</span></div><div><b>No full win rate</b><span>Pairwise comparisons are not a complete round-robin, so no full ranking is inferred.</span></div><div><b>Raw files stay private</b><span>The site excludes raw attempts, absolute paths, PIDs, private bindings, and unprocessed logs.</span></div></div></section>
    <section class="section-shell reproducibility"><div><div class="eyebrow">Rebuild</div><h2>Reproducible, with boundaries.</h2><p>The site is rebuilt by <code>build_site.py</code> with the Python standard library. It reads the committed frozen CSV snapshot, applies an explicit allow-list, and writes docs. If the result or method files change, re-check public fields and licensing.</p></div><pre><code>python build_site.py
python build_site.py --results &lt;latest_results&gt; --materials &lt;materials&gt;</code></pre></section>
    """


def downloads_body_en() -> str:
    return """
    <section class="section-shell page-hero compact"><div class="eyebrow">Downloads</div><h1>Readable data first; the full bundle is under review.</h1><p>This page provides clear entry points for public data while the complete result package, materials, and code undergo separate licensing and privacy checks.</p></section>
    <section class="section-shell download-grid"><article class="download-card ready"><span class="download-status">Available now</span><h2>Safe translation data</h2><p>The allow-listed JSON contains 238 translations with source text, Traditional Chinese outputs, panel summaries, and generation seconds. It excludes raw paths and logs.</p><a class="button primary" href="data/translation_results.json" download>Download translation_results.json</a><a class="text-link" href="data/translation_comparison.csv">Open complete comparison CSV →</a></article><article class="download-card pending"><span class="download-status">Under review</span><h2>Complete result bundle</h2><p>The layered package of CSVs, method reports, rebuild code, and materials is waiting for the publisher's licensing and privacy review.</p><span class="muted">Link not published yet</span></article><article class="download-card pending"><span class="download-status">Under review</span><h2>Raw outputs and logs</h2><p>Raw reviewer responses, model paths, PIDs, and request logs are not part of the public site and will not be bundled just for convenience.</p><span class="muted">Not public</span></article></section>
    <section class="section-shell callout caution"><div class="callout-icon">i</div><div><strong>Read licensing before reuse</strong><p>Readable data is not the same as permission to redistribute third-party source material, model weights, or raw assets. <a class="text-link" href="licenses-en.html">See licensing and source status →</a></p></div></section>
    """


def licenses_body_en() -> str:
    return """
    <section class="section-shell page-hero compact"><div class="eyebrow">Licensing and sources</div><h1>Make rights clear before publishing.</h1><p>This page is an explicit under-review placeholder. The fact that a field is visible on the site is not presented as a complete license statement.</p></section>
    <section class="section-shell license-list"><article><span class="license-state">Under review</span><h2>Models and runtime</h2><p>The model cards, weights, quantized files, and runtime/code licenses for the seven candidates are being consolidated by the publisher. This site does not provide model weights.</p></article><article><span class="license-state">Under review</span><h2>Text sources</h2><p>Source labels and text are retained in the frozen review. Terms for third-party public-health pages and citations will be listed item by item after licensing research.</p></article><article><span class="license-state">Under review</span><h2>Image materials</h2><p>The image cases include synthetic layouts and rendered source pages. This site publishes case-level descriptions and translation outputs, not the original PNG files.</p></article><article><span class="license-state">Under review</span><h2>Site code</h2><p>The site is generated with the Python standard library. The license file and third-party notices will be added with the complete release bundle; this page is not legal advice.</p></article></section>
    <section class="section-shell callout"><div class="callout-icon">✓</div><div><strong>Current boundary</strong><p>The public data is an allow-listed benchmark summary. It excludes private bindings, absolute paths, raw logs, model weights, and unprocessed request content.</p></div></section>
    """


def licenses_body() -> str:
    return """
    <section class="section-shell page-hero compact"><div class="eyebrow">授權與來源</div><h1>公開前，先把權利說清楚。</h1><p>這頁目前是明確的研究中佔位；不把「資料在網站上」誤寫成完整授權聲明。</p></section>
    <section class="section-shell license-list"><article><span class="license-state">研究中</span><h2>模型與 runtime</h2><p>七個候選模型與執行 runtime 的模型卡、權重、量化檔和程式碼授權尚待主發布者整合。本站不提供模型權重。</p></article><article><span class="license-state">研究中</span><h2>文字來源</h2><p>原始來源標籤和內容保留於 frozen review；第三方網頁、公共衛生資料與引用條款將在授權研究完成後逐項列出。</p></article><article><span class="license-state">研究中</span><h2>圖片素材</h2><p>圖片案例包含合成版面與來源頁面 render；本站只公開案例級描述與翻譯輸出，不重新散布原始 PNG。</p></article><article><span class="license-state">研究中</span><h2>本網站程式</h2><p>網站由 Python 標準庫產生，授權檔與第三方 notices 待完整發布包一起補上。直到那時，請不要把本頁當作法律意見。</p></article></section>
    <section class="section-shell callout"><div class="callout-icon">✓</div><div><strong>目前可確認的邊界</strong><p>公開資料是經 allow-list 的 benchmark 摘要，不含 private bindings、絕對路徑、raw logs、模型權重或未整理的請求內容。</p></div></section>
    """


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


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


STYLE_CSS = r"""
.language-switch{padding:7px 11px;margin-left:8px;border:1px solid #9ed3ce;border-radius:99px;text-decoration:none;color:var(--teal);font-size:.8rem;font-weight:800;background:#fff}.language-switch:hover{background:var(--teal-soft)}
.quick-compare-section{padding-top:22px;padding-bottom:30px}.quick-compare-wrap{box-shadow:0 10px 28px rgba(27,45,49,.05)}.quick-compare{min-width:760px}.quick-compare th,.quick-compare td{padding:9px 13px}.quick-compare td:first-child{font-weight:700}.quick-compare td a{text-decoration:none}.quick-compare td a:hover{color:var(--teal)}.quick-compare td strong{font-size:1rem}
.hero h1{font-size:clamp(2rem,4vw,3.4rem)}
.hero{padding-top:48px;padding-bottom:42px;min-height:0}.hero-aside{min-height:250px}.orbit-one{width:245px;height:165px}.orbit-two{width:180px;height:260px}.hero-orbit-card{width:175px;height:175px}.hero-orbit-card strong{font-size:3rem}
.image-case-highlight{padding-top:22px;padding-bottom:30px}.image-case-highlight article{background:linear-gradient(135deg,#fff7ef,#fff);border:1px solid #f0d3b9;border-radius:18px;padding:24px 28px;box-shadow:0 10px 28px rgba(216,107,82,.07)}.image-case-highlight h2{max-width:780px}.image-case-highlight p{max-width:850px}.image-case-highlight code{background:#ffe9dc;color:#9a4e3b;border-radius:5px;padding:2px 5px;font-family:"SFMono-Regular",Consolas,monospace;font-weight:800}
.curated-example{border-top:4px solid var(--purple)}.curated-evidence{margin-top:14px;padding:12px 14px;background:var(--purple-soft);border-radius:10px}.curated-evidence p{font-size:.86rem;color:#52616a;margin:0}
:root{--ink:#1b2530;--muted:#647181;--paper:#f8faf9;--card:#fff;--line:#dbe5e5;--teal:#087f7b;--teal-soft:#d9f2ee;--coral:#d86b52;--coral-soft:#fff0eb;--purple:#7556a5;--purple-soft:#eee8fa;--amber:#a36b16;--amber-soft:#fff4d7;--shadow:0 18px 50px rgba(27,45,49,.09);--radius:20px;--max:1160px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;line-height:1.72}a{color:inherit}p{margin:0 0 1rem}.muted,.metric-note,.section-intro,small{color:var(--muted)}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.skip-link{position:absolute;left:1rem;top:-4rem;background:var(--ink);color:#fff;padding:.6rem 1rem;border-radius:0 0 10px 10px;z-index:20}.skip-link:focus{top:0}.site-header{position:sticky;top:0;z-index:10;background:rgba(248,250,249,.94);backdrop-filter:blur(14px);border-bottom:1px solid rgba(219,229,229,.86)}.header-inner{max-width:var(--max);margin:auto;display:flex;align-items:center;min-height:74px;padding:0 24px;gap:28px}.brand{display:flex;align-items:center;gap:11px;text-decoration:none;font-weight:800;line-height:1.15;letter-spacing:.02em}.brand em{font-size:.7rem;font-style:normal;letter-spacing:.14em;color:var(--muted);font-weight:600}.brand-mark{display:grid;place-items:center;width:37px;height:37px;border-radius:12px;background:var(--ink);color:#fff;font-size:1.18rem;box-shadow:6px 6px 0 var(--teal-soft)}.site-nav{margin-left:auto;display:flex;align-items:center;gap:4px}.nav-link{padding:8px 12px;text-decoration:none;border-radius:10px;color:var(--muted);font-size:.92rem}.nav-link:hover,.nav-link.active{background:#e8f2f1;color:var(--teal);font-weight:700}.nav-toggle{display:none;border:1px solid var(--line);background:var(--card);border-radius:10px;padding:8px 11px;color:var(--ink)}
.section-shell{max-width:var(--max);margin:0 auto;padding:64px 24px}.hero{display:grid;grid-template-columns:1.16fr .84fr;gap:64px;align-items:center;padding-top:94px;padding-bottom:74px;min-height:560px}.eyebrow{font-size:.77rem;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:800;margin-bottom:12px}.hero h1,.page-hero h1{font-size:clamp(2.7rem,6vw,5.5rem);line-height:1.08;letter-spacing:-.055em;margin:0 0 24px;max-width:760px}.accent-text{color:var(--teal)}.hero-lede{font-size:1.15rem;color:#4a5a67;max-width:650px}.hero-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:30px}.button{display:inline-flex;align-items:center;gap:10px;padding:12px 18px;border-radius:12px;text-decoration:none;font-weight:800;border:1px solid transparent;transition:transform .18s ease,box-shadow .18s ease}.button:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(27,45,49,.1)}.button.primary{background:var(--teal);color:#fff}.button.ghost{border-color:var(--line);background:var(--card);color:var(--ink)}.hero-aside{min-height:350px;display:grid;place-items:center;position:relative}.orbit{position:absolute;border:1px solid #b8ddda;border-radius:50%;transform:rotate(-18deg)}.orbit-one{width:310px;height:210px}.orbit-two{width:220px;height:320px;border-color:#f0c1b5;transform:rotate(58deg)}.hero-orbit-card{width:220px;height:220px;border-radius:50%;background:linear-gradient(145deg,#fff,#e7f5f2);box-shadow:var(--shadow);display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;position:relative;border:7px solid #fff}.hero-orbit-card strong{font-size:3.7rem;line-height:1;color:var(--teal);letter-spacing:-.08em}.orbit-label{font-size:.78rem;color:var(--muted);letter-spacing:.12em}.hero-orbit-card span:last-child{font-size:.86rem;margin-top:8px;color:#4a5a67}.stats-grid,.metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;padding-top:0}.metric-card{padding:20px 20px 18px;border:1px solid var(--line);border-radius:16px;background:var(--card);min-height:126px}.metric-card.teal{border-top:4px solid var(--teal)}.metric-card.coral{border-top:4px solid var(--coral)}.metric-card.purple{border-top:4px solid var(--purple)}.metric-card.amber{border-top:4px solid var(--amber)}.metric-label{font-size:.86rem;color:var(--muted);font-weight:700}.metric-value{font-size:2.15rem;font-weight:850;line-height:1.2;letter-spacing:-.045em;margin:6px 0 3px}.metric-note{font-size:.78rem;line-height:1.45}.callout{display:flex;gap:16px;align-items:flex-start;background:#eef7f5;border:1px solid #cae6e1;border-radius:16px;padding-top:24px;padding-bottom:24px}.callout.caution{background:var(--amber-soft);border-color:#f0dcae}.callout-icon{display:grid;place-items:center;flex:0 0 28px;height:28px;border-radius:50%;background:var(--teal);color:#fff;font-weight:800}.caution .callout-icon{background:var(--amber)}.callout strong{display:block;margin-bottom:2px}.callout p{margin-bottom:0;color:#52616a}.section-heading{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:26px}.section-heading h2,.section-shell h2{font-size:clamp(1.8rem,3vw,2.65rem);letter-spacing:-.04em;line-height:1.18;margin:0 0 8px}.section-intro{max-width:660px}.text-link{color:var(--teal);font-weight:800;text-decoration:none;white-space:nowrap}.text-link:hover{text-decoration:underline}.finding-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.finding{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;min-height:225px}.finding-no{display:inline-grid;place-items:center;width:32px;height:32px;border-radius:10px;background:var(--teal-soft);color:var(--teal);font-weight:850;margin-bottom:24px}.finding h3{font-size:1.2rem;margin:0 0 8px;letter-spacing:-.02em}.finding p{color:#52616a;font-size:.96rem}.model-section{padding-top:42px}.legend{font-size:.83rem;color:var(--muted);white-space:nowrap}.legend-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 10px}.legend-dot.teal{background:var(--teal)}.legend-dot.coral{background:var(--coral)}.models-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:17px}.model-card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:23px;transition:transform .18s ease,box-shadow .18s ease}.model-card:hover{transform:translateY(-3px);box-shadow:var(--shadow)}.card-kicker{font-size:.72rem;letter-spacing:.13em;color:var(--muted);text-transform:uppercase}.model-card h3{margin:4px 0 18px;font-size:1.28rem}.model-card h3 a{text-decoration:none}.model-card h3 a:hover{color:var(--teal)}.model-score-line,.bar-label{display:flex;justify-content:space-between;gap:10px;align-items:baseline;font-size:.84rem;color:var(--muted)}.model-score-line strong,.bar-label strong{color:var(--ink);font-size:1.04rem}.image-line{margin-top:18px}.bar-row{margin-top:6px}.bar-track{height:10px;border-radius:99px;background:#edf1f1;overflow:hidden}.bar-fill{height:100%;display:block;border-radius:inherit}.bar-fill.teal{background:linear-gradient(90deg,#0a8d86,#48b8a4)}.bar-fill.coral{background:linear-gradient(90deg,#d86b52,#ecaa8d)}.mini-facts{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0 15px;color:var(--muted);font-size:.78rem}.mini-facts span{background:#f3f6f6;padding:5px 8px;border-radius:8px}.mini-facts b{color:var(--ink)}.split-panel{display:grid;grid-template-columns:1fr 1fr;gap:52px;align-items:center}.split-panel p{color:#52616a}.link-stack{display:grid;gap:10px}.link-tile{display:grid;grid-template-columns:35px 1fr;column-gap:13px;align-items:center;padding:15px 17px;border:1px solid var(--line);background:var(--card);border-radius:14px;text-decoration:none}.link-tile:hover{border-color:#9ed3ce}.link-tile span{grid-row:span 2;color:var(--teal);font-size:.82rem;font-weight:850}.link-tile small{grid-column:2;font-size:.8rem}.page-hero{padding-bottom:30px}.page-hero.compact{padding-top:72px;min-height:0}.page-hero h1{font-size:clamp(2.3rem,5vw,4.2rem);max-width:820px}.page-hero p{max-width:750px;color:#52616a;font-size:1.08rem}.model-chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:28px}.model-chip{padding:7px 11px;border-radius:99px;border:1px solid var(--line);background:var(--card);text-decoration:none;color:var(--muted);font-size:.82rem}.model-chip:hover,.model-chip.active{color:var(--teal);border-color:#9ed3ce;background:var(--teal-soft);font-weight:750}.metrics-row{padding-top:12px;padding-bottom:12px}.model-reading{padding-top:38px;padding-bottom:38px}.accent-note{background:var(--coral-soft);border:1px solid #f2cec3;border-radius:16px;padding:20px;display:flex;flex-direction:column;gap:7px}.accent-note strong{font-size:1.65rem;color:var(--coral)}.accent-note span{font-size:.9rem}.accent-note small{margin-top:10px}.examples-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.example-card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:21px}.example-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}.case-label{font-size:.76rem;letter-spacing:.08em;color:var(--muted);font-weight:800;text-transform:uppercase}.pill,.track-badge{display:inline-block;padding:4px 8px;border-radius:99px;font-size:.72rem;font-weight:800}.pill{background:var(--teal-soft);color:var(--teal)}.label{font-size:.72rem;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);font-weight:800;margin-bottom:5px}.example-source p,.example-target p{font-size:.94rem;line-height:1.72}.example-source{background:#f8f6ef;border-radius:12px;padding:14px}.example-target{background:#eef7f5;border-radius:12px;padding:14px}.example-target small{display:block;font-size:.77rem}.example-arrow{text-align:center;color:var(--coral);font-weight:900;height:20px}.image-example-card{background:linear-gradient(135deg,#f5eeff,#fff);border:1px solid #e2d6f6;border-radius:18px;padding:24px;max-width:700px}.image-example-card h3{margin:7px 0 12px}.image-example-card p{background:#fff;border-radius:12px;padding:14px;color:#4d5961}.image-example-card small{display:block}.table-section{padding-top:35px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:15px;background:var(--card)}table{width:100%;border-collapse:collapse;min-width:750px}th,td{padding:13px 14px;text-align:left;border-bottom:1px solid #e8eeee;vertical-align:top;font-size:.84rem}th{background:#f1f6f5;color:#52616a;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}tr:last-child td{border-bottom:0}td small{display:block;font-size:.73rem;line-height:1.4;margin-top:2px}.limitations{padding-top:20px}.limit-list{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.limit-list div{padding:16px;border-radius:12px;background:var(--card);border:1px solid var(--line)}.limit-list b{display:block;margin-bottom:4px}.limit-list span{display:block;color:var(--muted);font-size:.84rem;line-height:1.55}.reproducibility{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}.reproducibility pre{margin:0;background:#1e2a34;color:#e5f4f0;border-radius:14px;padding:20px;overflow:auto;font-size:.85rem}.reproducibility code{font-family:"SFMono-Regular",Consolas,monospace}.method-flow{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;padding-top:18px}.method-flow article{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px}.method-flow article>span{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:var(--teal);color:#fff;font-weight:850;margin-bottom:18px}.method-flow h2{font-size:1.25rem!important;margin-bottom:7px!important}.method-flow p{color:#52616a;font-size:.93rem}.big-number{border-radius:22px;background:var(--ink);color:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:240px}.big-number strong{font-size:6rem;line-height:1;color:#85d5c3;letter-spacing:-.08em}.big-number span{font-size:1.05rem;font-weight:800}.big-number small{margin-top:4px;color:#c5d4d4}.definition-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.definition{padding:20px;border-left:4px solid var(--teal);background:var(--card);border-radius:0 14px 14px 0}.definition h3{margin:0 0 5px}.definition p{color:#52616a;font-size:.9rem}.download-grid,.license-list{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding-top:18px}.download-card,.license-list article{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:23px}.download-card h2,.license-list h2{font-size:1.25rem!important;margin-top:9px!important}.download-card p,.license-list p{color:#52616a;font-size:.92rem}.download-card .button{margin-top:8px}.download-status,.license-state{display:inline-block;padding:4px 8px;border-radius:99px;font-size:.7rem;font-weight:850;letter-spacing:.08em}.ready .download-status{background:var(--teal-soft);color:var(--teal)}.pending .download-status,.license-state{background:var(--amber-soft);color:var(--amber)}.download-card.pending{background:#fffdf7}.license-list{grid-template-columns:repeat(2,1fr)}.license-list article{min-height:190px}.results-toolbar{display:grid;grid-template-columns:minmax(250px,1fr) 150px 190px auto;gap:12px;align-items:end;padding-top:18px;padding-bottom:24px}.results-toolbar label{display:flex;flex-direction:column;gap:5px;font-size:.75rem;color:var(--muted);font-weight:800}.results-toolbar input,.results-toolbar select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 11px;min-height:43px}.result-count{font-size:.82rem;color:var(--muted);padding:10px 0;white-space:nowrap}.results-table-section{padding-top:0}.results-table-section table{min-width:1250px}.case-cell strong{display:block}.case-cell small{color:var(--muted)}.track-badge.text{background:var(--teal-soft);color:var(--teal)}.track-badge.image{background:var(--coral-soft);color:var(--coral)}.source-cell{max-width:250px}.target-cell{max-width:300px}.source-cell details,.target-cell details{line-height:1.5}.source-cell summary,.target-cell summary{cursor:pointer;color:#465963}.source-cell p,.target-cell p{white-space:pre-wrap;margin:.5rem 0 0;color:#52616a;max-width:400px}.score-cell strong{display:block}.score-cell details{margin-top:6px}.score-cell summary{color:var(--teal);cursor:pointer;font-size:.75rem}.score-cell details p{font-size:.73rem;color:var(--muted);max-width:220px}.time-cell{white-space:nowrap}.time-cell small{display:inline;margin-left:3px}.excluded-row{background:#fffaf0}.excluded-row .score-cell{color:var(--amber)}
.site-footer{border-top:1px solid var(--line);margin-top:40px;background:#f1f5f4}.footer-inner{max-width:var(--max);padding:32px 24px 14px;margin:auto;display:flex;justify-content:space-between;gap:20px;font-size:.86rem}.footer-links{display:flex;gap:17px}.footer-links a{color:var(--teal);text-decoration:none}.footer-note{max-width:var(--max);padding:0 24px 30px;margin:auto;color:var(--muted);font-size:.76rem}
@media (max-width:900px){.header-inner{padding:0 18px}.nav-toggle{display:block;margin-left:auto}.site-nav{display:none;position:absolute;top:74px;left:0;right:0;background:var(--paper);padding:10px 18px 16px;border-bottom:1px solid var(--line);box-shadow:0 12px 20px rgba(27,45,49,.06);flex-direction:column;align-items:stretch}.site-nav.open{display:flex}.nav-link{padding:11px 12px}.hero{grid-template-columns:1fr;gap:22px;padding-top:64px;min-height:0}.hero-aside{min-height:280px}.stats-grid,.metrics-row{grid-template-columns:repeat(2,1fr)}.finding-grid,.limit-list{grid-template-columns:1fr}.models-grid{grid-template-columns:1fr}.split-panel,.reproducibility{grid-template-columns:1fr;gap:26px}.download-grid{grid-template-columns:1fr}.method-flow{grid-template-columns:1fr}.definition-grid{grid-template-columns:1fr}.results-toolbar{grid-template-columns:1fr 1fr}.results-toolbar .search-field{grid-column:1/-1}.result-count{grid-column:1/-1}.section-heading{align-items:start;flex-direction:column;gap:8px}}
@media (max-width:560px){.section-shell{padding:44px 18px}.header-inner{min-height:66px}.site-nav{top:66px}.brand{font-size:.9rem}.brand-mark{width:32px;height:32px}.hero h1,.page-hero h1{font-size:2.55rem}.hero-lede{font-size:1rem}.hero-aside{min-height:245px}.orbit-one{width:250px;height:170px}.orbit-two{width:180px;height:270px}.hero-orbit-card{width:175px;height:175px}.hero-orbit-card strong{font-size:3rem}.stats-grid,.metrics-row{grid-template-columns:1fr}.metrics-row{padding-top:0}.examples-grid{grid-template-columns:1fr}.footer-inner{flex-direction:column;padding:26px 18px 10px}.footer-note{padding:0 18px 25px}.footer-links{flex-wrap:wrap}.results-toolbar{grid-template-columns:1fr}.results-toolbar .search-field{grid-column:auto}.result-count{grid-column:auto}.big-number{min-height:190px}.limit-list{grid-template-columns:1fr}.section-heading h2,.section-shell h2{font-size:1.9rem}}
.hero{padding-top:48px;padding-bottom:42px;min-height:0}.hero h1{font-size:clamp(2rem,4vw,3.4rem)}.hero-aside{min-height:250px}.orbit-one{width:245px;height:165px}.orbit-two{width:180px;height:260px}.hero-orbit-card{width:175px;height:175px}.hero-orbit-card strong{font-size:3rem}
.hero-result-panel{max-width:360px}
@media (max-width:900px){.overview-legend{justify-content:flex-start}.source-card-grid{grid-template-columns:repeat(2,1fr)}.runtime-grid{grid-template-columns:1fr}.runtime-facts>div{grid-template-columns:145px 1fr auto}}
@media (max-width:560px){.hero-result-panel{width:100%;padding:16px}.hero-result-row{padding:7px 0}.hero-result-row strong{font-size:1.25rem}.overview-section{padding-left:0;padding-right:0}.overview-section>.section-heading{padding-left:18px;padding-right:18px}.overview-chart{border-radius:0;border-left:0;border-right:0;padding-left:8px;padding-right:8px}.overview-row{grid-template-columns:150px minmax(410px,1fr) 205px;gap:10px;min-width:800px;padding:10px 6px}.overview-line{grid-template-columns:40px 1fr 56px}.source-card-grid{grid-template-columns:1fr}.source-card{min-height:0}.runtime-facts>div{grid-template-columns:1fr;gap:2px}.runtime-facts span,.runtime-facts strong{text-align:left}.quick-compare-section{padding-left:0;padding-right:0}.quick-compare-section>.section-heading{padding-left:18px;padding-right:18px}.quick-compare-wrap{border-radius:0;border-left:0;border-right:0}.models-grid{gap:10px}.model-card{padding:17px}.table-wrap{max-width:100%;-webkit-overflow-scrolling:touch}}
.overview-section{padding-top:24px;padding-bottom:36px}.overview-chart{overflow-x:auto;border:1px solid var(--line);border-radius:16px;background:var(--card);padding:10px}.overview-row{display:grid;grid-template-columns:185px minmax(440px,1fr) 235px;gap:18px;align-items:center;min-width:900px;padding:12px 10px;border-bottom:1px solid #e8eeee}.overview-row:last-child{border-bottom:0}.overview-name a{font-weight:800;text-decoration:none}.overview-name a:hover{color:var(--teal)}.overview-line{display:grid;grid-template-columns:48px 1fr 58px;align-items:center;gap:8px;margin:4px 0}.overview-key{font-size:.73rem;font-weight:800}.text-key{color:var(--teal)}.image-key{color:var(--coral)}.overview-track{height:9px;border-radius:99px;background:#edf1f1;overflow:hidden}.overview-fill{display:block;height:100%;border-radius:inherit}.text-fill{background:linear-gradient(90deg,#0a8d86,#48b8a4)}.image-fill{background:linear-gradient(90deg,#d86b52,#ecaa8d)}.overview-line b{font-size:.82rem;text-align:right}.overview-speed{padding-left:4px}.speed-line{display:grid;grid-template-columns:92px 1fr;gap:8px;align-items:center;font-size:.72rem;color:var(--muted);margin:6px 0}.speed-track{height:7px;border-radius:99px;background:linear-gradient(90deg,#d9f2ee 0%,#edf1f1 100%);position:relative}.speed-track:before,.speed-track:after{position:absolute;top:9px;font-size:.6rem;color:#94a0a7}.speed-track:before{content:"0s";left:0}.speed-track:after{content:"6s";right:0}.speed-track i{position:absolute;top:-3px;width:13px;height:13px;border-radius:50%;background:var(--purple);border:2px solid #fff;box-shadow:0 0 0 1px #bca9dc;transform:translateX(-50%)}.overview-legend{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:10px;font-size:.78rem;color:var(--muted)}.overview-legend>span{white-space:nowrap}.speed-legend i{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--purple);margin-right:4px}.runtime-section{padding-top:28px;padding-bottom:28px}.runtime-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.runtime-facts,.runtime-settings{border:1px solid var(--line);border-radius:15px;background:var(--card);padding:18px}.runtime-facts>div{display:grid;grid-template-columns:170px 1fr auto;gap:10px;align-items:baseline;padding:8px 0;border-bottom:1px solid #edf1f1;font-size:.82rem}.runtime-facts>div:last-child{border-bottom:0}.runtime-facts b{color:var(--muted)}.runtime-facts code,.runtime-settings code{font-family:"SFMono-Regular",Consolas,monospace;font-size:.73rem;overflow-wrap:anywhere}.runtime-facts span,.runtime-facts strong{font-weight:800;text-align:right}.runtime-settings>b{display:block;margin-bottom:8px}.runtime-settings p{font-size:.84rem;color:#52616a}.runtime-note{background:var(--amber-soft);padding:10px;border-radius:9px}.sha-details{margin-top:12px;border:1px dashed #bccbcb;border-radius:11px;padding:10px 13px;background:#fbfdfc}.sha-details summary{cursor:pointer;color:var(--teal);font-weight:800;font-size:.82rem}.sha-details div{display:grid;gap:6px;margin-top:9px}.sha-details code{font-family:"SFMono-Regular",Consolas,monospace;font-size:.7rem;overflow-wrap:anywhere}.source-card-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.source-card{border:1px solid var(--line);border-radius:15px;background:var(--card);padding:17px;min-height:188px}.source-card h3{font-size:1rem;margin:7px 0}.source-card p{font-size:.82rem;color:#52616a;min-height:44px}.source-card small{display:block;font-size:.72rem;line-height:1.45;margin-top:10px}.source-links{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.source-links .text-link{font-size:.77rem}.source-condition{color:var(--muted)}
 .hero-result-panel{width:min(100%,360px);padding:22px 22px 18px;border:1px solid #b9dfd9;border-radius:18px;background:linear-gradient(145deg,#fff,#eaf7f4);box-shadow:var(--shadow)}.hero-result-row{display:flex;justify-content:space-between;align-items:baseline;padding:10px 0;border-bottom:1px solid #d7ebe8;font-size:.94rem;color:#52616a}.hero-result-row strong{font-size:1.55rem;color:var(--teal);letter-spacing:-.04em}.hero-result-note{font-size:.76rem;color:var(--muted);margin-top:12px;line-height:1.45}
 @media (max-width:900px){.runtime-grid{grid-template-columns:1fr}.source-card-grid{grid-template-columns:repeat(2,1fr)}}
 @media (max-width:560px){.hero-result-panel{width:100%;padding:16px}.hero-result-row{padding:7px 0}.hero-result-row strong{font-size:1.25rem}.overview-section{padding-left:0;padding-right:0}.overview-section>.section-heading{padding-left:18px;padding-right:18px}.overview-chart{border-radius:0;border-left:0;border-right:0;padding-left:8px;padding-right:8px}.overview-row{grid-template-columns:150px minmax(410px,1fr) 205px;gap:10px;min-width:800px;padding:10px 6px}.overview-line{grid-template-columns:40px 1fr 56px}.source-card-grid{grid-template-columns:1fr}.source-card{min-height:0}.runtime-facts>div{grid-template-columns:1fr;gap:2px}.runtime-facts span,.runtime-facts strong{text-align:left}.quick-compare-section{padding-left:0;padding-right:0}.quick-compare-section>.section-heading{padding-left:18px;padding-right:18px}.quick-compare-wrap{border-radius:0;border-left:0;border-right:0}.models-grid{gap:10px}.model-card{padding:17px}.table-wrap{max-width:100%;-webkit-overflow-scrolling:touch}}
 @media (max-width:560px){.overview-chart{overflow-x:visible;padding:8px 6px}.overview-row{grid-template-columns:105px minmax(0,1fr);gap:5px;min-width:0;padding:9px 3px}.overview-name{font-size:.76rem;line-height:1.3;overflow-wrap:anywhere}.overview-scores{min-width:0}.overview-line{grid-template-columns:34px minmax(0,1fr) 52px;gap:5px}.overview-line b{font-size:.73rem}.overview-key{font-size:.68rem}.overview-track{height:7px}.overview-speed{grid-column:2;display:flex;flex-wrap:wrap;gap:3px 10px;padding-left:0}.speed-line{display:block;font-size:.68rem;margin:1px 0 0}.speed-track{display:none}.overview-legend{font-size:.7rem;gap:6px}.overview-legend>span{white-space:normal}.speed-legend i{width:7px;height:7px}}
.judge-explainer{padding-top:24px;padding-bottom:30px;scroll-margin-top:84px}.judge-explainer-panel{border:1px solid #cfe3df;border-radius:18px;background:linear-gradient(145deg,#fff,#f0f9f7);padding:22px 24px;box-shadow:0 10px 28px rgba(27,45,49,.05)}.judge-explainer h2{font-size:clamp(1.55rem,2.5vw,2.2rem)!important;max-width:820px}.judge-explainer-panel>p{max-width:900px;color:#52616a}.rubric-mini-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:18px 0 14px}.rubric-mini-item{border:1px solid #dcebe8;border-radius:12px;background:#fff;padding:12px 13px;min-height:94px}.rubric-mini-item b{display:block;color:var(--teal);font-size:.94rem;line-height:1.35;margin-bottom:5px}.rubric-mini-item span{display:block;color:#52616a;font-size:.87rem;line-height:1.45}.judge-icc-mini{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;border-top:1px solid #cfe3df;padding-top:13px;font-size:.9rem;color:#52616a}.judge-icc-mini span{line-height:1.55}.judge-icc-mini small{color:var(--muted);line-height:1.45}.judge-icc-mini .text-link{font-size:.78rem}.hero-result-note{display:grid;gap:4px}.hero-result-note a{color:var(--teal);font-weight:800;text-decoration:none}.hero-result-note a:hover{text-decoration:underline}.rubric-detail{padding-top:30px;scroll-margin-top:84px}.rubric-table-wrap{overflow:auto;border:1px solid var(--line);border-radius:15px;background:var(--card)}.rubric-table{min-width:760px}.rubric-table th,.rubric-table td{font-size:.82rem}.icc-detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}.icc-detail-grid article{border:1px solid var(--line);border-radius:15px;background:var(--card);padding:18px}.icc-detail-grid h3{font-size:1.05rem;margin:0 0 10px}.icc-detail-grid p{font-size:.86rem;color:#52616a}.icc-detail-grid p b{color:var(--teal)}.rubric-note{margin:17px 0;color:#52616a;font-size:.84rem}.rubric-sources{gap:14px}
@media (max-width:900px){.rubric-mini-grid{grid-template-columns:repeat(2,1fr)}.judge-icc-mini{grid-template-columns:1fr auto}.icc-detail-grid{grid-template-columns:1fr}}
@media (max-width:560px){.judge-explainer-panel{padding:18px}.rubric-mini-grid{grid-template-columns:1fr 1fr;gap:7px}.rubric-mini-item{padding:10px;min-height:0}.rubric-mini-item b{font-size:.88rem}.rubric-mini-item span{font-size:.82rem}.judge-icc-mini{grid-template-columns:1fr;gap:8px}.judge-icc-mini .text-link{white-space:normal}.rubric-detail{padding-top:20px}.icc-detail-grid{grid-template-columns:1fr}}
"""


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
    if editorial_path.is_file():
        loaded_editorial = read_json(editorial_path)
        if isinstance(loaded_editorial, dict):
            editorial_sections = loaded_editorial
    verification_meta = load_google_verification(docs)

    def write_page(relative_path: str, rendered: str) -> None:
        key = relative_path.replace("\\", "/")
        write_text(docs / relative_path, apply_editorial_sections(rendered, key, editorial_sections))

    write_text(docs / "assets" / "style.css", STYLE_CSS.strip() + "\n")
    write_text(docs / "assets" / "site.js", SITE_JS.strip() + "\n")

    # Machine-readable public data is intentionally a small, explicit projection.
    public_manifest = {
        "schema_version": "1.0",
        "title": "心理健康翻譯基準：安全 allow-list 結果",
        "generated_by": "build_site.py",
        "rows": dataset["rows"],
    }
    write_text(docs / "data" / "translation_results.json", json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n")

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
    write_page("index-en.html", page_html("Seven local models tested · Psychological health English–Traditional Chinese", "home", home_body_en(dataset), lang="en", self_href="index-en.html", switch_href="index.html", verification_meta=verification_meta))
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
