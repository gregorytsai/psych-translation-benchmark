#!/usr/bin/env python3
"""Static final checks for the public psych-translation-benchmark release.

This checker is intentionally offline and standard-library only.  It validates
the generated bilingual pages, public data shape/parity, chart values, links,
optional download placeholders, and obvious secret/private-path leaks.  It
does not call models, browsers, or network services.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import posixpath
import re
import sys
import zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
TEXT_SUFFIXES = {
    ".html", ".htm", ".css", ".js", ".json", ".jsonl", ".csv", ".md", ".py",
    ".ps1", ".sh", ".txt", ".yaml", ".yml", ".toml", ".ini", ".jinja",
}
OPTIONAL_DOWNLOADS = {
    "downloads/complete-results.zip",
    "downloads/benchmark-code.zip",
    "downloads/materials.zip",
}
EXPECTED_HTML_COUNT = 24
EXPECTED_MODEL_COUNT = 7
EXPECTED_CASE_COUNT = 34
PUBLIC_BASE_URL = "https://gregorytsai.github.io/psych-translation-benchmark/"
PUBLIC_BASE = urlsplit(PUBLIC_BASE_URL)
PUBLIC_BASE_PATH = PUBLIC_BASE.path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def num(value, default=None):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def same_num(a, b, tolerance=1e-8) -> bool:
    x, y = num(a), num(b)
    return x is not None and y is not None and abs(x - y) <= tolerance


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lang = None
        self.ids: set[str] = set()
        self.names: set[str] = set()
        self.links: list[tuple[str, str, str, dict[str, str]]] = []
        self.alternates: list[tuple[str, str]] = []
        self.canonical: list[str] = []
        self.language_switches: list[tuple[str, dict[str, str]]] = []
        self.result_rows = 0

    @staticmethod
    def attrs_dict(attrs):
        return {k: (v or "") for k, v in attrs}

    def handle_starttag(self, tag, attrs):
        attrs_d = self.attrs_dict(attrs)
        if tag == "html":
            self.lang = attrs_d.get("lang")
        if attrs_d.get("id"):
            self.ids.add(attrs_d["id"])
        if attrs_d.get("name"):
            self.names.add(attrs_d["name"])
        for attr in ("href", "src"):
            if attrs_d.get(attr):
                self.links.append((tag, attr, attrs_d[attr], attrs_d))
        if tag == "link" and "alternate" in attrs_d.get("rel", "").split():
            if attrs_d.get("hreflang") and attrs_d.get("href"):
                self.alternates.append((attrs_d["hreflang"], attrs_d["href"]))
        if tag == "link" and "canonical" in attrs_d.get("rel", "").split() and attrs_d.get("href"):
            self.canonical.append(attrs_d["href"])
        if tag == "a" and "language-switch" in attrs_d.get("class", "").split():
            self.language_switches.append((attrs_d.get("href", ""), attrs_d))
        if tag == "tr" and "result-row" in attrs_d.get("class", "").split():
            self.result_rows += 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def parse_pages() -> tuple[list[Path], dict[Path, PageParser]]:
    pages = sorted(DOCS.rglob("*.html"))
    parsed = {}
    for path in pages:
        parser = PageParser()
        parser.feed(read_text(path))
        parsed[path] = parser
    return pages, parsed


class Report:
    def __init__(self):
        self.failures: list[dict] = []
        self.warnings: list[dict] = []
        self.info: list[dict] = []

    def fail(self, code: str, message: str, path: Path | None = None):
        item = {"code": code, "message": message}
        if path is not None:
            item["path"] = rel(path)
        self.failures.append(item)

    def warn(self, code: str, message: str, path: Path | None = None):
        item = {"code": code, "message": message}
        if path is not None:
            item["path"] = rel(path)
        self.warnings.append(item)

    def note(self, code: str, message: str, path: Path | None = None):
        item = {"code": code, "message": message}
        if path is not None:
            item["path"] = rel(path)
        self.info.append(item)


def paired_page(path: Path) -> Path:
    if path.name.endswith("-en.html"):
        return path.with_name(path.name[:-8] + ".html")
    return path.with_name(path.stem + "-en.html")


def expected_language(path: Path) -> str:
    return "en" if path.name.endswith("-en.html") else "zh-Hant-TW"


def page_relative(path: Path) -> str:
    return path.resolve().relative_to(DOCS.resolve()).as_posix()


def normalize_repo_url(value: str, page: Path | None = None, require_absolute: bool = False) -> str | None:
    """Map a relative or exact public-base URL to a docs-relative path.

    Absolute SEO URLs are accepted only for the configured GitHub Pages site;
    an arbitrary external domain is never treated as an internal page.
    """
    parsed = urlsplit(value.strip())
    if parsed.query or parsed.fragment:
        return None
    if parsed.scheme or parsed.netloc:
        if parsed.scheme.lower() != PUBLIC_BASE.scheme or parsed.netloc.lower() != PUBLIC_BASE.netloc.lower():
            return None
        if not parsed.path.startswith(PUBLIC_BASE_PATH):
            return None
        relative = parsed.path[len(PUBLIC_BASE_PATH):]
    else:
        if require_absolute or page is None:
            return None
        relative = posixpath.join(posixpath.dirname(page_relative(page)), parsed.path)
    relative = posixpath.normpath(relative or "index.html").lstrip("/")
    if relative in {"", "."}:
        relative = "index.html"
    if relative == ".." or relative.startswith("../"):
        return None
    return relative


def normalize_href_path(value: str) -> str:
    return Path(unquote(urlsplit(value).path)).as_posix()


def validate_html_structure(report: Report, pages: list[Path], parsed: dict[Path, PageParser]):
    if len(pages) != EXPECTED_HTML_COUNT:
        report.fail("html_count", f"expected {EXPECTED_HTML_COUNT} HTML files, found {len(pages)}")

    for path in pages:
        page = parsed[path]
        expected_lang = expected_language(path)
        if page.lang != expected_lang:
            report.fail("html_lang", f"expected lang={expected_lang!r}, found {page.lang!r}", path)

        pair = paired_page(path)
        if not pair.exists():
            report.fail("language_pair_missing", f"missing paired page {rel(pair)}", path)
            continue
        expected_page = page_relative(path)
        if len(page.canonical) != 1:
            report.fail("canonical_count", f"expected one canonical URL, found {len(page.canonical)}", path)
        else:
            canonical_path = normalize_repo_url(page.canonical[0], path, require_absolute=True)
            if canonical_path is None:
                report.fail("canonical_url_scope", f"canonical must use exact public base {PUBLIC_BASE_URL!r}", path)
            elif canonical_path != expected_page:
                report.fail("canonical_target", f"canonical resolves to {canonical_path!r}, expected {expected_page!r}", path)
            elif not (DOCS / canonical_path).exists():
                report.fail("canonical_missing", f"canonical target missing: {canonical_path!r}", path)
        expected = {
            "zh-Hant": page_relative(path if expected_lang != "en" else pair),
            "en": page_relative(pair if expected_lang != "en" else path),
            "x-default": page_relative(path if expected_lang != "en" else pair),
        }
        seen = {}
        for hreflang, href in page.alternates:
            normalized = normalize_repo_url(href, path, require_absolute=True)
            if normalized is None:
                report.fail("hreflang_url_scope", f"{hreflang} URL must use exact public base {PUBLIC_BASE_URL!r}: {href!r}", path)
                continue
            seen[hreflang] = normalized
            if not (DOCS / normalized).exists():
                report.fail("hreflang_missing", f"{hreflang} target missing: {normalized!r}", path)
        for hreflang, expected_href in expected.items():
            got = seen.get(hreflang)
            if got != expected_href:
                report.fail(
                    "hreflang",
                    f"{hreflang} should point to {expected_href!r}, found {got!r}",
                    path,
                )
        switch_targets = [normalize_href_path(href) for href, _ in page.language_switches if href]
        expected_switch = normalize_href_path(
            Path(pair).relative_to(path.parent).as_posix()
            if pair.parent == path.parent else
            posixpath.relpath(pair, path.parent)
        )
        if expected_switch not in switch_targets:
            report.fail(
                "language_switch",
                f"language-switch should target {expected_switch!r}, found {switch_targets!r}",
                path,
            )
        expected_switch_lang = "en" if expected_lang != "en" else "zh-Hant-TW"
        if not any(
            normalize_href_path(href) == expected_switch and attrs.get("lang") == expected_switch_lang
            for href, attrs in page.language_switches
        ):
            report.fail(
                "language_switch_lang",
                f"language-switch to {expected_switch!r} should declare lang={expected_switch_lang!r}",
                path,
            )

    if pages:
        report.note("html_pairs", f"validated {len(pages) // 2} Chinese/English page pairs")


def validate_sitemap(report: Report, pages: list[Path]):
    path = DOCS / "sitemap.xml"
    if not path.exists():
        report.fail("sitemap_missing", "sitemap.xml is missing", path)
        return
    try:
        root = ET.fromstring(read_text(path))
    except ET.ParseError as exc:
        report.fail("sitemap_parse", str(exc), path)
        return
    locs = [node.text.strip() for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "loc" and node.text]
    expected = {page_relative(page) for page in pages}
    normalized = set()
    for loc in locs:
        target = normalize_repo_url(loc, require_absolute=True)
        if target is None:
            report.fail("sitemap_url_scope", f"sitemap URL must use exact public base: {loc!r}", path)
            continue
        normalized.add(target)
        if not (DOCS / target).exists():
            report.fail("sitemap_target_missing", f"sitemap target missing: {target!r}", path)
    if len(locs) != EXPECTED_HTML_COUNT or len(set(locs)) != EXPECTED_HTML_COUNT:
        report.fail("sitemap_count", f"expected {EXPECTED_HTML_COUNT} unique URLs, found {len(locs)} rows/{len(set(locs))} unique", path)
    if normalized != expected:
        report.fail("sitemap_pages", f"sitemap/pages mismatch: sitemap-only={sorted(normalized - expected)}, page-only={sorted(expected - normalized)}", path)
    else:
        report.note("sitemap", f"validated {len(normalized)} exact public-base page URLs")


def is_external(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme or parsed.netloc)


def validate_links(report: Report, pages: list[Path], parsed: dict[Path, PageParser]):
    page_by_resolved = {p.resolve(): p for p in pages}
    for path in pages:
        page = parsed[path]
        for tag, attr, value, _attrs in page.links:
            value = value.strip()
            if not value or value.startswith(("data:", "javascript:", "mailto:", "tel:")):
                continue
            parsed_url = urlsplit(value)
            if parsed_url.scheme or parsed_url.netloc:
                if parsed_url.scheme in {"http", "https"}:
                    continue
                report.warn("unsupported_external_scheme", f"skipped {value!r}", path)
                continue

            path_part = unquote(parsed_url.path)
            fragment = unquote(parsed_url.fragment)
            if path_part:
                target = (path.parent / Path(path_part)).resolve()
            else:
                target = path.resolve()
            try:
                target.relative_to(DOCS.resolve())
            except ValueError:
                report.fail("link_escapes_docs", f"{attr}={value!r} resolves outside docs/", path)
                continue

            if not target.exists():
                normalized = Path(path_part).as_posix()
                if normalized in OPTIONAL_DOWNLOADS:
                    # validate_downloads emits one consolidated warning per
                    # optional ZIP instead of duplicating it for both languages.
                    continue
                else:
                    report.fail("broken_internal_link", f"{attr}={value!r} target missing", path)
                continue

            if fragment:
                target_page = page_by_resolved.get(target)
                target_parser = parsed.get(target_page) if target_page else None
                if target_parser is None:
                    report.fail("anchor_target_not_html", f"fragment target is not an HTML page: {value!r}", path)
                elif fragment not in target_parser.ids and fragment not in target_parser.names:
                    report.fail("broken_anchor", f"fragment #{fragment} not found in {rel(target)}", path)


def validate_downloads(report: Report):
    for relative in sorted(OPTIONAL_DOWNLOADS):
        path = DOCS / relative
        if not path.exists():
            report.warn("optional_download_missing", f"expected release ZIP is not present yet: {relative}", path)
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    report.fail("invalid_zip", f"CRC failure in member {bad!r}", path)
        except zipfile.BadZipFile as exc:
            report.fail("invalid_zip", str(exc), path)


def source_text(row: dict) -> str:
    if row.get("source_kind") == "image":
        return row.get("image_description_en", "") or ""
    return row.get("source_en", "") or ""


def comparison_source(row: dict) -> str:
    if row.get("source_kind") == "image":
        return row.get("image_ground_truth_en", "") or ""
    return row.get("source_en", "") or ""


def validate_translation_data(report: Report):
    json_path = DATA / "translation_results.json"
    csv_path = DATA / "translation_comparison.csv"
    payload = read_json(json_path, {}) or {}
    jrows = payload.get("rows", []) if isinstance(payload, dict) else []
    crows = read_csv(csv_path)
    if len(jrows) != 238:
        report.fail("json_translation_count", f"translation_results.json has {len(jrows)} rows", json_path)
    if len(crows) != 238:
        report.fail("csv_translation_count", f"translation_comparison.csv has {len(crows)} rows", csv_path)

    def keyed(rows, material_field):
        result = {}
        duplicates = []
        for row in rows:
            key = (row.get("model_id", ""), row.get(material_field, ""))
            if key in result:
                duplicates.append(key)
            result[key] = row
        return result, duplicates

    jby, jdupes = keyed(jrows, "material_id")
    cby, cdupes = keyed(crows, "material_id")
    if jdupes:
        report.fail("json_duplicate_keys", f"{len(jdupes)} duplicate model/material keys", json_path)
    if cdupes:
        report.fail("csv_duplicate_keys", f"{len(cdupes)} duplicate model/material keys", csv_path)

    if set(jby) != set(cby):
        report.fail(
            "json_csv_key_mismatch",
            f"JSON-only={len(set(jby) - set(cby))}, CSV-only={len(set(cby) - set(jby))}",
        )
    mismatches = []
    for key in sorted(set(jby) & set(cby)):
        j, c = jby[key], cby[key]
        checks = {
            "source_id": (j.get("source_id", ""), c.get("source_id", "")),
            "source_kind": (j.get("source_kind", ""), c.get("source_kind", "")),
            "source": (source_text(j), comparison_source(c)),
            "translation": (j.get("target_zh_tw", ""), c.get("target_zh_tw", "")),
            "status": (j.get("translation_status", ""), c.get("translation_status", "")),
        }
        for field, (a, b) in checks.items():
            if a != b:
                mismatches.append({"key": key, "field": field})
        if not same_num(j.get("translation_elapsed_seconds"), c.get("translation_elapsed_seconds"), 1e-9):
            mismatches.append({"key": key, "field": "translation_elapsed_seconds"})
    if mismatches:
        report.fail("json_csv_translation_parity", f"{len(mismatches)} field mismatches; first={mismatches[0]}")
    else:
        report.note("json_csv_translation_parity", "238 JSON rows exactly match CSV source/target/status/elapsed fields")

    models = sorted({r.get("model_id", "") for r in crows})
    cases = sorted({r.get("material_id", "") for r in crows})
    if len(models) != EXPECTED_MODEL_COUNT:
        report.fail("model_count", f"expected {EXPECTED_MODEL_COUNT} models, found {len(models)}")
    if len(cases) != EXPECTED_CASE_COUNT:
        report.fail("case_count", f"expected {EXPECTED_CASE_COUNT} materials, found {len(cases)}")

    model_counts = Counter(r.get("model_id", "") for r in crows)
    case_counts = Counter(r.get("material_id", "") for r in crows)
    bad_models = {m: n for m, n in model_counts.items() if n != EXPECTED_CASE_COUNT}
    bad_cases = {c: n for c, n in case_counts.items() if n != EXPECTED_MODEL_COUNT}
    if bad_models:
        report.fail("model_case_matrix", f"models not 34 rows each: {bad_models}")
    if bad_cases:
        report.fail("case_model_matrix", f"materials not 7 models each: {bad_cases}")

    text_cases = {r.get("material_id") for r in crows if r.get("source_kind") == "text"}
    image_cases = {r.get("material_id") for r in crows if r.get("source_kind") == "image"}
    if len(text_cases) != 30 or len(image_cases) != 4:
        report.fail("track_shape", f"expected 30 text + 4 image cases, found {len(text_cases)} + {len(image_cases)}")
    bad_status = Counter(r.get("translation_status", "") for r in crows if r.get("translation_status", "") != "ok")
    if bad_status:
        report.fail("translation_status", f"non-ok statuses: {dict(bad_status)}")


def validate_numeric_charts(report: Report, pages: list[Path]):
    common = read_csv(DATA / "common_case_model_summary.csv")
    generation = read_csv(DATA / "translation_generation_summary.csv")
    if len(common) != 14:
        report.fail("common_summary_shape", f"expected 14 model/track rows, found {len(common)}")
    if len(generation) != 14:
        report.fail("generation_summary_shape", f"expected 14 model/track rows, found {len(generation)}")

    common_by_model = {
        (row.get("model_id"), row.get("track")): num(row.get("common_case_panel_mean"))
        for row in common
    }
    speed_by_model = {
        (row.get("model_id"), row.get("track")): num(row.get("translation_elapsed_median_s"))
        for row in generation
    }
    model_ids = sorted({r.get("model_id") for r in common})

    def overview_snippets(html: str) -> list[str]:
        starts = [m.start() for m in re.finditer(r'<div class="overview-row">', html)]
        snippets = []
        for index, start in enumerate(starts):
            next_start = starts[index + 1] if index + 1 < len(starts) else len(html)
            section_end = html.find("</div></section>", start)
            end = min(next_start, section_end if section_end >= 0 else next_start)
            snippets.append(html[start:end])
        return snippets

    index_pages = [DOCS / "index.html", DOCS / "index-en.html"]
    for path in index_pages:
        if not path.exists():
            report.fail("index_missing", "index page missing", path)
            continue
        html = read_text(path)
        if "238" not in html:
            report.fail("headline_count", "index does not expose 238 translation count", path)
        for row in common:
            value = f"{float(row['common_case_panel_mean']):.3f}"
            if value not in html:
                report.fail("index_metric_missing", f"common score {row['model_id']}/{row['track']}={value} missing", path)
            if f'style="width: {value}%' not in html:
                report.fail("index_chart_width_missing", f"bar width {value}% missing", path)
        for row in generation:
            value = f"{float(row['translation_elapsed_median_s']):.3f}"
            if f"{value} s" not in html:
                report.fail("index_latency_missing", f"median latency {row['model_id']}/{row['track']}={value}s missing", path)

        overview = overview_snippets(html)
        if len(overview) != EXPECTED_MODEL_COUNT:
            report.fail("overview_row_count", f"expected {EXPECTED_MODEL_COUNT} overview rows, found {len(overview)}", path)
        overview_seen = set()
        for snippet in overview:
            model_match = re.search(r'href="models/([^"/]+)\.html"', snippet)
            model_id = model_match.group(1) if model_match else ""
            if model_id.endswith("-en"):
                model_id = model_id[:-3]
            overview_seen.add(model_id)
            scores = [num(x) for x in re.findall(r'<div class="overview-line">.*?<b>([0-9.]+)</b>', snippet, re.S)]
            speeds = [num(x) for x in re.findall(r'<span>(?:文字|圖片|Text|Image) ([0-9.]+)s</span>', snippet)]
            speed_positions = [num(x) for x in re.findall(r'<i style="left:([0-9.]+)%"></i>', snippet)]
            expected_scores = [common_by_model.get((model_id, track)) for track in ("text", "image")]
            expected_speeds = [speed_by_model.get((model_id, track)) for track in ("text", "image")]
            if len(scores) != 2 or any(not same_num(a, b, 0.0006) for a, b in zip(scores, expected_scores)):
                report.fail("overview_scores", f"overview {model_id!r} scores do not match public common means", path)
            if len(speeds) != 2 or any(not same_num(a, b, 0.0006) for a, b in zip(speeds, expected_speeds)):
                report.fail("overview_speeds", f"overview {model_id!r} seconds do not match public generation medians", path)
            expected_positions = [round((speed / 6.0) * 100.0, 2) if speed is not None else None for speed in expected_speeds]
            if len(speed_positions) != 2 or any(not same_num(a, b, 0.011) for a, b in zip(speed_positions, expected_positions)):
                report.fail("overview_speed_positions", f"overview {model_id!r} 0–6s dot positions do not match rendered medians", path)
        if overview_seen != set(model_ids):
            report.fail("overview_models", f"overview model IDs differ: found={sorted(overview_seen)}, expected={model_ids}", path)

    for model_id in model_ids:
        for suffix in ("", "-en"):
            path = DOCS / "models" / f"{model_id}{suffix}.html"
            if not path.exists():
                report.fail("model_page_missing", f"missing model page {path.name}", path)
                continue
            html = read_text(path)
            for row in [r for r in common if r.get("model_id") == model_id]:
                value = f"{float(row['common_case_panel_mean']):.3f}"
                if value not in html:
                    report.fail("model_metric_missing", f"common score {model_id}/{row['track']}={value} missing", path)

    for path in (DOCS / "results.html", DOCS / "results-en.html"):
        if not path.exists():
            report.fail("results_page_missing", "results page missing", path)
            continue
        page = PageParser()
        page.feed(read_text(path))
        if page.result_rows != 238:
            report.fail("results_row_count", f"expected 238 result rows, found {page.result_rows}", path)
    report.note("numeric_chart_fields", "validated single-card bars, 7-row overview dual scores/times, generation medians, and 238 result rows")


def scan_public_text(report: Report):
    allowed_synthetic_keys = {"sk-test", "sk-llama-local", "sk-litellm-change-me"}
    secret_patterns = [
        re.compile(r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}\b"),
        re.compile(r"\b(?:xoxb|xoxp|xoxa|xoxr)-[A-Za-z0-9-]{20,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{15,}\b"),
    ]
    private_patterns = [
        re.compile(r"(?i)(?:[A-Z]:[\\\\/](?:Users|Documents|OneDrive|Performance|Program Files)|/Users/|/home/|/private/var/|/Volumes/)"),
        re.compile(r"(?i)(?:OneDrive\s*-\s*|中心共用資料|translation-review-20261025)"),
        re.compile(r"(?i)\\\\\\\\[A-Za-z0-9_.-]+\\\\"),
    ]
    host_patterns = [
        re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[0-1]))\.\d{1,3}\.\d{1,3}\b"),
        re.compile(r"(?i)\b[A-Za-z0-9-]+\.(?:local|lan|internal|corp|intranet)\b"),
    ]
    allowlisted = Counter()
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules"}
    skip_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".gguf", ".pyc", ".ico", ".woff", ".woff2"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in skip_dirs for part in path.parts):
            continue
        # The scanner's own regex/allowlist strings are diagnostics, not
        # shipped credentials or workstation paths.
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in skip_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "example.invalid" in text:
            allowlisted["example.invalid"] += text.count("example.invalid")
        for synthetic_key in allowed_synthetic_keys:
            if synthetic_key in text:
                allowlisted[synthetic_key] += text.count(synthetic_key)

        # The release boundary explicitly permits generic /benchmark paths;
        # only workstation/user roots are treated as private-path findings.
        for pattern in secret_patterns:
            for match in pattern.finditer(text):
                value = match.group(0)
                if value in allowed_synthetic_keys or value.startswith("sk-test-"):
                    continue
                report.fail("secret_pattern", f"possible credential-like token {value[:12]}…", path)
        for pattern in private_patterns:
            for match in pattern.finditer(text):
                report.fail("private_path", f"possible workstation/private path {match.group(0)!r}", path)
        for pattern in host_patterns:
            for match in pattern.finditer(text):
                report.fail("private_host", f"possible private host/address {match.group(0)!r}", path)
    report.note("allowlisted_fixtures", f"allowed synthetic fixtures: {dict(allowlisted)}")


def run() -> tuple[Report, dict]:
    report = Report()
    pages, parsed = parse_pages()
    validate_html_structure(report, pages, parsed)
    validate_sitemap(report, pages)
    validate_links(report, pages, parsed)
    validate_downloads(report)
    validate_translation_data(report)
    validate_numeric_charts(report, pages)
    scan_public_text(report)
    result = {
        "schema_version": "public-release-validation-v1",
        "root": ".",
        "status": "PASS" if not report.failures else "FAIL",
        "summary": {
            "html_files": len(pages),
            "failures": len(report.failures),
            "warnings": len(report.warnings),
            "notes": len(report.info),
        },
        "failures": report.failures,
        "warnings": report.warnings,
        "notes": report.info,
    }
    return report, result


def main() -> int:
    global ROOT, DOCS, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="public project root")
    parser.add_argument("--json-out", type=Path, default=None, help="optional validation JSON output")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    DOCS = ROOT / "docs"
    DATA = DOCS / "data"
    report, result = run()
    if args.json_out:
        out = args.json_out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

