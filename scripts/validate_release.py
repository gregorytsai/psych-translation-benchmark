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
import hashlib
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
OVERVIEW_METRICS = [
    "total",
    "terminology_locale",
    "semantic_safety",
    "fluency",
    "completeness_format",
]
OVERVIEW_TRACKS = ["both", "text", "image"]
OVERVIEW_SORTS = ["score_desc", "time_asc"]
CRITERION_MAXIMA = {
    "terminology_locale": 30,
    "semantic_safety": 30,
    "fluency": 20,
    "completeness_format": 20,
}
IMAGE_IDS = {"img-01", "img-04", "img-09", "img-10"}
SYNTHETIC_IMAGE_IDS = {"img-01", "img-04"}
WHO_IMAGE_IDS = IMAGE_IDS - SYNTHETIC_IMAGE_IDS
SYNTHETIC_IMAGE_BASENAMES = {
    "img01_synthetic_depression_note.png",
    "img04_synthetic_medication_label.png",
}
WHO_IMAGE_BASENAMES = {
    "img09_original_who_suicide_policy_p1.png",
    "img10_original_who_depression_brochure_p1.png",
}
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def format_trimmed(value, digits: int) -> str:
    """Format a measured value without meaningless trailing zeroes."""
    number = num(value)
    if number is None:
        return "—"
    rendered = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def numeric_token_present(text: str, token: str) -> bool:
    """Find a complete numeric token, not a prefix of a longer value."""
    return bool(re.search(rf"(?<![0-9.]){re.escape(token)}(?![0-9.])", text))


def numeric_unit_present(text: str, token: str, unit: str) -> bool:
    """Find a complete displayed number immediately paired with its unit."""
    return bool(
        re.search(
            rf"(?<![0-9.]){re.escape(token)}(?![0-9.])\s*{re.escape(unit)}\b",
            text,
        )
    )


def criterion_display_precisions(criterion_payload: dict) -> dict[str, int]:
    """Choose the minimum shared score precision that keeps each track distinct.

    The public report starts at one decimal.  If two distinct source means in
    either track would collapse to one displayed value, that criterion uses
    two decimals for every model/track.  Raw and normalized data attributes
    remain independently checked at their stored precision.
    """
    models = (criterion_payload.get("models", {}) or {})
    precisions: dict[str, int] = {}
    for metric in OVERVIEW_METRICS:
        values_by_track: list[list[float]] = []
        for track in ("text", "image"):
            values: list[float] = []
            for model in models.values():
                record = (model.get(track, {}) or {})
                if metric == "total":
                    item = record.get("total", {}) or {}
                else:
                    item = (record.get("dimensions", {}) or {}).get(metric, {}) or {}
                value = num(item.get("normalized"))
                if value is not None:
                    values.append(value)
            values_by_track.append(values)
        digits = 1
        for values in values_by_track:
            rendered_groups: dict[str, set[float]] = {}
            for value in values:
                rendered_groups.setdefault(format_trimmed(value, 1), set()).add(value)
            if any(len(originals) > 1 for originals in rendered_groups.values()):
                digits = 2
                break
        precisions[metric] = digits
    return precisions


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


class OverviewParser(HTMLParser):
    """Small structural parser for the homepage comparison contract.

    The chart and no-JavaScript table deliberately share ``.overview-row``
    data attributes.  Keeping this parser independent from the builder makes
    the release check verify the rendered contract rather than Python source
    implementation details.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str]]] = []
        self.open_rows: list[tuple[int, dict[str, object]]] = []
        self.open_cells: list[tuple[int, dict[str, object]]] = []
        self.rows: list[dict[str, object]] = []
        self.metric_buttons: list[dict[str, str]] = []
        self.track_buttons: list[dict[str, str]] = []
        self.sort_buttons: list[dict[str, str]] = []
        self.ids: Counter[str] = Counter()

    @staticmethod
    def attrs_dict(attrs):
        return {k: (v or "") for k, v in attrs}

    @staticmethod
    def has_class(attrs: dict[str, str], name: str) -> bool:
        return name in attrs.get("class", "").split()

    def _context(self, attrs: dict[str, str]) -> str:
        ancestors = [a for _, a in self.stack] + [attrs]
        if any(self.has_class(a, "overview-table-wrap") for a in ancestors):
            return "table"
        if any(a.get("id") == "overview-chart" for a in ancestors):
            return "chart"
        return ""

    def handle_starttag(self, tag, attrs):
        attrs_d = self.attrs_dict(attrs)
        if attrs_d.get("id"):
            self.ids[attrs_d["id"]] += 1
        if tag == "button":
            if attrs_d.get("data-overview-metric"):
                self.metric_buttons.append(attrs_d)
            if attrs_d.get("data-overview-track"):
                self.track_buttons.append(attrs_d)
            if attrs_d.get("data-overview-sort"):
                self.sort_buttons.append(attrs_d)

        self.stack.append((tag, attrs_d))
        if (self.has_class(attrs_d, "overview-row") or self.has_class(attrs_d, "overview-table-row")) and attrs_d.get("data-model-id"):
            row: dict[str, object] = {
                "tag": tag,
                "context": self._context(attrs_d),
                "attrs": attrs_d,
                "text": "",
                "score_widths": [],
                "speed_positions": [],
                "cells": [],
            }
            self.rows.append(row)
            self.open_rows.append((len(self.stack), row))
        if self.open_rows and tag in {"td", "th"}:
            for _, row in self.open_rows:
                if row.get("context") == "table":
                    cell: dict[str, object] = {"tag": tag, "attrs": attrs_d, "text": ""}
                    row["cells"].append(cell)
                    self.open_cells.append((len(self.stack), cell))
        if self.open_rows and tag == "i":
            style = attrs_d.get("style", "")
            width_match = re.search(r"\bwidth\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", style, re.I)
            left_match = re.search(r"\bleft\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", style, re.I)
            for _, row in self.open_rows:
                if width_match:
                    row["score_widths"].append(float(width_match.group(1)))
                if left_match:
                    row["speed_positions"].append(float(left_match.group(1)))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        if data and self.open_rows:
            for _, row in self.open_rows:
                row["text"] = str(row.get("text", "")) + data
            for _, cell in self.open_cells:
                cell["text"] = str(cell.get("text", "")) + data

    def handle_endtag(self, tag):
        depth = len(self.stack)
        self.open_cells = [(d, cell) for d, cell in self.open_cells if d != depth]
        self.open_rows = [(d, r) for d, r in self.open_rows if d != depth]
        if self.stack:
            self.stack.pop()


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


def parse_jsonl_bytes(report: Report, data: bytes, label: str, path: Path) -> list[dict]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        report.fail("jsonl_encoding", f"{label} is not UTF-8: {exc}", path)
        return []
    rows: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            report.fail("jsonl_invalid", f"{label} line {line_number} is not valid JSON: {exc.msg}", path)
            continue
        if not isinstance(value, dict):
            report.fail("jsonl_record_type", f"{label} line {line_number} is not an object", path)
            continue
        rows.append(value)
    return rows


def safe_material_target(materials: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    candidate = Path(relative.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    target = (materials / candidate).resolve()
    try:
        target.relative_to(materials.resolve())
    except ValueError:
        return None
    return target


def official_who_pdf(url: object) -> bool:
    if not isinstance(url, str):
        return False
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and (host == "who.int" or host.endswith(".who.int"))
        and parsed.path.lower().endswith(".pdf")
    )


def is_pdf_page_one(value: object) -> bool:
    return value == 1 or str(value).strip() == "1"


def validate_image_sources(report: Report, pages: list[Path]):
    """Enforce the public image/source boundary without fetching remote files.

    The two synthetic PNGs remain directly downloadable.  WHO cases retain
    four-case metadata, PDF provenance, page-one hashes and official links,
    but no WHO PNG may be shipped in docs, HTML or an archive.
    """
    materials = DOCS / "materials"
    selection_path = materials / "image_selection.json"
    attribution_path = materials / "source_attribution.json"
    selection_payload = read_json(selection_path, {}) or {}
    selected = selection_payload.get("selected_images", []) if isinstance(selection_payload, dict) else []
    selection_by_id = {
        row.get("image_id"): row for row in selected if isinstance(row, dict) and row.get("image_id")
    }
    if len(selected) != 4 or set(selection_by_id) != IMAGE_IDS:
        report.fail("image_selection_shape", f"expected exactly four image records {sorted(IMAGE_IDS)}, found {sorted(selection_by_id)}", selection_path)

    attribution_payload = read_json(attribution_path, []) or []
    if isinstance(attribution_payload, dict):
        attribution_rows = attribution_payload.get("records", attribution_payload.get("sources", []))
    else:
        attribution_rows = attribution_payload
    attribution_by_id = {
        row.get("material_id"): row for row in attribution_rows if isinstance(row, dict) and row.get("material_id")
    }
    expected_source_ids = {f"text-{index:02d}" for index in range(1, 31)} | IMAGE_IDS
    if set(attribution_by_id) != expected_source_ids:
        report.fail("source_attribution_shape", f"source attribution must retain 30 text + 4 image records; found {len(attribution_by_id)}", attribution_path)
    for text_id in sorted(expected_source_ids - IMAGE_IDS):
        source_hash = str(attribution_by_id.get(text_id, {}).get("source_sha256", ""))
        if not HEX_SHA256.fullmatch(source_hash):
            report.fail("text_attribution_hash", f"{text_id} attribution lacks a frozen SHA256", attribution_path)

    text_path = materials / "frozen_text_materials.jsonl"
    text_rows = parse_jsonl_bytes(report, text_path.read_bytes(), "frozen_text_materials.jsonl", text_path) if text_path.exists() else []
    text_ids = {row.get("item_id") for row in text_rows}
    expected_text_ids = {f"text-{index:02d}" for index in range(1, 31)}
    if len(text_rows) != 30 or text_ids != expected_text_ids:
        report.fail("text_materials_shape", f"expected 30 frozen text materials, found {len(text_rows)} records/{len(text_ids)} IDs", text_path)
    for row in text_rows:
        if not row.get("source_en") or not row.get("source_url"):
            report.fail("text_material_record", f"text material {row.get('item_id', '')!r} lacks source text or URL", text_path)

    images_dir = materials / "images"
    actual_pngs = {p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"} if images_dir.exists() else set()
    if actual_pngs != SYNTHETIC_IMAGE_BASENAMES:
        report.fail("public_image_files", f"public materials/images must contain only the two synthetic PNGs; found {sorted(actual_pngs)}", images_dir)
    for shipped_path in DOCS.rglob("*"):
        if shipped_path.is_file() and shipped_path.name in WHO_IMAGE_BASENAMES and shipped_path.parent.resolve() != images_dir.resolve():
            report.fail("who_png_redistribution", f"WHO PNG must not be present anywhere under docs/: {rel(shipped_path)}", shipped_path)
    for basename in WHO_IMAGE_BASENAMES:
        if basename in actual_pngs:
            report.fail("who_png_redistribution", f"WHO PNG must not be shipped: {basename}", images_dir / basename)

    selection_pdf: dict[str, tuple[str, str]] = {}
    for image_id in sorted(IMAGE_IDS):
        selected_row = selection_by_id.get(image_id, {})
        source_hash = str(selected_row.get("sha256", ""))
        if not HEX_SHA256.fullmatch(source_hash):
            report.fail("image_hash_missing", f"{image_id} image selection hash is not a frozen SHA256", selection_path)
        target = safe_material_target(materials, selected_row.get("source_relative_path"))
        if target is None:
            report.fail("image_source_path", f"{image_id} has an invalid source_relative_path", selection_path)
        elif image_id in SYNTHETIC_IMAGE_IDS:
            if not target.exists():
                report.fail("synthetic_image_missing", f"direct synthetic image is missing for {image_id}", target)
            elif HEX_SHA256.fullmatch(source_hash) and sha256_file(target) != source_hash.lower():
                report.fail("synthetic_image_hash", f"frozen hash differs for {image_id}", target)
        elif target is not None and target.exists():
            report.fail("who_png_redistribution", f"WHO source render must not be shipped for {image_id}", target)

        if image_id in WHO_IMAGE_IDS:
            pdf_url = selected_row.get("source_pdf_url")
            pdf_hash = str(selected_row.get("source_pdf_sha256", ""))
            page_number = selected_row.get("source_pdf_page")
            if not official_who_pdf(pdf_url):
                report.fail("who_pdf_url", f"{image_id} lacks an official HTTPS WHO PDF URL", selection_path)
            if not is_pdf_page_one(page_number):
                report.fail("who_pdf_page", f"{image_id} source PDF page must be 1", selection_path)
            if not HEX_SHA256.fullmatch(pdf_hash):
                report.fail("who_pdf_hash", f"{image_id} lacks a frozen source PDF SHA256", selection_path)
            if isinstance(pdf_url, str) and isinstance(pdf_hash, str):
                selection_pdf[image_id] = (pdf_url, pdf_hash.lower())

    for image_id in sorted(IMAGE_IDS):
        row = attribution_by_id.get(image_id, {})
        if image_id in SYNTHETIC_IMAGE_IDS:
            expected_hash = str(selection_by_id.get(image_id, {}).get("sha256", "")).lower()
            attr_hash = str(row.get("source_sha256", "")).lower()
            if expected_hash and attr_hash != expected_hash:
                report.fail("synthetic_attribution_hash", f"{image_id} attribution hash differs from image selection", attribution_path)
            continue
        selection_record = selection_pdf.get(image_id)
        attr_url = row.get("source_pdf_url") or row.get("pdf_url")
        if not attr_url and isinstance(row.get("source_url"), str) and ".pdf" in row.get("source_url", "").lower():
            attr_url = row.get("source_url")
        # Accept either an explicit PDF hash field or a deliberately
        # repurposed source_sha256 field in the compact attribution record;
        # parity with image_selection.json prevents an image-PNG hash from
        # being mistaken for the frozen PDF hash.
        attr_hash = row.get("source_pdf_sha256") or row.get("pdf_sha256") or row.get("source_sha256")
        attr_page = row.get("source_pdf_page") or row.get("pdf_page")
        if not official_who_pdf(attr_url):
            report.fail("who_attribution_pdf_url", f"{image_id} attribution lacks an official WHO PDF URL", attribution_path)
        if not is_pdf_page_one(attr_page):
            report.fail("who_attribution_pdf_page", f"{image_id} attribution PDF page must be 1", attribution_path)
        if not HEX_SHA256.fullmatch(str(attr_hash or "")):
            report.fail("who_attribution_pdf_hash", f"{image_id} attribution lacks a frozen PDF SHA256", attribution_path)
        if selection_record:
            selected_url, selected_hash = selection_record
            if urlsplit(str(attr_url)).path != urlsplit(selected_url).path or str(attr_hash).lower() != selected_hash:
                report.fail("who_source_record_parity", f"{image_id} attribution PDF URL/hash differs from image selection", attribution_path)

    html_text = "\n".join(read_text(path) for path in pages)
    for basename in sorted(WHO_IMAGE_BASENAMES):
        if basename in html_text:
            report.fail("who_html_png_link", f"HTML still links to a WHO PNG: {basename}")
    for image_id, (pdf_url, _pdf_hash) in selection_pdf.items():
        pdf_path = urlsplit(pdf_url).path
        if pdf_path not in html_text:
            report.fail("who_pdf_link_missing", f"HTML lacks the official PDF link for {image_id}")
    for basename in sorted(SYNTHETIC_IMAGE_BASENAMES):
        if f"materials/images/{basename}" not in html_text:
            report.fail("synthetic_image_link_missing", f"HTML lacks a direct link for synthetic image {basename}")
    report.note("image_source_policy", "validated 30 frozen text materials, two downloadable synthetic PNGs, four image records, WHO page-1 PDF links/hashes, and no WHO PNG redistribution")


def validate_downloads(report: Report):
    manifest = read_json(DATA / "download_manifest.json", []) or []
    manifest_by_name = {
        row.get("name"): row for row in manifest if isinstance(row, dict) and row.get("name")
    }
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
                members = [name.replace("\\", "/") for name in archive.namelist()]
                if path.name == "materials.zip":
                    for suffix in ("frozen_text_materials.jsonl", "image_selection.json", "source_attribution.json"):
                        if not any(name.endswith(suffix) for name in members):
                            report.fail("materials_archive_member_missing", f"materials.zip lacks {suffix}", path)
                    image_members = {Path(name).name for name in members if "/images/" in f"/{name}" and name.lower().endswith(".png")}
                    if image_members != SYNTHETIC_IMAGE_BASENAMES:
                        report.fail("materials_archive_images", f"materials.zip must contain only two synthetic PNGs; found {sorted(image_members)}", path)
                    if image_members & WHO_IMAGE_BASENAMES:
                        report.fail("materials_archive_who_png", "materials.zip contains a WHO PNG", path)
                    text_member = next((name for name in members if name.endswith("frozen_text_materials.jsonl")), None)
                    if text_member:
                        text_rows = parse_jsonl_bytes(report, archive.read(text_member), "materials.zip frozen_text_materials.jsonl", path)
                        if len(text_rows) != 30:
                            report.fail("materials_archive_text_count", f"materials.zip text material count is {len(text_rows)}, expected 30", path)
                if path.name == "complete-results.zip":
                    required = ("docs/data/translation_results.json", "docs/data/translation_comparison.csv", "docs/data/source_file_provenance.json")
                    for suffix in required:
                        if not any(name.endswith(suffix) for name in members):
                            report.fail("complete_archive_member_missing", f"complete-results.zip lacks {suffix}", path)
                    if any(Path(name).name in WHO_IMAGE_BASENAMES for name in members):
                        report.fail("complete_archive_who_png", "complete-results.zip contains a WHO PNG", path)
                    json_member = next((name for name in members if name.endswith("docs/data/translation_results.json")), None)
                    csv_member = next((name for name in members if name.endswith("docs/data/translation_comparison.csv")), None)
                    if json_member:
                        try:
                            archive_rows = json.loads(archive.read(json_member).decode("utf-8")).get("rows", [])
                        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                            archive_rows = []
                            report.fail("complete_archive_translation_json", "complete-results.zip translation_results.json is invalid", path)
                        if len(archive_rows) != 238:
                            report.fail("complete_archive_translation_count", f"complete-results.zip JSON has {len(archive_rows)} rows, expected 238", path)
                        archive_image_ids = {row.get("material_id") for row in archive_rows if isinstance(row, dict) and row.get("source_kind") == "image"}
                        if archive_image_ids != IMAGE_IDS:
                            report.fail("complete_archive_image_ids", f"complete-results.zip JSON image IDs differ: {sorted(archive_image_ids)}", path)
                    if csv_member:
                        try:
                            archive_csv_rows = list(csv.DictReader(archive.read(csv_member).decode("utf-8-sig").splitlines()))
                        except (UnicodeDecodeError, csv.Error):
                            archive_csv_rows = []
                            report.fail("complete_archive_translation_csv", "complete-results.zip translation_comparison.csv is invalid", path)
                        if len(archive_csv_rows) != 238:
                            report.fail("complete_archive_csv_count", f"complete-results.zip CSV has {len(archive_csv_rows)} rows, expected 238", path)
                manifest_row = manifest_by_name.get(path.name)
                if manifest_row:
                    if num(manifest_row.get("bytes")) != path.stat().st_size:
                        report.fail("download_manifest_bytes", f"download manifest byte count differs for {path.name}", path)
                    if num(manifest_row.get("files")) != len(members):
                        report.fail("download_manifest_files", f"download manifest member count differs for {path.name}", path)
                    if str(manifest_row.get("sha256", "")).lower() != sha256_file(path):
                        report.fail("download_manifest_hash", f"download manifest SHA256 differs for {path.name}", path)
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
    if image_cases != IMAGE_IDS:
        report.fail("image_case_ids", f"expected image cases {sorted(IMAGE_IDS)}, found {sorted(image_cases)}")
    for row in [r for r in crows if r.get("source_kind") == "image"]:
        if row.get("source_id") != row.get("material_id") or not HEX_SHA256.fullmatch(row.get("image_sha256", "")):
            report.fail("image_source_record", f"image case {row.get('material_id', '')} lacks a preserved source ID/hash")
    bad_status = Counter(r.get("translation_status", "") for r in crows if r.get("translation_status", "") != "ok")
    if bad_status:
        report.fail("translation_status", f"non-ok statuses: {dict(bad_status)}")


def validate_criterion_scores(report: Report):
    path = DATA / "criterion_scores.json"
    payload = read_json(path, {}) or {}
    if not isinstance(payload, dict) or payload.get("available") is not True:
        report.fail("criterion_scores_unavailable", "criterion_scores.json is missing or unavailable", path)
        return
    if payload.get("schema_version") != "criterion-scores-v1":
        report.fail("criterion_scores_schema", "unexpected criterion_scores.json schema", path)
    tracks = payload.get("tracks", {})
    expected_tracks = {"text": (29, 5), "image": (4, 4)}
    for track, (item_count, judge_count) in expected_tracks.items():
        record = tracks.get(track, {}) if isinstance(tracks, dict) else {}
        if record.get("eligible_items") != item_count or record.get("judge_count") != judge_count:
            report.fail("criterion_scores_shape", f"{track} criterion shape does not match {item_count} items × {judge_count} judges", path)

    dimensions = {
        "terminology_locale": 30,
        "semantic_safety": 30,
        "fluency": 20,
        "completeness_format": 20,
    }
    common = read_csv(DATA / "common_case_model_summary.csv")
    common_by_key = {(row.get("model_id"), row.get("track")): num(row.get("common_case_panel_mean")) for row in common}
    models = payload.get("models", {})
    model_ids = sorted({row.get("model_id") for row in common})
    if not isinstance(models, dict) or set(models) != set(model_ids):
        report.fail("criterion_scores_models", "criterion model IDs do not match common-case summaries", path)
        return
    failures = []
    for model_id in model_ids:
        model = models.get(model_id, {})
        for track, (_items, _judges) in expected_tracks.items():
            record = model.get(track, {}) if isinstance(model, dict) else {}
            total = record.get("total", {}) if isinstance(record, dict) else {}
            dim_records = record.get("dimensions", {}) if isinstance(record, dict) else {}
            raw_sum = 0.0
            for key, maximum in dimensions.items():
                item = dim_records.get(key, {}) if isinstance(dim_records, dict) else {}
                raw = num(item.get("raw_mean"))
                normalized = num(item.get("normalized"))
                if raw is None or normalized is None or not same_num(normalized, raw / maximum * 100.0, 0.00001):
                    failures.append(f"{model_id}/{track}/{key}")
                else:
                    raw_sum += raw
            if not same_num(total.get("raw_mean"), raw_sum, 0.00001):
                failures.append(f"{model_id}/{track}/sum")
            if not same_num(total.get("normalized"), total.get("raw_mean"), 0.00001):
                failures.append(f"{model_id}/{track}/total-scale")
            if not same_num(total.get("normalized"), common_by_key.get((model_id, track)), 0.0011):
                failures.append(f"{model_id}/{track}/published-total")
    if failures:
        report.fail("criterion_scores_parity", f"criterion numeric parity failures: {failures[:5]}", path)
    else:
        report.note("criterion_scores_parity", "7 models × 2 tracks × 4 dimensions match raw-scale sums and published totals")


def overview_expected_values(criterion_payload: dict, model_id: str, track: str) -> dict[str, dict[str, float | None]]:
    """Return normalized/raw/max values used by the homepage overview rows."""
    model = (criterion_payload.get("models", {}) or {}).get(model_id, {}) or {}
    record = model.get(track, {}) or {}
    total = record.get("total", {}) or {}
    values: dict[str, dict[str, float | None]] = {
        "total": {
            "normalized": num(total.get("normalized")),
            "raw": num(total.get("raw_mean")),
            "max": num(total.get("max_raw"), 100.0),
        }
    }
    dimensions = record.get("dimensions", {}) or {}
    for metric, maximum in CRITERION_MAXIMA.items():
        item = dimensions.get(metric, {}) or {}
        values[metric] = {
            "normalized": num(item.get("normalized")),
            "raw": num(item.get("raw_mean")),
            "max": num(item.get("max_raw"), float(maximum)),
        }
    return values


def _overview_row_attrs(row: dict[str, object]) -> dict[str, str]:
    attrs = row.get("attrs", {})
    return attrs if isinstance(attrs, dict) else {}


def validate_overview_contract(
    report: Report,
    path: Path,
    html: str,
    common_by_model: dict[tuple[str, str], float | None],
    speed_by_model: dict[tuple[str, str], float | None],
    criterion_payload: dict,
    model_ids: list[str],
    runtime_by_model: dict[str, dict[str, float | None]] | None = None,
) -> bool:
    """Validate the research-report homepage chart/table contract.

    The annual-report homepage must expose the track/sort controls and the
    no-JavaScript table.  Missing markers are a release failure; the checker
    deliberately does not accept the retired card-only homepage as a legacy
    fallback.
    """
    overview = OverviewParser()
    overview.feed(html)
    score_precisions = criterion_display_precisions(criterion_payload)
    has_new_marker = bool(
        overview.track_buttons
        or overview.sort_buttons
        or any(row.get("context") == "table" for row in overview.rows)
    )
    if not has_new_marker:
        report.fail(
            "overview_contract_missing",
            "homepage lacks the required research-report track/sort controls or overview table",
            path,
        )
        return True

    if overview.ids.get("overview-chart", 0) != 1:
        report.fail("overview_chart_anchor", "new overview contract requires exactly one #overview-chart", path)
    if overview.ids.get("overview-sort-note", 0) != 1:
        report.fail("overview_sort_note", "new overview contract requires #overview-sort-note", path)
    if overview.ids.get("overview-table-heading", 0) != 1:
        report.fail("overview_table_heading", "new overview contract requires #overview-table-heading", path)

    metric_values = [button.get("data-overview-metric", "") for button in overview.metric_buttons]
    track_values = [button.get("data-overview-track", "") for button in overview.track_buttons]
    sort_values = [button.get("data-overview-sort", "") for button in overview.sort_buttons]
    if metric_values != OVERVIEW_METRICS:
        report.fail("overview_metric_controls", f"overview metric controls differ: {metric_values!r}", path)
    if track_values != OVERVIEW_TRACKS:
        report.fail("overview_track_controls", f"overview track controls differ: {track_values!r}", path)
    if sort_values != OVERVIEW_SORTS:
        report.fail("overview_sort_controls", f"overview sort controls differ: {sort_values!r}", path)
    for label, buttons in (("metric", overview.metric_buttons), ("track", overview.track_buttons), ("sort", overview.sort_buttons)):
        for button in buttons:
            if button.get("aria-pressed") not in {"true", "false"}:
                report.fail("overview_control_state", f"{label} control lacks boolean aria-pressed", path)

    chart_rows = [row for row in overview.rows if row.get("context") == "chart"]
    table_rows = [row for row in overview.rows if row.get("context") == "table"]
    if len(chart_rows) != EXPECTED_MODEL_COUNT:
        report.fail("overview_chart_row_count", f"expected {EXPECTED_MODEL_COUNT} chart rows, found {len(chart_rows)}", path)
    if len(table_rows) != EXPECTED_MODEL_COUNT:
        report.fail("overview_table_row_count", f"expected {EXPECTED_MODEL_COUNT} table rows, found {len(table_rows)}", path)

    chart_by_model = {_overview_row_attrs(row).get("data-model-id", ""): row for row in chart_rows}
    table_by_model = {_overview_row_attrs(row).get("data-model-id", ""): row for row in table_rows}
    expected_models = set(model_ids)
    if set(chart_by_model) != expected_models:
        report.fail("overview_chart_models", f"chart model IDs differ: found={sorted(chart_by_model)}, expected={sorted(expected_models)}", path)
    if set(table_by_model) != expected_models:
        report.fail("overview_table_models", f"table model IDs differ: found={sorted(table_by_model)}, expected={sorted(expected_models)}", path)

    required_attrs: list[str] = []
    for track in ("text", "image"):
        for metric in OVERVIEW_METRICS:
            # comparison_tool uses the canonical criterion key verbatim in
            # data-* attributes (e.g. terminology_locale).
            key = metric
            base = f"data-{track}-{key}"
            required_attrs.extend([base, f"{base}-display", f"{base}-raw", f"{base}-max"])
        required_attrs.extend([
            f"data-{track}-warm",
            f"data-{track}-warm-display",
        ])

    # Compare chart/table data attributes and verify each value against the
    # public criterion and generation summaries.  The table may use a <tr>
    # while the chart uses a <div>; the data contract is intentionally shared.
    for model_id in sorted(expected_models):
        chart_attrs = _overview_row_attrs(chart_by_model.get(model_id, {}))
        table_attrs = _overview_row_attrs(table_by_model.get(model_id, {}))
        if not chart_attrs or not table_attrs:
            continue
        for key in required_attrs:
            if key not in chart_attrs:
                report.fail("overview_data_attribute", f"chart row {model_id} missing {key}", path)
            if key not in table_attrs:
                report.fail("overview_table_attribute", f"table row {model_id} missing {key}", path)
            if key in chart_attrs and key in table_attrs and chart_attrs[key] != table_attrs[key]:
                report.fail("overview_chart_table_parity", f"{model_id} {key} differs between chart and table", path)

        expected_values = {
            track: overview_expected_values(criterion_payload, model_id, track)
            for track in ("text", "image")
        }
        for track in ("text", "image"):
            for metric in OVERVIEW_METRICS:
                expected = expected_values[track][metric]
                # The public comparison_tool keeps the criterion key in its
                # data attribute (for example
                # ``data-text-terminology_locale``).  Do not silently accept
                # the retired visual-overview hyphenated form here.
                base = f"data-{track}-{metric}"
                expected_normalized = expected.get("normalized")
                expected_raw = expected.get("raw")
                expected_max = expected.get("max")
                if expected_normalized is None or expected_raw is None or expected_max is None:
                    report.fail("overview_source_value_missing", f"criterion source missing {model_id}/{track}/{metric}", path)
                    continue
                attrs = chart_attrs
                if not same_num(attrs.get(base), expected_normalized, 0.00001):
                    report.fail("overview_normalized_value", f"{model_id}/{track}/{metric} normalized value differs from criterion_scores.json", path)
                if not same_num(attrs.get(f"{base}-raw"), expected_raw, 0.00001):
                    report.fail("overview_raw_value", f"{model_id}/{track}/{metric} raw value differs from criterion_scores.json", path)
                if not same_num(attrs.get(f"{base}-max"), expected_max, 0.00001):
                    report.fail("overview_max_value", f"{model_id}/{track}/{metric} maximum differs from criterion_scores.json", path)
                expected_display = format_trimmed(expected_normalized, score_precisions[metric])
                if attrs.get(f"{base}-display") != expected_display:
                    report.fail(
                        "overview_display_value",
                        f"{model_id}/{track}/{metric} display value is not the required {score_precisions[metric]}-decimal normalized value",
                        path,
                    )

            expected_speed = speed_by_model.get((model_id, track))
            if expected_speed is None:
                report.fail("overview_speed_source_missing", f"generation summary missing {model_id}/{track}", path)
            else:
                base = f"data-{track}-warm"
                attrs = chart_attrs
                if not same_num(attrs.get(base), expected_speed, 0.00001):
                    report.fail("overview_speed_value", f"{model_id}/{track} warm median differs from generation summary", path)
                expected_speed_display = format_trimmed(expected_speed, 2)
                if attrs.get(f"{base}-display") != expected_speed_display:
                    report.fail("overview_speed_display", f"{model_id}/{track} warm median display is not the required two-decimal value", path)

        chart_row = chart_by_model[model_id]
        chart_text = str(chart_row.get("text", ""))
        expected_chart_scores = [
            expected_values[track]["total"].get("normalized") for track in ("text", "image")
        ]
        expected_chart_speeds = [speed_by_model.get((model_id, track)) for track in ("text", "image")]
        for track, expected_score, expected_speed in zip(("text", "image"), expected_chart_scores, expected_chart_speeds):
            if expected_score is not None:
                score_token = format_trimmed(expected_score, score_precisions["total"])
                if not numeric_token_present(chart_text, score_token):
                    report.fail("overview_visible_score", f"chart row {model_id}/{track} is missing displayed score {score_token}", path)
            if expected_speed is not None:
                speed_token = format_trimmed(expected_speed, 2)
                if not numeric_token_present(chart_text, speed_token):
                    report.fail("overview_visible_speed", f"chart row {model_id}/{track} is missing displayed median {speed_token}s", path)

        score_widths = chart_row.get("score_widths", [])
        if len(score_widths) != 2 or any(
            expected is None or not same_num(actual, expected, 0.0011)
            for actual, expected in zip(score_widths, expected_chart_scores)
        ):
            report.fail("overview_score_bar_widths", f"chart row {model_id} score bars do not match normalized values at rendered precision", path)
        speed_positions = chart_row.get("speed_positions", [])
        expected_positions = [
            round((speed / 6.0) * 100.0, 2) if speed is not None else None
            for speed in expected_chart_speeds
        ]
        if len(speed_positions) != 2 or any(
            expected is None or not same_num(actual, expected, 0.011)
            for actual, expected in zip(speed_positions, expected_positions)
        ):
            report.fail("overview_speed_positions", f"chart row {model_id} 0–6s dot positions do not match rounded medians", path)

        # The table must be a real no-JavaScript readable fallback, not just
        # hidden duplicate data attributes.
        table_text = str(table_by_model[model_id].get("text", "")).strip()
        if not table_text:
            report.fail("overview_table_readable", f"table row {model_id} has no readable text", path)
        else:
            table_cells = table_by_model[model_id].get("cells", [])
            for track, expected_score, expected_speed in zip(("text", "image"), expected_chart_scores, expected_chart_speeds):
                score_cells = [
                    cell for cell in table_cells
                    if cell.get("attrs", {}).get("data-table-track") == track
                    and "overview-table-score-cell" in cell.get("attrs", {}).get("class", "").split()
                ]
                if len(score_cells) != 1:
                    report.fail("overview_table_score_cell", f"table row {model_id}/{track} has {len(score_cells)} score cells", path)
                elif expected_score is not None and not numeric_token_present(
                    str(score_cells[0].get("text", "")), format_trimmed(expected_score, score_precisions["total"])
                ):
                    report.fail("overview_table_visible_score", f"table row {model_id}/{track} score cell is missing its displayed value", path)

                time_cells = [
                    cell for cell in table_cells
                    if cell.get("attrs", {}).get("data-table-track") == track
                    and "overview-table-time-cell" in cell.get("attrs", {}).get("class", "").split()
                ]
                if len(time_cells) != 1:
                    report.fail("overview_table_time_cell", f"table row {model_id}/{track} has {len(time_cells)} time cells", path)
                elif expected_speed is not None and not numeric_token_present(
                    str(time_cells[0].get("text", "")), format_trimmed(expected_speed, 2)
                ):
                    report.fail("overview_table_visible_speed", f"table row {model_id}/{track} time cell is missing its displayed value", path)

            runtime = (runtime_by_model or {}).get(model_id, {})
            total_size = runtime.get("total_size_gib")
            runtime_cells = [
                cell for cell in table_cells
                if "overview-table-files" in cell.get("attrs", {}).get("class", "").split()
            ]
            if len(runtime_cells) != 1:
                report.fail("overview_table_runtime_cell", f"table row {model_id} has {len(runtime_cells)} runtime cells", path)
            elif total_size is not None and not numeric_unit_present(
                str(runtime_cells[0].get("text", "")), format_trimmed(total_size, 1), "GiB"
            ):
                report.fail("overview_table_visible_runtime", f"table row {model_id} runtime cell is missing one-decimal total file size", path)

    # The new homepage must retain the conditions and definitions needed to
    # interpret the choices.  These checks are intentionally semantic markers,
    # not claims that static HTML proves browser behavior.
    lower = html.lower()
    # Exact denominator prose is intentionally not hard-coded: the public
    # summary CSV/JSON checks above are authoritative, while this only checks
    # that the homepage states the score scale and timing interpretation.
    conditions = ["238", "0–100"]
    if path.name.endswith("-en.html"):
        conditions.extend(["after model loading"])
        if not any(marker in lower for marker in ("reviewer", "panel", "judge")):
            report.fail("homepage_conditions", "homepage selection contract lacks reviewer/panel wording", path)
    else:
        conditions.extend(["評審", "載入後"])
    for marker in conditions:
        if marker.lower() not in lower:
            report.fail("homepage_conditions", f"homepage selection contract missing condition marker {marker!r}", path)
    for marker in ("terminology_locale", "semantic_safety", "fluency", "completeness_format"):
        if marker not in html:
            report.fail("homepage_criterion_definition", f"homepage selection contract missing criterion marker {marker!r}", path)

    # Independent numeric audit of the three intended trade-offs: 26B/31B
    # shared text score and speed, then 31B/12B image score and speed.  Values
    # are checked by their formatted tokens so the prose can be localized.
    text_26 = common_by_model.get(("gemma4-26b", "text"))
    text_31 = common_by_model.get(("gemma4-31b-qat", "text"))
    image_31 = common_by_model.get(("gemma4-31b-qat", "image"))
    image_12 = common_by_model.get(("gemma4-12b", "image"))
    speed_text_26 = speed_by_model.get(("gemma4-26b", "text"))
    speed_text_31 = speed_by_model.get(("gemma4-31b-qat", "text"))
    speed_image_31 = speed_by_model.get(("gemma4-31b-qat", "image"))
    speed_image_12 = speed_by_model.get(("gemma4-12b", "image"))
    facts = [
        ("26B text score", text_26, score_precisions["total"]),
        ("31B text score", text_31, score_precisions["total"]),
        ("26B text speed", speed_text_26, 2),
        ("31B text speed", speed_text_31, 2),
        ("31B image score", image_31, score_precisions["total"]),
        ("12B image score", image_12, score_precisions["total"]),
        ("31B image speed", speed_image_31, 2),
        ("12B image speed", speed_image_12, 2),
    ]
    for label, value, digits in facts:
        if value is not None and not numeric_token_present(html, format_trimmed(value, digits)):
            report.fail("homepage_selection_fact", f"homepage is missing independently checked {label} value", path)
    if all(value is not None for _, value, _ in facts):
        differences = [
            abs(text_26 - text_31),
            abs(speed_text_31 - speed_text_26),
            abs(image_31 - image_12),
            abs(speed_image_31 - speed_image_12),
        ]
        for label, value in zip(("26B/31B text score gap", "26B/31B text speed gap", "31B/12B image score gap", "31B/12B image speed gap"), differences):
            if not numeric_token_present(html, format_trimmed(value, 2)):
                report.fail("homepage_selection_difference", f"homepage is missing independently checked {label} value", path)
    return True


def validate_numeric_charts(report: Report, pages: list[Path]):
    common = read_csv(DATA / "common_case_model_summary.csv")
    generation = read_csv(DATA / "translation_generation_summary.csv")
    criterion_payload = read_json(DATA / "criterion_scores.json", {}) or {}
    score_precisions = criterion_display_precisions(criterion_payload)
    inventory_payload = read_json(DATA / "model_inventory.json", {}) or {}
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
    runtime_by_model: dict[str, dict[str, float | None]] = {}
    inventory_models = inventory_payload.get("models", {}) if isinstance(inventory_payload, dict) else {}
    for model_id in model_ids:
        item = inventory_models.get(model_id, {}) if isinstance(inventory_models, dict) else {}
        model_bytes = num(item.get("model_size_bytes"))
        mmproj_bytes = num(item.get("mmproj_size_bytes"))
        runtime_by_model[model_id] = {
            "model_size_gib": model_bytes / (1024**3) if model_bytes is not None else None,
            "mmproj_size_gib": mmproj_bytes / (1024**3) if mmproj_bytes is not None else None,
            "total_size_gib": (model_bytes + mmproj_bytes) / (1024**3)
            if model_bytes is not None and mmproj_bytes is not None
            else None,
        }

    expected_denominators = {"text": (29, 5, 145), "image": (4, 4, 16)}
    for row in common:
        track = row.get("track", "")
        expected_items, expected_judges, expected_cells = expected_denominators.get(track, (None, None, None))
        if expected_items is None:
            report.fail("common_summary_track", f"unexpected common-summary track {track!r}")
            continue
        checks = {
            "common_case_items": expected_items,
            "required_judge_k": expected_judges,
            "valid_judge_cells": expected_cells,
            "expected_judge_cells": expected_cells,
        }
        for field, expected in checks.items():
            if row.get(field) != str(expected):
                report.fail("common_summary_denominator", f"{row.get('model_id')}/{track} {field}={row.get(field)!r}, expected {expected}")
        if track == "text" and row.get("excluded_case_ids") != "text-15":
            report.fail("common_summary_exclusion", f"{row.get('model_id')}/text must exclude only text-15")
        if track == "image" and row.get("excluded_case_ids", "") not in {"", None}:
            report.fail("common_summary_image_exclusion", f"{row.get('model_id')}/image unexpectedly excludes cases")
    for row in generation:
        track = row.get("track", "")
        expected_cases = {"text": 30, "image": 4}.get(track)
        if expected_cases is None:
            report.fail("generation_summary_track", f"unexpected generation-summary track {track!r}")
            continue
        if row.get("translation_cases") != str(expected_cases):
            report.fail("generation_denominator", f"{row.get('model_id')}/{track} translation_cases={row.get('translation_cases')!r}, expected {expected_cases}")
        if row.get("translation_status_ok") != str(expected_cases):
            report.fail("generation_status_denominator", f"{row.get('model_id')}/{track} translation_status_ok={row.get('translation_status_ok')!r}, expected {expected_cases}")
        if row.get("not_judge_latency", "").lower() != "true":
            report.fail("generation_latency_definition", f"{row.get('model_id')}/{track} does not mark latency as generation-only")
        if num(row.get("translation_elapsed_median_s")) is None or num(row.get("translation_elapsed_p95_s")) is None:
            report.fail("generation_latency_numeric", f"{row.get('model_id')}/{track} has missing median/p95 seconds")

    index_pages = [DOCS / "index.html", DOCS / "index-en.html"]
    for path in index_pages:
        if not path.exists():
            report.fail("index_missing", "index page missing", path)
            continue
        html = read_text(path)
        if "238" not in html:
            report.fail("headline_count", "index does not expose 238 translation count", path)
        validate_overview_contract(
            report,
            path,
            html,
            common_by_model,
            speed_by_model,
            criterion_payload,
            model_ids,
            runtime_by_model,
        )
    for model_id in model_ids:
        for suffix in ("", "-en"):
            path = DOCS / "models" / f"{model_id}{suffix}.html"
            if not path.exists():
                report.fail("model_page_missing", f"missing model page {path.name}", path)
                continue
            html = read_text(path)
            for row in [r for r in common if r.get("model_id") == model_id]:
                value = format_trimmed(row.get("common_case_panel_mean"), score_precisions["total"])
                if not numeric_token_present(html, value):
                    report.fail("model_metric_missing", f"common score {model_id}/{row['track']}={value} missing", path)
            for row in [r for r in generation if r.get("model_id") == model_id]:
                value = format_trimmed(row.get("translation_elapsed_median_s"), 2)
                if not numeric_token_present(html, value):
                    report.fail("model_latency_missing", f"generation median {model_id}/{row['track']}={value}s missing", path)
            for field in ("model_size_gib", "mmproj_size_gib", "total_size_gib"):
                value = runtime_by_model.get(model_id, {}).get(field)
                if value is not None and not numeric_unit_present(html, format_trimmed(value, 1), "GiB"):
                    report.fail("model_runtime_missing", f"{field} {model_id}={format_trimmed(value, 1)} GiB missing", path)

    for path in (DOCS / "results.html", DOCS / "results-en.html"):
        if not path.exists():
            report.fail("results_page_missing", "results page missing", path)
            continue
        page = PageParser()
        page.feed(read_text(path))
        if page.result_rows != 238:
            report.fail("results_row_count", f"expected 238 result rows, found {page.result_rows}", path)
    report.note("numeric_chart_fields", "validated annual-report overview controls, chart/table parity, raw/normalized criterion values, denominators, generation medians, model pages, and 238 result rows")


def validate_reliability_display(report: Report):
    """Check the public ICC display precision without treating it as accuracy."""
    rows = read_csv(DATA / "reliability_summary.csv")
    if not rows:
        report.fail("reliability_data_missing", "public reliability_summary.csv is missing or empty")
        return
    methods_paths = (DOCS / "methods.html", DOCS / "methods-en.html")
    fields = ("icc_2_1_absolute_agreement", "icc_2_k_absolute_agreement")
    for path in methods_paths:
        if not path.exists():
            report.fail("methods_page_missing", "methods page missing for ICC display check", path)
            continue
        html = read_text(path)
        for row in rows:
            track = row.get("track", "")
            for field in fields:
                value = num(row.get(field))
                if value is None:
                    report.fail("reliability_value_missing", f"{track}/{field} is not numeric", path)
                    continue
                token = format_trimmed(value, 2)
                if not numeric_token_present(html, token):
                    report.fail("reliability_display_missing", f"{track}/{field} two-decimal value {token} missing", path)
        # The compact ICC summary is also public; reject stale three-decimal
        # output even if a separate detailed section happens to contain a
        # two-decimal copy of the same estimate.
        for snippet in re.findall(r'class="judge-icc-mini".*?</div>', html, flags=re.IGNORECASE | re.DOTALL):
            if re.search(r"(?<![0-9.])[0-9]+\.[0-9]{3}(?![0-9])", snippet):
                report.fail("reliability_precision", "compact ICC summary still exposes a three-decimal value", path)
    report.note("reliability_display", "validated ICC values at two displayed decimals; this is reviewer agreement, not translation accuracy")


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
    validate_image_sources(report, pages)
    validate_translation_data(report)
    validate_criterion_scores(report)
    validate_numeric_charts(report, pages)
    validate_reliability_display(report)
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

