#!/usr/bin/env python3
"""Finalize one captured native Gemini UI response without contacting Gemini."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/benchmark/translation_benchmark_20261023_new_materials/judge_evaluation")
RAW = ROOT / "raw" / "cloud"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--conversation-url", required=True)
    parser.add_argument("--visible-model-label", required=True)
    parser.add_argument("--ax-path", required=True)
    parser.add_argument("--suffix", default="", help="optional capture suffix such as repair")
    parser.add_argument(
        "--invalid-reason",
        default="",
        help="record a preserved native-UI capture as invalid/missing when no unambiguous JSON can be accepted",
    )
    args = parser.parse_args()

    req_dir = RAW / "gemini_ui_requests"
    meta_path = req_dir / f"{args.request_id}.request.json"
    ax_path = Path(args.ax_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    ax_text = ax_path.read_text(encoding="utf-8")
    # The notification copy is the plain-text accessibility capture.  Prefer
    # that over the message container, whose AX serialization may add spacing.
    # Pointwise responses identify a blind_id/score; pairwise responses use
    # winner/confidence instead.  Keep both routes strict and request-scoped.
    if args.request_id.startswith("pairwise-"):
        response_markers = r'(?=.*"winner")(?=.*"confidence")'
        required_markers = ("winner", "confidence")
    else:
        response_markers = r'(?=.*"blind_id")(?=.*"score_0_100")'
        required_markers = ("blind_id", "score_0_100")
    extracted_nested_ax = False
    matches = re.findall(rf"\n\s*\d+\s+文字\s+(\{{{response_markers}.*?\}})\s*$", ax_text, flags=re.MULTILINE)
    if not matches:
        # The response message container is complete but macOS AX may insert
        # extra spacing between tokens.  Keep that exact line as raw evidence;
        # only the derived JSON parse below may collapse those AX separators.
        matches = re.findall(rf"\n\s*\d+\s+text\s+(\{{{response_markers}.*?\}})\s*$", ax_text, flags=re.MULTILINE)
    if not matches:
        # Some Antigravity AX trees split a code block into nested `text` /
        # `文字` nodes. Reassemble only nodes between an opening and closing
        # brace; the original AX capture remains the audit source.
        nested: list[str] = []
        in_object = False
        parts: list[str] = []
        for line in ax_text.splitlines():
            node = re.match(r"^\s*\d+\s+(?:文字|text)\s+(.*)$", line)
            if not in_object:
                if node and node.group(1).strip() == "{":
                    in_object = True
                    parts = ["{"]
                continue
            if not node:
                continue
            fragment = node.group(1)
            parts.append(fragment)
            if fragment.strip() == "}":
                candidate = "".join(parts)
                if all(marker in candidate for marker in required_markers):
                    nested.append(candidate)
                in_object = False
                parts = []
        if nested:
            matches = nested
            extracted_nested_ax = True
    if not matches:
        if not args.invalid_reason:
            raise SystemExit("no complete JSON response found in AX capture")
        now = datetime.now(timezone.utc).isoformat()
        invalid_record = {
            "conversation_url": args.conversation_url,
            "visible_model_label": args.visible_model_label,
            "sent_at": metadata.get("sent_at") or now,
            "captured_at": now,
            "capture_path": str(ax_path),
            "response_sha256": sha256_bytes(ax_text.encode("utf-8")),
            "status": "captured_native_ui_invalid",
            "invalid_reason": args.invalid_reason,
            "response_parse_note": "No single unambiguous JSON judgment was accepted; the complete AX capture remains preserved verbatim.",
        }
        metadata.update(invalid_record)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"request_id": args.request_id, "conversation_url": args.conversation_url, "visible_model_label": args.visible_model_label, "response_sha256": invalid_record["response_sha256"], "status": invalid_record["status"], "invalid_reason": args.invalid_reason}, ensure_ascii=False))
        return
    response_raw = matches[-1]
    parse_text = response_raw
    parse_normalized = False
    try:
        parsed = json.loads(parse_text)
    except json.JSONDecodeError:
        parse_text = re.sub(r"\s{2,}", " ", response_raw)
        parsed = json.loads(parse_text)
        parse_normalized = True
    suffix = f".{args.suffix}" if args.suffix else ""
    response_raw_path = req_dir / f"{args.request_id}{suffix}.response.raw.json"
    response_ax_path = req_dir / f"{args.request_id}{suffix}.response.ax.txt"
    response_ax_path.write_text(response_raw + "\n", encoding="utf-8")
    response_json_path = req_dir / f"{args.request_id}{suffix}.response.json"
    response_raw_path.write_text(response_raw + "\n", encoding="utf-8")
    response_json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    response_sha = sha256_bytes(response_raw.encode("utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    capture_record = {
            "conversation_url": args.conversation_url,
            "visible_model_label": args.visible_model_label,
            "sent_at": metadata.get("sent_at") or now,
            "captured_at": now,
            "capture_path": str(ax_path),
            "response_raw_path": str(response_raw_path),
            "response_ax_path": str(response_ax_path),
            "response_json_path": str(response_json_path),
            "response_sha256": response_sha,
            "response_parse_normalized_ax_spacing": parse_normalized,
            "response_parse_nested_ax": extracted_nested_ax,
            "response_parse_note": ("Raw AX response is preserved; derived JSON parsing collapsed AX-inserted repeated whitespace." if parse_normalized else "Raw AX response parsed without normalization.") + (" Nested AX code-block nodes were reassembled only for derived parsing." if extracted_nested_ax else ""),
            "status": "captured_native_ui",
            "response": parsed,
    }
    if args.suffix:
        metadata[args.suffix] = capture_record
    else:
        metadata.update(capture_record)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"request_id": args.request_id, "conversation_url": args.conversation_url, "visible_model_label": args.visible_model_label, "response_sha256": response_sha, "status": "captured_native_ui", "response": parsed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
