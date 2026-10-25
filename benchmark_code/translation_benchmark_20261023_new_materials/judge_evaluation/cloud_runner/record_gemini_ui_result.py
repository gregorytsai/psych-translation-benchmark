#!/usr/bin/env python3
"""Record one native Gemini UI result in the resumable cloud checkpoint.

This script only imports a previously captured UI response.  It never sends a
request, never changes a score, and refuses duplicate request IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/benchmark/translation_benchmark_20261023_new_materials/judge_evaluation")
RAW = ROOT / "raw" / "cloud"
PUBLIC = ROOT / "public"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument(
        "--capture-key",
        default="",
        help="metadata capture key for an authorized infrastructure retry (for example retry1)",
    )
    parser.add_argument(
        "--missing-reason",
        default="",
        help="record this captured attempt as explicit missing (no score accepted), with a non-empty route/contract reason",
    )
    args = parser.parse_args()

    checkpoint_path = RAW / "gemini.checkpoint.json"
    checkpoint = load_json(checkpoint_path)
    existing = {row.get("request_id") for row in checkpoint.get("results", [])}
    if args.request_id in existing:
        raise SystemExit(f"refusing duplicate request id: {args.request_id}")
    if args.index != checkpoint.get("last_index", 0):
        raise SystemExit(f"index mismatch: expected next {checkpoint.get('last_index', 0)}, got {args.index}")

    request_dir = RAW / "gemini_ui_requests"
    meta = load_json(request_dir / f"{args.request_id}.request.json")
    if args.repair and args.capture_key:
        raise SystemExit("--repair and --capture-key are mutually exclusive")
    if args.repair and args.missing_reason:
        raise SystemExit("--repair and --missing-reason are mutually exclusive")
    if args.missing_reason and not args.capture_key and meta.get("status") != "captured_native_ui_invalid":
        raise SystemExit("--missing-reason requires --capture-key so the preserved capture is explicit")
    capture_key = args.capture_key or ("repair" if args.repair else "")
    response_meta = meta.get(capture_key) if capture_key else meta
    if not response_meta or response_meta.get("status") not in {"captured_native_ui", "captured_native_ui_invalid"}:
        raise SystemExit("captured native response metadata is missing")
    response = response_meta.get("response")
    if args.request_id.startswith("pairwise-"):
        track = "pairwise"
        pair_job_id = args.request_id.removeprefix("pairwise-")
        row = next(
            load_json_line
            for load_json_line in (json.loads(line) for line in (PUBLIC / "pairwise" / "judge-05.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
            if load_json_line.get("pair_job_id") == pair_job_id
        )
    else:
        track = "pointwise"
        row = next(
            load_json_line
            for load_json_line in (json.loads(line) for line in (PUBLIC / "pointwise" / "judge-05.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
            if f"pointwise-{load_json_line.get('job_id')}" == args.request_id
        )
    raw_path = Path(response_meta.get("capture_path") or response_meta["response_raw_path"])
    missing = bool(args.missing_reason)
    if track == "pairwise":
        task_binding = {
            "pair_job_id": row.get("pair_job_id"),
            "pair_key": row.get("pair_key"),
            "orientation": row.get("orientation"),
            "source_kind": row.get("source_kind"),
            "image_alias": row.get("image_alias"),
            "candidate_a_blind_id": row.get("candidate_a", {}).get("blind_id"),
            "candidate_b_blind_id": row.get("candidate_b", {}).get("blind_id"),
        }
    else:
        task_binding = {
            "job_id": row.get("job_id"),
            "blind_id": row.get("blind_id"),
            "blind_source_id": row.get("blind_source_id"),
            "source_kind": row.get("source_kind"),
            "image_alias": row.get("image_alias"),
        }
    result = {
        "request_id": args.request_id,
        "status": "missing" if missing else "ok",
        "parsed": None if missing else response,
        "raw_path": str(raw_path),
        "attempt_kind": (
            "format_only_repair"
            if args.repair
            else (
                "native_ui_schema_embedded_malformed_missing"
                if missing and response_meta.get("status") == "captured_native_ui_invalid"
                else (
                    "route_contaminated_missing"
                    if missing
                    else ("native_ui_schema_embedded_infrastructure_retry1" if capture_key == "retry1" else "native_ui_schema_embedded")
                )
            )
        ),
        "capture_key": capture_key or None,
        "prior_request_status": meta.get("status") if capture_key else None,
        "regime": "schema_embedded_v2",
        "conversation_url": response_meta.get("conversation_url"),
        "visible_model_label": response_meta.get("visible_model_label"),
        "prompt_sha256": meta.get("prompt_sha256"),
        "instruction_sha256": meta.get("instruction_sha256"),
        "public_schema_sha256": meta.get("public_schema_sha256"),
        "response_sha256": response_meta.get("response_sha256"),
        "repair_prompt_sha256": sha256_file(request_dir / f"{args.request_id}.repair.prompt.txt") if args.repair else None,
        "original_response_path": meta.get("response_json_path") if args.repair else None,
        "original_response_sha256": meta.get("response_sha256") if args.repair else None,
        "block_reason": args.missing_reason if missing else "",
        "score_0_100": None if missing or track == "pairwise" else response.get("score_0_100"),
        "task_binding": task_binding,
    }
    checkpoint["results"].append(result)
    checkpoint["last_request_id"] = args.request_id
    checkpoint["last_index"] = args.index + 1
    checkpoint["counts"]["ok"] = sum(1 for x in checkpoint["results"] if x.get("status") == "ok")
    checkpoint["counts"]["failed"] = sum(1 for x in checkpoint["results"] if x.get("status") == "failed")
    checkpoint["counts"]["missing"] = sum(1 for x in checkpoint["results"] if x.get("status") == "missing")
    checkpoint["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    checkpoint.setdefault("regimes", {})["schema_embedded_v2"] = {
        "prompt_schema_embedded": True,
        "public_schema_sha256": meta.get("public_schema_sha256"),
        "notes": "Fresh native UI conversations with full frozen public schema embedded; older rows remain schema_not_embedded and are not re-evaluated.",
    }
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"request_id": args.request_id, "index": args.index, "status": result["status"], "score": result["score_0_100"], "next_index": checkpoint["last_index"], "counts": checkpoint["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
