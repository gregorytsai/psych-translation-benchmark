#!/usr/bin/env python3
"""Build the Qwen run's immutable-evidence manifest."""

from __future__ import annotations

import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
ARTIFACTS = [
    "qwen3.8-flash-next-local-thinking-xhigh-20261022.raw.json",
    "qwen3.8-flash-next-local-thinking-xhigh-20261022.checkpoint.json",
    "capability_smoke.raw.json",
    "props.json",
    "models.json",
    "template_render_xhigh_false.json",
    "strict_validate.json",
    "strict_protocol_audit.json",
    "template_hash_audit.json",
    "token_budget_audit.json",
    "post_cleanup_health.json",
    "progress.md",
    "protocol.md",
    "server.pid",
]


def digest(path: pathlib.Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    raw_path = ROOT / ARTIFACTS[0]
    raw = json.loads(raw_path.read_text())
    token_audit = json.loads((ROOT / "token_budget_audit.json").read_text())
    strict = json.loads((ROOT / "strict_protocol_audit.json").read_text())
    template = json.loads((ROOT / "template_hash_audit.json").read_text())
    imported = json.loads((ROOT / "strict_validate.json").read_text())
    cleanup = json.loads((ROOT / "post_cleanup_health.json").read_text())
    manifest = {
        "schema_version": "qwen-thinking-raw-manifest-v1",
        "judge_id": raw["judge"]["judge_id"],
        "route": raw["judge"]["base_url"],
        "model_alias": raw["judge"]["model"],
        "provider": raw["judge"]["provider"],
        "packet_id": raw["packet_id"],
        "packet_sha256": raw["packet_sha256"],
        "instructions_sha256": raw["instructions_sha256"],
        "raw": {
            "path": str(raw_path),
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "status": raw["status"],
            "complete": raw["complete"],
            "counts": raw["counts"],
            "raw_events": len(raw.get("raw_events", [])),
        },
        "settings": raw["settings"],
        "validation": {
            "validate_import": imported["status"],
            "strict_protocol": strict["status"],
            "template_hash": template["status"],
            "token_budget": {
                "rows": token_audit["counts"]["rows"],
                "tokenized": token_audit["counts"]["tokenized"],
                "audit_failures": token_audit["counts"]["audit_failures"],
                "reasoning_nonempty": token_audit["counts"]["reasoning_nonempty"],
                "reasoning_tokens_min": min(r["offline_reasoning_tokens"] for r in token_audit["rows"]),
                "reasoning_tokens_max": max(r["offline_reasoning_tokens"] for r in token_audit["rows"]),
                "final_tokens_min": min(r["offline_completion_tokens"] for r in token_audit["rows"]),
                "final_tokens_max": max(r["offline_completion_tokens"] for r in token_audit["rows"]),
            },
        },
        "cleanup": cleanup,
        "scope_controls": {
            "frozen_packet_only": True,
            "frozen_instructions_only": True,
            "candidate_owner_map_in_judge_context": False,
            "qualification_hidden_key_in_judge_context": False,
            "prior_scores_in_judge_context": False,
            "score_selected_retries": 0,
            "extra_material_or_repeats": 0,
            "formal_8080_modified": False,
            "litellm_called": False,
            "cloud_called": False,
        },
        "artifacts": [digest(ROOT / name) for name in ARTIFACTS],
    }
    out = ROOT / "raw_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "raw_sha256": manifest["raw"]["sha256"], "artifacts": len(ARTIFACTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
