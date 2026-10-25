#!/usr/bin/env python3
"""Validate the isolated Qwen native template/alias evidence captured before the run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


EXPECTED_ALIAS = "qwen3.8-flash-next-isolated"
EXPECTED_MODEL = "/benchmark/Qwen3.8-Flash-Next-UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf"
EXPECTED_TEMPLATE_SHA = "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"
EXPECTED_BUILD = "b524-620d2225c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    args = ap.parse_args()
    root = args.root
    props_path = root / "props.json"
    render_path = root / "template_render_xhigh_false.json"
    models_path = root / "models.json"
    props: dict[str, Any] = json.loads(props_path.read_text())
    render: dict[str, Any] = json.loads(render_path.read_text())
    models: dict[str, Any] = json.loads(models_path.read_text())
    errors: list[str] = []
    template = props.get("chat_template") or ""
    prompt = render.get("prompt") or ""
    if props.get("model_alias") != EXPECTED_ALIAS:
        errors.append(f"model_alias mismatch: {props.get('model_alias')!r}")
    if props.get("model_path") != EXPECTED_MODEL:
        errors.append(f"model_path mismatch: {props.get('model_path')!r}")
    if props.get("build_info") != EXPECTED_BUILD:
        errors.append(f"build_info mismatch: {props.get('build_info')!r}")
    if hashlib.sha256(template.encode()).hexdigest() != EXPECTED_TEMPLATE_SHA:
        errors.append("chat_template sha256 mismatch")
    if len(template) != 9993:
        errors.append(f"chat_template length mismatch: {len(template)}")
    if ((props.get("default_generation_settings") or {}).get("n_ctx")) != 32768:
        errors.append("props n_ctx mismatch")
    caps = props.get("chat_template_caps") or {}
    for key in ("supports_reasoning_effort", "supports_preserve_reasoning", "supports_tools"):
        if caps.get(key) is not True:
            errors.append(f"chat_template_caps.{key} is not true")
    if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
        errors.append("rendered prompt does not end with native assistant thinking prefix")
    if prompt.count("<|im_start|>assistant") != 1 or "</think>" in prompt:
        errors.append("rendered prompt assistant/thinking sentinels mismatch")
    if "QWEN_PREFLIGHT_SYSTEM" not in prompt or "QWEN_PREFLIGHT_USER" not in prompt:
        errors.append("render smoke markers missing")
    model_ids = [m.get("id") for m in models.get("data", [])]
    if EXPECTED_ALIAS not in model_ids:
        errors.append("/v1/models alias missing")
    model_meta = next((m for m in models.get("data", []) if m.get("id") == EXPECTED_ALIAS), {})
    if (model_meta.get("meta") or {}).get("n_ctx") != 32768:
        errors.append("/v1/models n_ctx mismatch")
    output = {
        "schema_version": "qwen-native-template-hash-audit-v1",
        "status": "PASS" if not errors else "FAIL",
        "props_path": str(props_path),
        "props_sha256": hashlib.sha256(props_path.read_bytes()).hexdigest(),
        "render_path": str(render_path),
        "render_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
        "models_path": str(models_path),
        "models_sha256": hashlib.sha256(models_path.read_bytes()).hexdigest(),
        "alias": props.get("model_alias"),
        "model_path": props.get("model_path"),
        "build_info": props.get("build_info"),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "template_length": len(template),
        "render_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "render_prompt_length": len(prompt),
        "render_native_thinking_prefix": prompt.endswith("<|im_start|>assistant\n<think>\n"),
        "render_has_close_think": "</think>" in prompt,
        "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
        "chat_template_caps": caps,
        "models_aliases": model_ids,
        "errors": errors,
        "note": "Captured from the isolated 18082 endpoint; formal 8080 configuration was not read or modified.",
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": output["status"], "errors": len(errors)}, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
