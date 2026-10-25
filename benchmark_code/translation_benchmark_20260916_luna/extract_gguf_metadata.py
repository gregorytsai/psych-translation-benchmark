#!/usr/bin/env python3
"""Read GGUF header metadata without third-party dependencies.

This is intentionally small and read-only with respect to model files.  It
extracts scalar/string metadata, including tokenizer.chat_template, and skips
the very large tokenizer arrays while retaining their element counts.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path


TYPE_NAMES = [
    "uint8", "int8", "uint16", "int16", "uint32", "int32", "float32",
    "bool", "string", "array", "uint64", "int64", "float64",
]
SCALAR_FORMATS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}


def read_scalar(handle, value_type: int):
    size = struct.calcsize("<" + SCALAR_FORMATS[value_type])
    return struct.unpack("<" + SCALAR_FORMATS[value_type], handle.read(size))[0]


def read_value(handle, value_type: int, keep_array: bool = False):
    if value_type == 8:
        length = struct.unpack("<Q", handle.read(8))[0]
        value = handle.read(length)
        return value.decode("utf-8", errors="replace")
    if value_type == 9:
        element_type = struct.unpack("<I", handle.read(4))[0]
        length = struct.unpack("<Q", handle.read(8))[0]
        if keep_array and length <= 128:
            return [read_value(handle, element_type, keep_array=True) for _ in range(length)]
        for _ in range(length):
            read_value(handle, element_type, keep_array=False)
        return {
            "type": TYPE_NAMES[element_type],
            "length": length,
        }
    return read_scalar(handle, value_type)


def read_gguf(path: str) -> dict:
    metadata = {}
    with open(path, "rb") as handle:
        magic = handle.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: invalid GGUF magic {magic!r}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", handle.read(20))
        for _ in range(kv_count):
            key_length = struct.unpack("<Q", handle.read(8))[0]
            key = handle.read(key_length).decode("utf-8", errors="replace")
            value_type = struct.unpack("<I", handle.read(4))[0]
            metadata[key] = read_value(
                handle,
                value_type,
                keep_array=key not in {
                    "tokenizer.ggml.tokens",
                    "tokenizer.ggml.scores",
                    "tokenizer.ggml.token_type",
                    "tokenizer.ggml.merges",
                },
            )
    return {
        "path": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "version": version,
        "tensor_count": tensor_count,
        "kv_count": kv_count,
        "metadata": metadata,
    }


def main() -> int:
    output_json = None
    template_dir = None
    paths = []
    args = iter(sys.argv[1:])
    for arg in args:
        if arg == "--output-json":
            output_json = next(args)
        elif arg == "--template-dir":
            template_dir = next(args)
        else:
            paths.append(arg)
    if not paths:
        raise SystemExit("usage: extract_gguf_metadata.py MODEL.gguf [...]")
    output = [read_gguf(path) for path in paths]
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if output_json:
        Path(output_json).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if template_dir:
        target = Path(template_dir)
        target.mkdir(parents=True, exist_ok=True)
        for model in output:
            template = model["metadata"].get("tokenizer.chat_template")
            if template is None:
                continue
            stem = Path(model["path"]).name.replace(".gguf", "")
            (target / f"{stem}.jinja").write_text(template, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
