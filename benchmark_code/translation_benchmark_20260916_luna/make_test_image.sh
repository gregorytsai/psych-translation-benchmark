#!/bin/zsh
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
python3 "$ROOT/make_test_image.py" "$ROOT/image_clinical_note.png"
file "$ROOT/image_clinical_note.png"
