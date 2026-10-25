#!/bin/zsh
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$(cd "$ROOT/.." && pwd)"
CLI="/opt/homebrew/bin/llama-cli"
IMAGE="$ROOT/image_clinical_note.png"
RUNS="$ROOT/raw_image_llama10964"
ONLY_MODEL_ID="${ONLY_MODEL_ID:-}"
mkdir -p "$RUNS"

if [[ ! -x "$CLI" || ! -f "$IMAGE" ]]; then
  print -u2 "missing b10964 llama-cli or image"
  exit 2
fi

typeset -a MODEL_IDS MODEL_PATHS MMPROJ_PATHS
MODEL_IDS=(translategemma-27b translategemma-12b translategemma-4b gemma-4-e4b gemma-4-12b)
MODEL_PATHS=(
  "$WORK/translategemma-27b-it-q8_0.gguf"
  "$WORK/translategemma-12b-it.Q8_0.gguf"
  "$WORK/translategemma-4b-it.Q8_0.gguf"
  "$WORK/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
  "$WORK/Gemma4 12b/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
)
MMPROJ_PATHS=(
  "$WORK/translategemma-27b-it.mmproj-Q8_0.gguf"
  "$WORK/translategemma-12b-it.mmproj-Q8_0.gguf"
  "$WORK/translategemma-4b-it.mmproj-Q8_0.gguf"
  "$WORK/gemma-4-E4B-mmproj-F16.gguf"
  "$WORK/Gemma4 12b/mmproj-F16.gguf"
)

typeset -a TASK_IDS TASK_PROMPTS
TASK_IDS=(image_ocr image_translation)
TASK_PROMPTS=(
  "Transcribe all visible English text in the image exactly. Preserve line breaks, capitalization, punctuation, numbers, and the medication name. Output only the transcription."
  "Translate all English text in the image into Traditional Chinese. Output only the translation."
)

for i in {1..${#MODEL_IDS}}; do
    mid="${MODEL_IDS[$i]}"
    if [[ -n "$ONLY_MODEL_ID" && "$mid" != "$ONLY_MODEL_ID" ]]; then
      continue
    fi
  model="${MODEL_PATHS[$i]}"
  mmproj="${MMPROJ_PATHS[$i]}"
  for j in {1..${#TASK_IDS}}; do
    tid="${TASK_IDS[$j]}"
    prompt="${TASK_PROMPTS[$j]}"
    stem="${mid}__${tid}"
    out="$RUNS/$stem.stdout"
    err="$RUNS/$stem.stderr"
    meta="$RUNS/$stem.meta"
    if [[ ! -f "$model" || ! -f "$mmproj" ]]; then
      print -r -- "status=unavailable" > "$meta"
      print -r -- "model=$model" >> "$meta"
      print -r -- "mmproj=$mmproj" >> "$meta"
      continue
    fi
    if [[ "$mid" == translategemma-* ]]; then
      template_args=(--no-jinja --chat-template gemma --reasoning off)
      template_note="--no-jinja --chat-template gemma --reasoning off"
    else
      template_args=(--jinja --reasoning off)
      template_note="--jinja --reasoning off"
    fi
    start_ns="$(python3 -c 'import time; print(time.time_ns())')"
    command="$CLI -m $model --mmproj $mmproj ${template_note} --image $IMAGE --temp 0 --top-k 1 --top-p 1 --min-p 0 --seed 123 --ctx-size 4096 --batch-size 2048 --ubatch-size 2048 -n 256 --single-turn -p <prompt>"
    print -r -- "status=started" > "$meta"
    print -r -- "model_id=$mid" >> "$meta"
    print -r -- "task_id=$tid" >> "$meta"
    print -r -- "model_path=$model" >> "$meta"
    print -r -- "mmproj_path=$mmproj" >> "$meta"
    print -r -- "image_path=$IMAGE" >> "$meta"
    print -r -- "command=$command" >> "$meta"
    print -r -- "template_args=$template_note" >> "$meta"
    print -r -- "mtp=disabled (no draft model argument)" >> "$meta"
    print -r -- "prompt=$prompt" >> "$meta"
    print -r -- "start_ns=$start_ns" >> "$meta"
    "$CLI" -m "$model" --mmproj "$mmproj" "${template_args[@]}" --image "$IMAGE" --temp 0 --top-k 1 --top-p 1 --min-p 0 --seed 123 --ctx-size 4096 --batch-size 2048 --ubatch-size 2048 -n 256 --single-turn -p "$prompt" > "$out" 2> "$err"
    rc=$?
    end_ns="$(python3 -c 'import time; print(time.time_ns())')"
    print -r -- "end_ns=$end_ns" >> "$meta"
    print -r -- "wall_ms=$(( (end_ns - start_ns) / 1000000 ))" >> "$meta"
    print -r -- "exit_code=$rc" >> "$meta"
    print -r -- "status=finished" >> "$meta"
    print -r -- "completed $mid / $tid (rc=$rc, wall_ms=$(( (end_ns - start_ns) / 1000000 )))"
  done
done
