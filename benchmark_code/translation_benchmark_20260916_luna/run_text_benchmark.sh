#!/bin/zsh
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$(cd "$ROOT/.." && pwd)"
CLI="$WORK/llama-b10679/llama-completion"
INPUT="$ROOT/inputs.jsonl"
RUNS="$ROOT/raw_text_metal"
mkdir -p "$RUNS"

if [[ ! -x "$CLI" ]]; then
  print -u2 "missing executable: $CLI"
  exit 2
fi

typeset -a MODEL_IDS MODEL_PATHS
MODEL_IDS=(translategemma-27b translategemma-12b translategemma-4b gemma-4-e4b gemma-4-12b)
MODEL_PATHS=(
  "$WORK/translategemma-27b-it-q8_0.gguf"
  "$WORK/translategemma-12b-it.Q8_0.gguf"
  "$WORK/translategemma-4b-it.Q8_0.gguf"
  "$WORK/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
  "$WORK/Gemma4 12b/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
)

typeset -i count=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  id="$(print -r -- "$line" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
  english="$(print -r -- "$line" | sed -n 's/.*"english":"\(.*\)"}/\1/p' | sed 's/\\"/"/g')"
  [[ -z "$id" || -z "$english" ]] && { print -u2 "cannot parse input line: $line"; exit 3; }
  count=$((count + 1))
  for i in {1..${#MODEL_IDS}}; do
    mid="${MODEL_IDS[$i]}"
    model="${MODEL_PATHS[$i]}"
    stem="${mid}__${id}"
    out="$RUNS/$stem.stdout"
    err="$RUNS/$stem.stderr"
    meta="$RUNS/$stem.meta"
    if [[ ! -f "$model" ]]; then
      print -r -- "status=missing_model" > "$meta"
      print -r -- "model=$model" >> "$meta"
      continue
    fi
    instruction="You are a professional English (en) to Chinese (zh-TW) translator with clinical psychology and psychiatry expertise. Translate the English passage below faithfully into Traditional Chinese (zh-TW). Preserve every fact, number, medication name, dosage, symptom, uncertainty, and paragraph structure. Use standard Taiwan Traditional Chinese clinical terminology. Produce only the Chinese translation, without explanations or commentary."
    if [[ "$mid" == translategemma-* ]]; then
      prompt="<start_of_turn>user\n$instruction\n\nEnglish passage:\n$english<end_of_turn>\n<start_of_turn>model\n"
      format="manual Gemma turn tokens (TranslateGemma)"
    else
      prompt="<|turn>user\n$instruction\n\nEnglish passage:\n$english<turn|><|turn>model\n<|channel>final\n"
      format="manual Gemma4 turn/channel tokens"
    fi
    start_ns="$(python3 -c 'import time; print(time.time_ns())')"
    command="$CLI -m $model -no-cnv --reasoning off --temp 0 --top-k 1 --top-p 1 --min-p 0 --seed 123 --ctx-size 4096 -n 384 --threads -1 --threads-batch -1 --batch-size 512 --ubatch-size 512 --flash-attn on --no-display-prompt --single-turn -f /dev/stdin"
    print -r -- "status=started" > "$meta"
    print -r -- "model_id=$mid" >> "$meta"
    print -r -- "model_path=$model" >> "$meta"
    print -r -- "input_id=$id" >> "$meta"
    print -r -- "command=$command" >> "$meta"
    print -r -- "prompt_format=$format" >> "$meta"
    print -r -- "start_ns=$start_ns" >> "$meta"
    print -r -- "prompt=$prompt" >> "$meta"
    print -r -- "$prompt" | "$CLI" -m "$model" -no-cnv --reasoning off --temp 0 --top-k 1 --top-p 1 --min-p 0 --seed 123 --ctx-size 4096 -n 384 --threads -1 --threads-batch -1 --batch-size 512 --ubatch-size 512 --flash-attn on --no-display-prompt --single-turn -f /dev/stdin > "$out" 2> "$err"
    rc=$?
    end_ns="$(python3 -c 'import time; print(time.time_ns())')"
    print -r -- "end_ns=$end_ns" >> "$meta"
    print -r -- "wall_ms=$(( (end_ns - start_ns) / 1000000 ))" >> "$meta"
    print -r -- "exit_code=$rc" >> "$meta"
    print -r -- "status=finished" >> "$meta"
    print -r -- "stderr_tail=$(tail -8 "$err" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')" >> "$meta"
    print -r -- "completed $mid / $id (rc=$rc, wall_ms=$(( (end_ns - start_ns) / 1000000 )))"
  done
done < "$INPUT"

print -r -- "completed_cases=$count"
