#!/bin/zsh
# Independent guarded server helper for the registered Mistral targeted run.
# This file never edits models.ini, launchd, the formal router, LiteLLM, or
# the existing Mistral off/high helpers.

set -euo pipefail

target_root="/benchmark/translation_benchmark_20261022_judge_pilot/mistral_targeted_retest_20261023"
raw_root="$target_root/raw/mistral-medium-3.5-128b-local-thinking-high-targeted-20261023"
runtime_bin="/benchmark/llama-v0.7.0/build-release/bin/llama-server"
mistral_model="/benchmark/Mistral-Medium-3.5-128B/Mistral-Medium-3.5-128B-UD-Q4_K_XL-00001-of-00003.gguf"
listen_host="127.0.0.1"
listen_port="18080"
alias="mistral-medium-3.5-128b-targeted-thinking-high-20261023"
pid_file="$raw_root/server.pid"
command_file="$raw_root/server.command.txt"
log_file="$raw_root/server.log"

mkdir -p "$raw_root"

is_listening() {
  lsof -nP -iTCP:"$listen_port" -sTCP:LISTEN 2>/dev/null | grep -q .
}

exact_command() {
  cat <<EOF
"$runtime_bin" \\
  --model "$mistral_model" \\
  --alias "$alias" \\
  --host "$listen_host" \\
  --port "$listen_port" \\
  --jinja --no-webui --offline \\
  --n-gpu-layers all --flash-attn on \\
  --ctx-size 32768 --parallel 1 --no-cont-batching \\
  --fit on --fit-target 32768 --fit-ctx 32768 \\
  --load-mode mmap --no-cache-prompt \\
  --reasoning auto --reasoning-format deepseek \\
  --reasoning-budget 8192 --no-reasoning-preserve \\
  --log-verbosity 2
EOF
}

start_foreground() {
  if is_listening; then
    echo "refusing start: TCP $listen_port is already listening" >&2
    return 2
  fi
  if [[ ! -x "$runtime_bin" || ! -f "$mistral_model" ]]; then
    echo "runtime or exact Mistral first shard is missing" >&2
    return 2
  fi
  if [[ -s "$pid_file" ]]; then
    old_pid="$(<"$pid_file")"
    if [[ "$old_pid" == <-> ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "refusing start: recorded targeted PID $old_pid is still alive" >&2
      return 2
    fi
  fi
  exact_command > "$command_file"
  : > "$log_file"
  printf '%s\n' "$$" > "$pid_file"
  exec "$runtime_bin" \
    --model "$mistral_model" \
    --alias "$alias" \
    --host "$listen_host" \
    --port "$listen_port" \
    --jinja --no-webui --offline \
    --n-gpu-layers all --flash-attn on \
    --ctx-size 32768 --parallel 1 --no-cont-batching \
    --fit on --fit-target 32768 --fit-ctx 32768 \
    --load-mode mmap --no-cache-prompt \
    --reasoning auto --reasoning-format deepseek \
    --reasoning-budget 8192 --no-reasoning-preserve \
    --log-verbosity 2 > "$log_file" 2>&1
}

stop_guarded() {
  if [[ ! -s "$pid_file" ]]; then
    echo "no targeted PID file" >&2
    return 1
  fi
  targeted_pid="$(<"$pid_file")"
  if [[ "$targeted_pid" != <-> ]] || (( targeted_pid <= 0 )); then
    echo "refusing stop: targeted PID is not a positive integer" >&2
    return 2
  fi
  if ! kill -0 "$targeted_pid" 2>/dev/null; then
    echo "targeted PID $targeted_pid is not running" >&2
    return 0
  fi
  process_cmd="$(ps -p "$targeted_pid" -o command= 2>/dev/null || true)"
  if [[ "$process_cmd" != *"$runtime_bin"* || "$process_cmd" != *"--model $mistral_model"* || "$process_cmd" != *"--alias $alias"* || "$process_cmd" != *"--port $listen_port"* ]]; then
    echo "refusing stop: PID $targeted_pid is not the exact targeted Mistral command" >&2
    return 2
  fi
  listener_pids="$(lsof -nP -iTCP:"$listen_port" -sTCP:LISTEN -t 2>/dev/null || true)"
  while IFS= read -r listener_pid; do
    [[ -z "$listener_pid" ]] && continue
    if [[ "$listener_pid" != "$targeted_pid" ]]; then
      echo "refusing stop: TCP $listen_port also has unexpected PID $listener_pid" >&2
      return 2
    fi
  done <<< "$listener_pids"
  kill -INT "$targeted_pid"
  for _ in {1..60}; do
    kill -0 "$targeted_pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$targeted_pid" 2>/dev/null; then
    echo "targeted PID $targeted_pid did not exit after SIGINT" >&2
    return 1
  fi
  echo "stopped targeted Mistral PID=$targeted_pid"
}

status_targeted() {
  if [[ -s "$pid_file" ]]; then
    targeted_pid="$(<"$pid_file")"
    if kill -0 "$targeted_pid" 2>/dev/null; then
      echo "targeted PID=$targeted_pid port=$listen_port state=running log=$log_file"
      return 0
    fi
  fi
  echo "targeted Mistral state=not-running port=$listen_port"
  return 1
}

case "${1:-}" in
  foreground) start_foreground ;;
  stop) stop_guarded ;;
  status) status_targeted ;;
  command) exact_command ;;
  *) echo "usage: $0 {foreground|stop|status|command}" >&2; exit 2 ;;
esac
