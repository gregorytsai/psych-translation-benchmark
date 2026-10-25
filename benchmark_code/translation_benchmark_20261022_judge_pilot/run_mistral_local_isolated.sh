#!/bin/zsh
# Review/pilot helper for the exact local Mistral Medium 3.5 GGUF.
# It never touches models.ini, launchd, the shared router, or another PID.

set -euo pipefail

pilot_root="/benchmark/translation_benchmark_20261022_judge_pilot"
runtime_bin="/benchmark/llama-v0.7.0/build-release/bin/llama-server"
mistral_model="/benchmark/Mistral-Medium-3.5-128B/Mistral-Medium-3.5-128B-UD-Q4_K_XL-00001-of-00003.gguf"
listen_host="127.0.0.1"
listen_port="18080"
state_dir="/private/tmp/mistral_medium_isolated_20261022"
pid_file="$state_dir/server.pid"
log_path_file="$state_dir/server.log.path"
log_file="$state_dir/server.log"

mkdir -p "$state_dir"

is_listening() {
  lsof -nP -iTCP:"$listen_port" -sTCP:LISTEN 2>/dev/null | grep -q .
}

start_server() {
  if is_listening; then
    echo "refusing start: TCP $listen_port is already listening" >&2
    return 2
  fi
  if [[ ! -x "$runtime_bin" || ! -f "$mistral_model" ]]; then
    echo "runtime or exact Mistral shard is missing" >&2
    return 2
  fi
  if [[ -s "$pid_file" ]]; then
    old_pid="$(<"$pid_file")"
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "refusing start: recorded isolated PID $old_pid is still alive" >&2
      return 2
    fi
  fi
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  log_file="$state_dir/server-$run_stamp.log"
  printf '%s\n' "$log_file" > "$log_path_file"
  : > "$log_file"
  "$runtime_bin" \
    --model "$mistral_model" \
    --alias mistral-medium-3.5-128b-isolated \
    --host "$listen_host" \
    --port "$listen_port" \
    --jinja \
    --no-webui \
    --offline \
    --n-gpu-layers all \
    --flash-attn on \
    --ctx-size 8192 \
    --parallel 1 \
    --no-cont-batching \
    --fit on \
    --fit-target 16384 \
    --fit-ctx 8192 \
    --load-mode mmap \
    --no-cache-prompt \
    --reasoning off \
    --reasoning-format none \
    --no-reasoning-preserve \
    --log-verbosity 2 \
    >"$log_file" 2>&1 &
  server_pid=$!
  printf '%s\n' "$server_pid" > "$pid_file"
  echo "started isolated Mistral PID=$server_pid port=$listen_port log=$log_file"
}

run_foreground() {
  if is_listening; then
    echo "refusing foreground start: TCP $listen_port is already listening" >&2
    return 2
  fi
  if [[ ! -x "$runtime_bin" || ! -f "$mistral_model" ]]; then
    echo "runtime or exact Mistral shard is missing" >&2
    return 2
  fi
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  log_file="$state_dir/server-$run_stamp.log"
  printf '%s\n' "$log_file" > "$log_path_file"
  printf '%s\n' "$$" > "$pid_file"
  exec "$runtime_bin" \
    --model "$mistral_model" \
    --alias mistral-medium-3.5-128b-isolated \
    --host "$listen_host" \
    --port "$listen_port" \
    --jinja \
    --no-webui \
    --offline \
    --n-gpu-layers all \
    --flash-attn on \
    --ctx-size 8192 \
    --parallel 1 \
    --no-cont-batching \
    --fit on \
    --fit-target 16384 \
    --fit-ctx 8192 \
    --load-mode mmap \
    --no-cache-prompt \
    --reasoning off \
    --reasoning-format none \
    --no-reasoning-preserve \
    --log-verbosity 2 \
    >"$log_file" 2>&1
}

stop_server() {
  if [[ ! -s "$pid_file" ]]; then
    echo "no recorded isolated PID" >&2
    return 1
  fi
  server_pid="$(<"$pid_file")"
  if [[ "$server_pid" != <-> ]] || (( server_pid <= 0 )); then
    echo "refusing stop: recorded PID is not a positive integer" >&2
    return 2
  fi
  if kill -0 "$server_pid" 2>/dev/null; then
    process_cmd="$(ps -p "$server_pid" -o command= 2>/dev/null || true)"
    if [[ "$process_cmd" != *"$runtime_bin"* || "$process_cmd" != *"--model $mistral_model"* || "$process_cmd" != *"--alias mistral-medium-3.5-128b-isolated"* || "$process_cmd" != *"--host $listen_host"* || "$process_cmd" != *"--port $listen_port"* ]]; then
      echo "refusing stop: PID $server_pid is not the exact isolated Mistral command" >&2
      return 2
    fi
    listener_pids="$(lsof -nP -iTCP:"$listen_port" -sTCP:LISTEN -t 2>/dev/null || true)"
    while IFS= read -r listener_pid; do
      [[ -z "$listener_pid" ]] && continue
      if [[ "$listener_pid" != "$server_pid" ]]; then
        echo "refusing stop: TCP $listen_port is also owned by unexpected PID $listener_pid" >&2
        return 2
      fi
    done <<< "$listener_pids"
    kill -TERM "$server_pid"
    for _ in {1..30}; do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      echo "isolated PID $server_pid did not exit after SIGTERM" >&2
      return 1
    fi
    echo "stopped isolated Mistral PID=$server_pid"
  else
    echo "isolated PID $server_pid is not running"
  fi
}

status_server() {
  if [[ -s "$log_path_file" ]]; then
    log_file="$(<"$log_path_file")"
  fi
  if [[ -s "$pid_file" ]]; then
    server_pid="$(<"$pid_file")"
    if kill -0 "$server_pid" 2>/dev/null; then
      echo "isolated PID=$server_pid port=$listen_port state=running log=$log_file"
      return 0
    fi
  fi
  echo "isolated Mistral state=not-running port=$listen_port"
  return 1
}

case "${1:-}" in
  start) start_server ;;
  foreground) run_foreground ;;
  stop) stop_server ;;
  status) status_server ;;
  *) echo "usage: $0 {start|foreground|stop|status}" >&2; exit 2 ;;
esac
