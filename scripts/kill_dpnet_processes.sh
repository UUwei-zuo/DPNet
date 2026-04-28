#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMEOUT_LOOPS=20
SLEEP_SEC=0.2

kill_session_processes() {
  local signal_name="$1"
  local target_sid="$2"
  local session_pids

  [[ -n "${target_sid}" ]] || return 0
  session_pids="$(ps -eo sid=,pid= 2>/dev/null | awk -v sid="${target_sid}" '$1 == sid {print $2}' || true)"
  [[ -n "${session_pids}" ]] || return 0

  while IFS= read -r session_pid; do
    [[ -n "${session_pid}" ]] || continue
    kill "${signal_name}" "${session_pid}" 2>/dev/null || true
  done <<< "${session_pids}"
}

terminate_group_of_pid() {
  local pid="$1"
  local signal_name="$2"
  [[ -n "${pid}" ]] || return 0
  kill -0 "${pid}" 2>/dev/null || return 0

  local pgid=""
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')" || true

  if [[ -n "${pgid}" ]]; then
    kill "${signal_name}" -- "-${pgid}" 2>/dev/null || true
  else
    kill "${signal_name}" "${pid}" 2>/dev/null || true
  fi
}

terminate_matches() {
  local pattern="$1"
  local signal_name="$2"
  local pids=""
  pids="$(pgrep -f -- "${pattern}" 2>/dev/null || true)"
  [[ -n "${pids}" ]] || return 0

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    terminate_group_of_pid "${pid}" "${signal_name}"
  done <<< "${pids}"
}

wait_until_empty() {
  local pattern="$1"
  local i=0
  while (( i < TIMEOUT_LOOPS )); do
    if ! pgrep -f -- "${pattern}" >/dev/null 2>&1; then
      return
    fi
    sleep "${SLEEP_SEC}"
    ((i += 1))
  done
}

kill_dynabarn_explicit() {
  local signal_name="$1"
  local dynabarn_script="${REPO_ROOT}/scripts/carla_dynabarn.py"
  local dynabarn_pids=""

  dynabarn_pids="$(pgrep -f -- "${dynabarn_script}" 2>/dev/null || true)"
  [[ -n "${dynabarn_pids}" ]] || return 0

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    local pgid=""
    local sid=""
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')" || true
    sid="$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')" || true

    if [[ -n "${pgid}" ]]; then
      kill "${signal_name}" -- "-${pgid}" 2>/dev/null || true
    else
      kill "${signal_name}" "${pid}" 2>/dev/null || true
    fi

    if [[ -n "${sid}" ]]; then
      kill_session_processes "${signal_name}" "${sid}"
    fi
  done <<< "${dynabarn_pids}"
}

rosnode kill -a >/dev/null 2>&1 || true

kill_dynabarn_explicit "-TERM"
sleep "${SLEEP_SEC}"

patterns=(
  "${REPO_ROOT}/scripts/carla_dynabarn.py"
  "${REPO_ROOT}/src/dpnet_planner_ros/src/DPNet_planner.py"
  "roslaunch dpnet_bridge_overlay DPNet_bridge.launch"
  "roslaunch dpnet_planner_ros DPNet_planner.launch"
  "carla_ros_bridge.*bridge.py"
)

for pattern in "${patterns[@]}"; do
  terminate_matches "${pattern}" "-TERM"
done

for pattern in "${patterns[@]}"; do
  wait_until_empty "${pattern}"
done

for pattern in "${patterns[@]}"; do
  terminate_matches "${pattern}" "-KILL"
done

kill_dynabarn_explicit "-KILL"
