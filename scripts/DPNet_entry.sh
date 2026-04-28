#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HYPERPARAM_FILE="${REPO_ROOT}/examples/DPNet_hyperparameters.yaml"

source "${REPO_ROOT}/devel/setup.bash"
export PYTHONWARNINGS="ignore::FutureWarning:cvxpy.reductions.solvers.solving_chain"

cleanup() {
  if [[ "${_cleanup_running:-0}" -eq 1 ]]; then
    return
  fi
  _cleanup_running=1
  trap - EXIT INT TERM

  terminate_group() {
    local pid="$1"
    [[ -n "${pid}" ]] || return
    kill -0 "${pid}" 2>/dev/null || return

    local pgid=""
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')" || true

    if [[ -n "${pgid}" ]]; then
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    else
      kill -TERM "${pid}" 2>/dev/null || true
    fi

    for _ in {1..30}; do
      kill -0 "${pid}" 2>/dev/null || return
      sleep 0.2
    done

    if [[ -n "${pgid}" ]]; then
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    else
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  }

  rosnode kill -a >/dev/null 2>&1 || true
  sleep 1

  terminate_group "${node_pid:-}"
  terminate_group "${traffic_pid:-}"
  terminate_group "${sim_pid:-}"

  bash "${REPO_ROOT}/scripts/kill_dpnet_processes.sh" >/dev/null 2>&1 || true

  wait "${node_pid:-}" 2>/dev/null || true
  wait "${traffic_pid:-}" 2>/dev/null || true
  wait "${sim_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 "${REPO_ROOT}/scripts/update_file_path.py"

python3 "${REPO_ROOT}/scripts/load_world.py" --minimal-layers
sleep 1
export CARLA_SKIP_TOWN_LOADING=true

setsid roslaunch dpnet_bridge_overlay DPNet_bridge.launch auto_control:=true include_static_props:=true &
sim_pid=$!
sleep 2

rosparam load "${HYPERPARAM_FILE}" /DPNet_config
DYNABARN_VEHICLES="$(rosparam get /DPNet_config/Carla_DynaBARN/vehicles 2>/dev/null || echo 4)"
DYNABARN_SEED="$(rosparam get /DPNet_config/Carla_DynaBARN/seed 2>/dev/null || echo 6)"

python3 "${REPO_ROOT}/scripts/carla_dynabarn.py" --vehicles "${DYNABARN_VEHICLES}" --seed "${DYNABARN_SEED}" &
traffic_pid=$!
sleep 5

setsid roslaunch dpnet_planner_ros DPNet_planner.launch &
node_pid=$!
wait "${node_pid}"
