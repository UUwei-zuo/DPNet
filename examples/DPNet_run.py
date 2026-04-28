#!/usr/bin/env python3

"""Supervisor for DPNet runs with collision-triggered retries."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


class DPNetRunner:
    """Run DPNet entry script and retry on collision events."""

    def __init__(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.hyperparam_path = self.repo_root / "examples" / "DPNet_hyperparameters.yaml"
        self.entry_script = self.repo_root / "scripts" / "DPNet_entry.sh"
        self.kill_script = self.repo_root / "scripts" / "kill_dpnet_processes.sh"
        self.robot_id = os.environ.get("DPNET_ROBOT_ID", "agent_1")
        self.collision_topic = f"/carla/{self.robot_id}/collision"
        self.control_topic = f"/carla/{self.robot_id}/vehicle_control_cmd"
        self.arrival_topic = f"/carla/{self.robot_id}/arrival"
        self.entry_process = None
        self.collision_watch_process = None
        self.control_watch_process = None
        self.arrival_watch_process = None
        self._printed_control_start = False
        self._printed_arrival = False
        self._stop_requested = False
        self._install_signal_handlers()

    def _install_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, _frame):
        self._stop_requested = True
        print(f"\n[DPNet_run] Received signal {signum}. Stopping all DPNet processes...")
        self._terminate_current_attempt(force=True)
        sys.exit(130 if signum == signal.SIGINT else 143)

    def _load_runtime_config(self):
        try:
            with self.hyperparam_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Hyperparameter file not found: {self.hyperparam_path}"
            )

        runtime_cfg = data.get("DPNet_runtime", {})
        retry_enabled = bool(runtime_cfg.get("enable_collision_retry", True))
        max_retry_times = int(runtime_cfg.get("max_retry_times", 3))
        retry_delay_sec = float(runtime_cfg.get("retry_delay_sec", 2.0))
        if max_retry_times < 0:
            max_retry_times = 0
        if retry_delay_sec < 0:
            retry_delay_sec = 0.0
        return retry_enabled, max_retry_times, retry_delay_sec

    @staticmethod
    def _terminate_process_group(proc, sig=signal.SIGTERM):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return

    def _wait_for_process_exit(self, proc, timeout_sec=8.0):
        if proc is None:
            return
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self._terminate_process_group(proc, sig=signal.SIGKILL)
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass

    def _force_cleanup(self):
        try:
            subprocess.run(
                ["bash", str(self.kill_script)],
                cwd=str(self.repo_root),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def _terminate_current_attempt(self, force=False):
        self._terminate_process_group(self.collision_watch_process, sig=signal.SIGTERM)
        self._wait_for_process_exit(self.collision_watch_process, timeout_sec=2.0)
        self.collision_watch_process = None
        self._terminate_process_group(self.control_watch_process, sig=signal.SIGTERM)
        self._wait_for_process_exit(self.control_watch_process, timeout_sec=2.0)
        self.control_watch_process = None
        self._terminate_process_group(self.arrival_watch_process, sig=signal.SIGTERM)
        self._wait_for_process_exit(self.arrival_watch_process, timeout_sec=2.0)
        self.arrival_watch_process = None

        self._terminate_process_group(self.entry_process, sig=signal.SIGTERM)
        self._wait_for_process_exit(self.entry_process, timeout_sec=8.0)
        self.entry_process = None

        if force:
            self._force_cleanup()

    def _start_attempt(self, attempt_idx):
        print(f"[DPNet_run] Starting attempt {attempt_idx} using {self.entry_script} ...")
        self.entry_process = subprocess.Popen(
            ["bash", str(self.entry_script)],
            cwd=str(self.repo_root),
            preexec_fn=os.setsid,
        )
        self._printed_control_start = False
        self._printed_arrival = False
        self._start_collision_watcher()
        self._start_control_watcher()
        self._start_arrival_watcher()

    def _start_collision_watcher(self):
        """Start a one-shot watcher that exits with 0 on first collision message."""
        self.collision_watch_process = subprocess.Popen(
            ["rostopic", "echo", "-n", "1", self.collision_topic],
            cwd=str(self.repo_root),
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _start_control_watcher(self):
        """Start a one-shot watcher that exits with 0 on first control command message."""
        self.control_watch_process = subprocess.Popen(
            ["rostopic", "echo", "-n", "1", self.control_topic],
            cwd=str(self.repo_root),
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _start_arrival_watcher(self):
        """Start a one-shot watcher that exits with 0 on first arrival message."""
        self.arrival_watch_process = subprocess.Popen(
            ["rostopic", "echo", "-n", "1", self.arrival_topic],
            cwd=str(self.repo_root),
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _check_attempt_state(self):
        """Return (finished, collided, entry_exit_code)."""
        if self.entry_process is None:
            return True, False, 1

        entry_code = self.entry_process.poll()
        if entry_code is not None:
            return True, False, entry_code

        watch_code = None if self.collision_watch_process is None else self.collision_watch_process.poll()
        if watch_code == 0:
            print(f"[DPNet_run] Collision detected from topic {self.collision_topic}.")
            return True, True, None
        if watch_code is not None and watch_code != 0:
            self._terminate_process_group(self.collision_watch_process, sig=signal.SIGTERM)
            self._wait_for_process_exit(self.collision_watch_process, timeout_sec=1.0)
            self.collision_watch_process = None
            if self.entry_process is not None and self.entry_process.poll() is None:
                self._start_collision_watcher()
                return False, False, None

        control_code = None if self.control_watch_process is None else self.control_watch_process.poll()
        if control_code == 0 and not self._printed_control_start:
            print("\n########   Control Start   ########\n")
            self._printed_control_start = True
            self._terminate_process_group(self.control_watch_process, sig=signal.SIGTERM)
            self._wait_for_process_exit(self.control_watch_process, timeout_sec=1.0)
            self.control_watch_process = None
        if control_code is not None and control_code != 0:
            self._terminate_process_group(self.control_watch_process, sig=signal.SIGTERM)
            self._wait_for_process_exit(self.control_watch_process, timeout_sec=1.0)
            self.control_watch_process = None
            if (
                self.entry_process is not None
                and self.entry_process.poll() is None
                and not self._printed_control_start
            ):
                self._start_control_watcher()
                return False, False, None

        arrival_code = None if self.arrival_watch_process is None else self.arrival_watch_process.poll()
        if arrival_code == 0 and not self._printed_arrival:
            print("\n###########   Arrival   #############\n")
            self._printed_arrival = True
            self._terminate_process_group(self.arrival_watch_process, sig=signal.SIGTERM)
            self._wait_for_process_exit(self.arrival_watch_process, timeout_sec=1.0)
            self.arrival_watch_process = None
        if arrival_code is not None and arrival_code != 0:
            self._terminate_process_group(self.arrival_watch_process, sig=signal.SIGTERM)
            self._wait_for_process_exit(self.arrival_watch_process, timeout_sec=1.0)
            self.arrival_watch_process = None
            if (
                self.entry_process is not None
                and self.entry_process.poll() is None
                and not self._printed_arrival
            ):
                self._start_arrival_watcher()
                return False, False, None

        return False, False, None

    def run(self):
        if not self.entry_script.exists():
            raise FileNotFoundError(f"Entry script not found: {self.entry_script}")

        retry_enabled, max_retry_times, retry_delay_sec = self._load_runtime_config()
        total_attempts = max_retry_times + 1 if retry_enabled else 1

        print(
            "[DPNet_run] Runtime config: "
            f"retry_enabled={retry_enabled}, max_retry_times={max_retry_times}, "
            f"retry_delay_sec={retry_delay_sec}"
        )
        print(f"[DPNet_run] Collision watch topic: {self.collision_topic}")

        for attempt_idx in range(1, total_attempts + 1):
            if self._stop_requested:
                return 130

            self._start_attempt(attempt_idx)

            collided = False
            entry_code = None
            while True:
                finished, collided, entry_code = self._check_attempt_state()
                if finished:
                    break
                time.sleep(0.2)

            if collided:
                self._terminate_current_attempt(force=True)
                if attempt_idx < total_attempts:
                    print(
                        f"[DPNet_run] Retrying after collision "
                        f"({attempt_idx}/{total_attempts}) in {retry_delay_sec:.1f}s..."
                    )
                    time.sleep(retry_delay_sec)
                    continue
                print("[DPNet_run] Collision retry limit reached. Run failed.")
                return 1

            self._terminate_current_attempt(force=False)
            if entry_code == 0:
                print("[DPNet_run] Entry exited normally without collision-triggered retry.")
                return 0

            print(f"[DPNet_run] Entry exited with code {entry_code}.")
            if attempt_idx < total_attempts:
                print(f"[DPNet_run] Retrying failed run in {retry_delay_sec:.1f}s...")
                time.sleep(retry_delay_sec)
                continue
            return entry_code if entry_code is not None else 1

        return 0


def main():
    runner = DPNetRunner()
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
