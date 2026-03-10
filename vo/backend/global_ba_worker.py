from __future__ import annotations

import threading
import time

from vo.backend.local_ba import LocalBundleAdjuster


class AsyncGlobalBAWorker:
    def __init__(self, map_manager, optimizer: LocalBundleAdjuster, max_version_lag: int = 200) -> None:
        self.map_manager = map_manager
        self.optimizer = optimizer
        self.max_version_lag = max_version_lag

        self._stop_event = threading.Event()
        self._work_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        self.run_count = 0
        self.success_count = 0
        self.drop_count = 0

    def start(self) -> None:
        self._thread.start()

    def request_optimize(self) -> None:
        self._work_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._work_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            fired = self._work_event.wait(timeout=0.1)
            if self._stop_event.is_set():
                break
            if not fired:
                continue

            self._work_event.clear()

            snapshot = self.map_manager.snapshot_for_global_ba(
                min_landmark_observations=self.optimizer.min_observations
            )
            if snapshot is None:
                continue

            self.run_count += 1
            result = self.optimizer.optimize_snapshot(snapshot)
            if result is None:
                continue

            applied = self.map_manager.apply_local_ba_result(
                result=result,
                max_version_lag=self.max_version_lag,
            )
            if applied:
                self.success_count += 1
            else:
                self.drop_count += 1

            time.sleep(0.001)