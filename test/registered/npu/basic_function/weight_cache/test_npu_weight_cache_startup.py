import os
import re
import socket
import subprocess
import sys
import time
import unittest

from sglang.srt.platforms import current_platform
from sglang.srt.utils import kill_process_tree
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

# A 32B dense model makes the disk-read + device-copy + TP-shard savings of the
# weight cache dominate over the fixed IPC overhead, so the warm-import speedup
# is observable and stable. TP=2 on 2 cards keeps each shard ~32GB.
DEFAULT_MODEL = QWEN3_32B_WEIGHTS_PATH
TP_SIZE = 2

register_npu_ci(est_time=1800, suite="full-2-npu-a3", nightly=True)

# Log markers emitted by the weight-cache pipeline. We parse these rather than
# re-timing the loads ourselves so the numbers match what production logs report.
BASELINE_RE = re.compile(r"Load weight end\.\s*elapsed=([0-9.]+)\s*s")
DAEMON_DISK_RE = re.compile(r"Model loaded from disk in ([0-9.]+)s")
IPC_TOTAL_RE = re.compile(r"Loaded model via IPC.*?total=([0-9.]+)s")
IPC_MARKER = "Loaded model via IPC"

BASELINE_STDOUT = "/tmp/test_npu_weight_cache_startup_baseline_stdout.log"
BASELINE_STDERR = "/tmp/test_npu_weight_cache_startup_baseline_stderr.log"
DAEMON_STDERR = "/tmp/test_npu_weight_cache_startup_daemon.log"
CLIENT_STDOUT = "/tmp/test_npu_weight_cache_startup_client_stdout.log"
CLIENT_STDERR = "/tmp/test_npu_weight_cache_startup_client_stderr.log"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_logs(paths) -> str:
    logs = ""
    for path in paths:
        if os.path.exists(path):
            with open(path, errors="replace") as f:
                logs += f.read()
    return logs


def _parse_metric(logs: str, pattern: re.Pattern):
    m = pattern.search(logs)
    return float(m.group(1)) if m else None


def _ping_daemon(device_uuid: str) -> bool:
    """Confirm the daemon's serve socket answers a ping, not just that the
    .ready file exists. The ready file can briefly outlive a daemon that
    started serving and then died."""
    from sglang.srt.weight_cache.protocol import recv_msg, send_msg

    sock_path = get_socket_path(device_uuid)
    if not os.path.exists(sock_path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect(sock_path)
            send_msg(s, {"type": "ping"})
            resp = recv_msg(s)
            return resp.get("status") == "ok"
    except Exception:
        return False


class TestNpuWeightCacheStartup(CustomTestCase):
    """Benchmark the weight-loading stage three ways and assert the weight cache
    (warm zero-copy IPC import) is faster than a cold disk load.

    - ``baseline``: server loads weights straight from disk (weight cache OFF);
      this is the cold-start cost every server pays without a daemon.
    - ``daemon``:   one-time disk load moved into the weight-cache daemon.
    - ``client``:   server imports the daemon's weights over zero-copy IPC
      (weight cache ON); this is the warm-start cost the cache optimizes.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL
        cls.tp_size = TP_SIZE
        cls.npu_uuids = [current_platform.get_device_uuid(i) for i in range(cls.tp_size)]

        # Clean up stale ready/socket files from previous runs.
        for device_uuid in cls.npu_uuids:
            for path in (get_ready_path(device_uuid), get_socket_path(device_uuid)):
                if os.path.exists(path):
                    os.unlink(path)

        npu_backend_args = ["--attention-backend", "ascend", "--disable-cuda-graph"]

        # Step 1: baseline — cold load straight from disk (weight cache OFF).
        cls.baseline_stdout = open(BASELINE_STDOUT, "w")
        cls.baseline_stderr = open(BASELINE_STDERR, "w")
        baseline_url = f"http://127.0.0.1:{_find_free_port()}"
        cls.baseline_process = popen_launch_server(
            cls.model,
            baseline_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp", str(cls.tp_size), *npu_backend_args],
            return_stdout_stderr=(cls.baseline_stdout, cls.baseline_stderr),
        )
        cls.baseline_logs = _read_logs([BASELINE_STDOUT, BASELINE_STDERR])
        cls.baseline_time = _parse_metric(cls.baseline_logs, BASELINE_RE)
        kill_process_tree(cls.baseline_process.pid)
        cls.baseline_process = None

        # Step 2: daemon — one-time disk load moved out of the serving process.
        cls.daemon_stderr = open(DAEMON_STDERR, "w")
        cls.daemon_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                cls.model,
                "--tp-size",
                str(cls.tp_size),
            ],
            stdout=cls.daemon_stderr,
            stderr=cls.daemon_stderr,
        )
        start = time.time()
        for device_uuid in cls.npu_uuids:
            ready_path = get_ready_path(device_uuid)
            while not os.path.exists(ready_path):
                if time.time() - start > DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH:
                    kill_process_tree(cls.daemon_process.pid)
                    raise TimeoutError(
                        f"Weight cache daemon for NPU {device_uuid} not ready "
                        f"within {DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH}s"
                    )
                if cls.daemon_process.poll() is not None:
                    raise RuntimeError(
                        f"Weight cache daemon exited prematurely with code "
                        f"{cls.daemon_process.returncode}"
                    )
                time.sleep(2)
        cls.daemon_logs = _read_logs([DAEMON_STDERR])
        cls.daemon_time = _parse_metric(cls.daemon_logs, DAEMON_DISK_RE)

        # The .ready file can linger after a daemon that finished serving and
        # then died. Confirm the daemon is actually alive and reachable before
        # launching the client, so a dead daemon surfaces its log here instead
        # of as a confusing client_time=None (client silently fell back to disk).
        if cls.daemon_process.poll() is not None:
            raise RuntimeError(
                f"Weight cache daemon exited with code {cls.daemon_process.returncode} "
                f"before the client launched. Daemon log:\n{cls.daemon_logs}"
            )
        for device_uuid in cls.npu_uuids:
            if not _ping_daemon(device_uuid):
                daemon_log = _read_logs([DAEMON_STDERR])
                raise RuntimeError(
                    f"Weight cache daemon for NPU {device_uuid} did not answer a "
                    f"ping after ready. Daemon log:\n{daemon_log}"
                )

        # Step 3: client — warm import over zero-copy IPC (weight cache ON).
        cls.client_stdout = open(CLIENT_STDOUT, "w")
        cls.client_stderr = open(CLIENT_STDERR, "w")
        client_url = f"http://127.0.0.1:{_find_free_port()}"
        cls.client_process = popen_launch_server(
            cls.model,
            client_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                str(cls.tp_size),
                "--weight-cache-mode",
                "client",
                *npu_backend_args,
            ],
            return_stdout_stderr=(cls.client_stdout, cls.client_stderr),
        )
        cls.client_logs = _read_logs([CLIENT_STDOUT, CLIENT_STDERR])
        cls.client_time = _parse_metric(cls.client_logs, IPC_TOTAL_RE)

    @classmethod
    def tearDownClass(cls):
        for proc in (
            getattr(cls, "client_process", None),
            getattr(cls, "baseline_process", None),
            getattr(cls, "daemon_process", None),
        ):
            if proc is not None and proc.poll() is None:
                kill_process_tree(proc.pid)
        for stream in (
            getattr(cls, "baseline_stdout", None),
            getattr(cls, "baseline_stderr", None),
            getattr(cls, "daemon_stderr", None),
            getattr(cls, "client_stdout", None),
            getattr(cls, "client_stderr", None),
        ):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for path in (
            BASELINE_STDOUT,
            BASELINE_STDERR,
            DAEMON_STDERR,
            CLIENT_STDOUT,
            CLIENT_STDERR,
        ):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass
        for device_uuid in getattr(cls, "npu_uuids", ()):
            for path in (get_ready_path(device_uuid), get_socket_path(device_uuid)):
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def test_client_loaded_via_ipc(self):
        """Guard that the client truly took the IPC path, not a disk fallback.

        If the client silently fell back to disk loading, ``client_time`` would
        measure another disk load and the speedup assertion in
        ``test_ipc_import_faster_than_disk_load`` would be meaningless.
        """
        self.assertIn(
            IPC_MARKER,
            self.client_logs,
            "Expected the client server to load weights via IPC, but the IPC "
            "load log line was not found — the loader likely fell back to disk.",
        )

    def test_ipc_import_faster_than_disk_load(self):
        """Assert the warm zero-copy IPC import beats a cold disk load."""
        self.assertIsNotNone(
            self.baseline_time,
            "No baseline disk-load timing captured from server logs.",
        )
        self.assertIsNotNone(
            self.client_time,
            "No IPC import timing captured from client logs.",
        )
        speedup = self.baseline_time / self.client_time
        self.assertLess(
            self.client_time,
            self.baseline_time,
            f"Expected IPC import ({self.client_time:.2f}s) to be faster than "
            f"disk load ({self.baseline_time:.2f}s), but it was "
            f"{speedup:.2f}x slower.",
        )


if __name__ == "__main__":
    unittest.main()