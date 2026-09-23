import os
import subprocess
import sys
import time
import unittest

import requests

from sglang.srt.platforms import current_platform
from sglang.srt.utils import kill_process_tree
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Lightweight 1B model keeps the daemon->client IPC handoff cheap to exercise on
# every NPU run while still covering the real block-load path. The test asserts
# the IPC path ran, not any particular model's quality.
DEFAULT_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

# Capture the client server's logs so test_loaded_via_ipc can assert the IPC
# load path actually ran (and did not silently fall back to disk).
STDOUT_FILENAME = "/tmp/test_npu_weight_cache_stdout.log"
STDERR_FILENAME = "/tmp/test_npu_weight_cache_stderr.log"
# Capture the daemon's own logs too so test_daemon_served_via_npu_ipc can assert
# the daemon genuinely served a fetch_state request over the NPU transport
# (proving a real client<->daemon handshake, not the client merely loading over
# IPC in isolation).
DAEMON_STDERR_FILENAME = "/tmp/test_npu_weight_cache_daemon_stderr.log"

# Daemon-side log marker emitted only when it serves a fetch_state request via
# the NPU transport backend. Absence means the daemon never handed out handles
# to this client, so the "IPC load" on the client side would be suspect.
DAEMON_SERVED_MARKER = "via npu_ipc transport"

PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "The future of AI is",
]


def _npu_uuids(tp_size: int) -> list:
    # Single-node, default base_gpu_id/gpu_id_step: rank i runs on physical NPU i.
    return [current_platform.get_device_uuid(i) for i in range(tp_size)]


class TestNpuWeightCacheDaemon(CustomTestCase):
    """E2E test: start a weight cache daemon, then launch a server in client
    mode (TP=1) and confirm it loads weights via zero-copy IPC and generates.

    This exercises the NPU `NpuIpcTransportBackend` path: the daemon loads the
    model to Ascend HBM and exports native torch_npu reduction handles; the
    client server maps them over IPC instead of reading the disk again.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tp_size = 1
        cls.npu_uuids = _npu_uuids(cls.tp_size)

        # Clean up stale ready/socket files from previous runs
        for device_uuid in cls.npu_uuids:
            for path in (get_ready_path(device_uuid), get_socket_path(device_uuid)):
                if os.path.exists(path):
                    os.unlink(path)

        # Step 1: Launch the weight cache daemon (blocks until one rank is ready,
        # then serves the exported handles over the weight-cache socket). Capture
        # its stderr so test_daemon_served_via_npu_ipc can assert the daemon
        # actually served handles to the client.
        cls.daemon_stderr = open(DAEMON_STDERR_FILENAME, "w")
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
            stderr=cls.daemon_stderr,
        )

        # Step 2: Wait for the daemon ready file
        timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        start = time.time()
        for device_uuid in cls.npu_uuids:
            ready_path = get_ready_path(device_uuid)
            while not os.path.exists(ready_path):
                if time.time() - start > timeout:
                    kill_process_tree(cls.daemon_process.pid)
                    raise TimeoutError(
                        f"Weight cache daemon for NPU {device_uuid} not ready "
                        f"within {timeout}s"
                    )
                if cls.daemon_process.poll() is not None:
                    raise RuntimeError(
                        f"Weight cache daemon exited prematurely "
                        f"with code {cls.daemon_process.returncode}"
                    )
                time.sleep(2)

        # Step 3: Launch server in client mode — loads weights via IPC from daemon
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                str(cls.tp_size),
                "--weight-cache-mode",
                "client",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "daemon_process") and cls.daemon_process:
            kill_process_tree(cls.daemon_process.pid)
        for stream in (
            getattr(cls, "stdout", None),
            getattr(cls, "stderr", None),
            getattr(cls, "daemon_stderr", None),
        ):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for path in (
            STDOUT_FILENAME,
            STDERR_FILENAME,
            DAEMON_STDERR_FILENAME,
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

    def test_generate(self):
        for prompt in PROMPTS:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 32,
                    "temperature": 0,
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            text = data["choices"][0]["text"]
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0, f"Empty output for prompt: {prompt}")

    def test_loaded_via_ipc(self):
        """Assert the server actually loaded weights over IPC.

        Without this, the test would still pass if the IPC path silently
        regressed to disk loading (the daemon would just sit unused), because
        generation output looks identical either way. The daemon-side loader
        logs "[IpcModelLoader] Loaded model via IPC", so its presence in the
        captured server logs is our proof the IPC path ran.
        """
        for stream in (getattr(self, "stdout", None), getattr(self, "stderr", None)):
            if stream is not None:
                try:
                    stream.flush()
                except OSError:
                    pass
        logs = ""
        for path in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(path):
                with open(path, errors="replace") as f:
                    logs += f.read()
        self.assertIn(
            "Loaded model via IPC",
            logs,
            "Expected the client server to load weights via IPC, but the IPC "
            "load log line was not found — the loader likely fell back to disk.",
        )

    def test_daemon_served_via_npu_ipc(self):
        """Assert the daemon itself served a fetch_state over the NPU transport.

        ``test_loaded_via_ipc`` proves the *client* took the IPC load path, but
        alone it can't rule out the client loading over IPC while a stale/absent
        daemon never participated. This closes the loop on the other end: the
        daemon only emits the marker when it serves tensors via
        ``NpuIpcTransportBackend`` (name ``npu_ipc``), so its presence proves a
        real client<->daemon handshake over the NPU transport happened.
        """
        if getattr(self, "daemon_stderr", None) is not None:
            try:
                self.daemon_stderr.flush()
            except OSError:
                pass
        logs = ""
        if os.path.exists(DAEMON_STDERR_FILENAME):
            with open(DAEMON_STDERR_FILENAME, errors="replace") as f:
                logs += f.read()
        self.assertIn(
            DAEMON_SERVED_MARKER,
            logs,
            "Expected the weight cache daemon to serve fetch_state via the "
            f"NPU transport (marker {DAEMON_SERVED_MARKER!r}), but the daemon "
            "log did not contain it — the daemon never handed handles to the "
            "client, so the IPC path is not actually end-to-end.",
        )


if __name__ == "__main__":
    unittest.main()