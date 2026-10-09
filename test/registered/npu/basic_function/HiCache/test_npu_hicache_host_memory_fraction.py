import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestNpuHiCacheHostMemoryFraction(CustomTestCase):
    """Explicitly exercise --hicache-host-memory-fraction on Ascend NPU.

    The parameter bounds the HiCache L2 host (CPU RAM) pool budget when neither
    --hicache-ratio nor --hicache-size is set. Sizing only depends on psutil and
    Linux cgroup, so this Knob is device-agnostic; these tests prove the value
    propagates into the auto-sizing path and that auto-sizing turns off when the
    host pool size is given explicitly.

    [Test Category] HiCache
    [Test Target] --hicache-host-memory-fraction
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    def _launch(self, extra_args):
        out_log = open("./hf_out.log", "w+", encoding="utf-8")
        err_log = open("./hf_err.log", "w+", encoding="utf-8")
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=extra_args,
            return_stdout_stderr=(out_log, err_log),
        )
        return process, out_log, err_log

    def _read_logs(self, out_log, err_log):
        out_log.seek(0)
        err_log.seek(0)
        return out_log.read() + err_log.read()

    def _cleanup(self, process, out_log, err_log):
        kill_process_tree(process.pid)
        out_log.close()
        err_log.close()
        os.remove("./hf_out.log")
        os.remove("./hf_err.log")

    def _common_args(self):
        return [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--tp-size",
            "1",
            "--enable-hierarchical-cache",
        ]

    def test_host_memory_fraction_auto_sizes(self):
        # No --hicache-ratio/--hicache-size: the fraction bounds the host pool.
        args = self._common_args() + ["--hicache-host-memory-fraction", "0.3"]
        process, out_log, err_log = self._launch(args)
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
            )
            self.assertEqual(response.status_code, 200)
            content = self._read_logs(out_log, err_log)
            self.assertIn("HiCache auto-sizing", content)
            self.assertIn("fraction 0.30", content)
        finally:
            self._cleanup(process, out_log, err_log)

    def test_fraction_inactive_with_explicit_size(self):
        # An explicit --hicache-size resolves the fraction to None (auto-sizing off).
        args = self._common_args() + [
            "--hicache-host-memory-fraction",
            "0.3",
            "--hicache-size",
            "2",
        ]
        process, out_log, err_log = self._launch(args)
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
            )
            self.assertEqual(response.status_code, 200)
            content = self._read_logs(out_log, err_log)
            self.assertNotIn("HiCache auto-sizing", content)
        finally:
            self._cleanup(process, out_log, err_log)


if __name__ == "__main__":
    unittest.main()
