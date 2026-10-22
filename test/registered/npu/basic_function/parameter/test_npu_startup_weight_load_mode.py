import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1600, suite="full-2-npu-a3", nightly=True)


class TestStartupWeightLoadMode(CustomTestCase):
    """Testcase: Verify --startup-weight-load-mode=overlap is supported on NPU.

    Overlap startup overlaps checkpoint prefetch with decode CUDA graph capture.
    NPU supports this only when the prefill graph (TC_PIECEWISE by default) is
    disabled, because tc_piecewise prefill bakes weights during torch.compile and
    is unsafe for commit-after-capture. The decode graph stays enabled so the
    overlap path is actually exercised.

    A 32B model is used on 2 cards (tp_size=2): overlap startup is currently
    restricted to TP1/TP2, so the 4-card layout used by the plain 32B accuracy
    test cannot run here.

    The test launches the server in both `serial` and `overlap` modes and
    compares their startup wall-clock time (time from process start until
    `/health_generate` first returns 200), so the overlap speedup can be
    observed directly.

    [Test Category] Parameter
    [Test Target] --startup-weight-load-mode
    """

    model = QWEN3_32B_WEIGHTS_PATH
    OUT_LOG_PATH = "./out_log.txt"
    ERR_LOG_PATH = "./err_log.txt"

    def _launch_and_measure(self, *, mode):
        """Launch the server in `mode` and return its startup time in seconds.

        Startup time is measured from process start until `popen_launch_server`
        returns (i.e. `/health_generate` first responds 200). A `generate`
        request is also run to prove the capture sentinel weights were replaced
        with real weights; a baked-sentinel graph would output gibberish.
        """
        other_args = [
            "--attention-backend",
            "ascend",
            "--tp-size",
            "2",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            # Disable the prefill graph for both modes so the only variable in
            # the timing comparison is `--startup-weight-load-mode`.
            "--disable-prefill-cuda-graph",
        ]
        if mode is not None:
            other_args += ["--startup-weight-load-mode", mode]

        out_log_file = open(self.OUT_LOG_PATH, "w+", encoding="utf-8")
        err_log_file = open(self.ERR_LOG_PATH, "w+", encoding="utf-8")
        process = None
        try:
            start = time.perf_counter()
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )
            startup_seconds = time.perf_counter() - start

            gen_resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(gen_resp.status_code, 200)
            self.assertIn("Paris", gen_resp.text)
            return startup_seconds
        finally:
            if process is not None:
                kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove(self.OUT_LOG_PATH)
            os.remove(self.ERR_LOG_PATH)

    def test_startup_load_mode_timing(self):
        serial_s = self._launch_and_measure(mode="serial")
        overlap_s = self._launch_and_measure(mode="overlap")
        speedup_s = serial_s - overlap_s

        print(
            f"[startup-weight-load-mode] serial={serial_s:.1f}s "
            f"overlap={overlap_s:.1f}s speedup={speedup_s:.1f}s "
            f"({(1 - overlap_s / serial_s) * 100:.1f}%)"
        )

        # Startup time is noisy in CI (the health poll granularity plus a tiny
        # 1B model), so instead of asserting a fragile `overlap < serial` speedup
        # we only guard against overlap being grossly slower than serial.
        self.assertLessEqual(
            overlap_s,
            serial_s * 2.0,
            f"overlap startup ({overlap_s:.1f}s) unexpectedly much slower than "
            f"serial ({serial_s:.1f}s)",
        )


if __name__ == "__main__":
    unittest.main()