import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestStartupWeightLoadMode(CustomTestCase):
    """Testcase: Verify --startup-weight-load-mode=overlap is supported on NPU.

    Overlap startup overlaps checkpoint prefetch with decode CUDA graph capture.
    NPU supports this only when the prefill graph (TC_PIECEWISE by default) is
    disabled, because tc_piecewise prefill bakes weights during torch.compile and
    is unsafe for commit-after-capture. The decode graph stays enabled so the
    overlap path is actually exercised.

    [Test Category] Parameter
    [Test Target] --startup-weight-load-mode
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    OUT_LOG_PATH = "./out_log.txt"
    ERR_LOG_PATH = "./err_log.txt"

    def test_overlap_mode(self):
        other_args = [
            "--startup-weight-load-mode",
            "overlap",
            "--attention-backend",
            "ascend",
            "--disable-prefill-cuda-graph",
        ]

        out_log_file = open(self.OUT_LOG_PATH, "w+", encoding="utf-8")
        err_log_file = open(self.ERR_LOG_PATH, "w+", encoding="utf-8")
        process = None
        try:
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )

            health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(health_resp.status_code, 200)

            gen_resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(gen_resp.status_code, 200)
            # Correct output proves the sentinel weights were replaced with real
            # weights after capture; a baked-sentinel graph would output gibberish.
            self.assertIn("Paris", gen_resp.text)
        finally:
            if process is not None:
                kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove(self.OUT_LOG_PATH)
            os.remove(self.ERR_LOG_PATH)


if __name__ == "__main__":
    unittest.main()