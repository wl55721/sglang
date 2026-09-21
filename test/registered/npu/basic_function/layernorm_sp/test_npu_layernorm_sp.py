"""E2E accuracy test for --enable-layernorm-sp (Megatron LayerNorm sequence
parallelism) on NPU/Ascend.

SP re-associates the row-parallel all-reduce as reduce-scatter + all-gather and
runs the norm/residual regions on sequence shards, so a correct implementation
must match the non-SP result within floating-point reordering noise. On NPU the
fast-path uses the CANN MC2 fused ops when available (npu_mm_reduce_scatter_base
/ npu_all_gather_base_mm), else the un-fused reduce-scatter/all-gather fallback.
Requires tp>1 (here tp=2) for SP to engage and a Qwen3 dense model (the SP
allowlist entry, architecture "Qwen3ForCausalLM").

NOTE: this uses the self-contained few-shot GSM8K runner (no CI-only evalscope
harness), so it runs on any NPU box rather than only the CI host rooted at
/root/sglang with a prepared test_env_evalscope.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)

# Qwen3-8B tp=2 bf16 GSM8K baseline; --enable-layernorm-sp must match it.
LAYERNORM_SP_ACCURACY = 0.85

# Marker printed by sglang.srt.layers.layernorm_sp.initialize_layernorm_sp when
# SP is truly engaged. Accuracy alone cannot prove the feature ran (a no-op
# passes too), so the test asserts this line appears in the server startup log.
LAYERNORM_SP_ENABLED_LINE = "LayerNorm sequence parallelism (SP) ENABLED"

LAYERNORM_SP_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

LAYERNORM_SP_OTHER_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    0.8,
    "--attention-backend",
    "ascend",
    "--tp-size",
    2,
    "--device",
    "npu",
    "--disable-cuda-graph",
    "--dtype",
    "bfloat16",
    "--enable-layernorm-sp",
]


class TestNPULayerNormSP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)

    def test_a_gsm8k(self):
        with tempfile.NamedTemporaryFile(
            mode="w+", buffering=1, encoding="utf-8", delete=False
        ) as stdout_file, tempfile.NamedTemporaryFile(
            mode="w+", buffering=1, encoding="utf-8", delete=False
        ) as stderr_file:
            process = popen_launch_server(
                self.model,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=list(LAYERNORM_SP_OTHER_ARGS),
                env={**os.environ, **LAYERNORM_SP_ENVS},
                return_stdout_stderr=(stdout_file, stderr_file),
            )
            try:
                args = SimpleNamespace(
                    num_shots=5,
                    data_path=None,
                    num_questions=200,
                    max_new_tokens=512,
                    parallel=64,
                    host=f"http://{self.url.hostname}",
                    port=int(self.url.port),
                )
                metrics = run_eval_few_shot_gsm8k(args)
                self.assertGreaterEqual(
                    metrics["accuracy"],
                    LAYERNORM_SP_ACCURACY,
                )

                # Prove --enable-layernorm-sp actually engaged (not a no-op that
                # happens to pass accuracy): the server must have logged the SP
                # marker during startup.
                stdout_file.seek(0)
                stderr_file.seek(0)
                server_log = stdout_file.read() + stderr_file.read()
                self.assertIn(
                    LAYERNORM_SP_ENABLED_LINE,
                    server_log,
                    "server log missing SP-enabled marker; "
                    "--enable-layernorm-sp likely did not take effect",
                )
            finally:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()