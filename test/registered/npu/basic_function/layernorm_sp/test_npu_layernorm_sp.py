"""E2E accuracy test for --enable-layernorm-sp (Megatron LayerNorm sequence
parallelism) on NPU/Ascend.

SP re-associates the row-parallel all-reduce as reduce-scatter + all-gather and
runs the norm/residual regions on sequence shards, so a correct implementation
must match the non-SP result within floating-point reordering noise. On NPU the
fast-path uses the CANN MC2 fused ops when available (npu_mm_reduce_scatter_base
/ npu_all_gather_base_mm), else the un-fused reduce-scatter/all-gather fallback.
Requires tp>1 (here tp=2) for SP to engage and a Qwen3 dense model (the SP
allowlist entry, architecture "Qwen3ForCausalLM").
"""

import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2800, suite="full-acc-2-npu-a3", nightly=True)

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
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--disable-cuda-graph",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.8,
    "--dtype",
    "bfloat16",
    "--enable-layernorm-sp",
]


class TestNPULayerNormSP(TestNpuAccuracyTestCaseBase):
    model = QWEN3_8B_WEIGHTS_PATH
    envs = LAYERNORM_SP_ENVS
    other_args = LAYERNORM_SP_OTHER_ARGS
    accuracy = 0.85
    datasets = ["gsm8k"]
    few_shot_num = 5
    generation_config = {
        "max_tokens": 512,
        "temperature": 0,
    }
    eval_batch_size = 64
    limit = 200

    def test_gsm8k(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()