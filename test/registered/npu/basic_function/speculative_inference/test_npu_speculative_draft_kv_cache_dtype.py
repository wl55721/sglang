import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestNpuSpeculativeDraftKVCacheDtype(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that EAGLE3 speculative decoding with a FP8 draft KV cache
    keeps GSM8K accuracy on NPU.

    This exercises the draft-worker-only parameter --speculative-draft-kv-cache-dtype,
    which sets the KV cache dtype of the draft model independently from the target
    --kv-cache-dtype. With fp8_e4m3 the draft KV pool is halved.

    [Test Category] Speculative Decoding
    [Test Target] --speculative-draft-kv-cache-dtype
    """

    model = QWEN3_8B_WEIGHTS_PATH
    timeout_for_server_launch = 1500
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--disable-radix-cache",
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        QWEN3_8B_EAGLE3_WEIGHTS_PATH,
        "--speculative-num-steps",
        "4",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "5",
        "--speculative-attention-mode",
        "decode",
        "--speculative-draft-kv-cache-dtype",
        "fp8_e4m3",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.7",
        "--disable-cuda-graph",
        "--dtype",
        "bfloat16",
    ]

    env = {
        **os.environ,
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    }

    accuracy = 0.81
    num_questions = 1319


if __name__ == "__main__":
    unittest.main()