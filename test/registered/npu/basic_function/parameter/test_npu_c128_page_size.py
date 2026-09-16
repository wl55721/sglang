import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1800, suite="base-b-test-16-npu-a3")
register_npu_ci(est_time=1800, suite="nightly-16-npu-a3", nightly=True)

# DeepSeek-V4-Flash W8A8 requires the same NPU runtime environment as the
# existing accuracy/performance DSV4 cases. Reused verbatim to keep the smoke
# launch aligned with the known-good 16-NPU deployment.
DEEPSEEK_V4_FLASH_W8A8_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "35",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
}

# The parameter under test: a valid C128 physical page size (positive multiple
# of 16). It must differ from the default 16 to prove the value actually takes
# effect on the DSV4 C128 KV pool.
C128_PAGE_SIZE = 32

DEEPSEEK_V4_FLASH_W8A8_OTHER_ARGS = [
    "--page-size",
    "128",
    "--tp-size",
    "16",
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--mem-fraction-static",
    "0.68",
    "--max-running-requests",
    "16",
    "--dp-size",
    "16",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--c128-page-size",
    str(C128_PAGE_SIZE),
    "--max-model-len",
    "8192",
    "--disable-radix-cache",
    "--skip-server-warmup",
]


class TestNpuC128PageSize(CustomTestCase):
    """Testcase: Verify the NPU DSV4 C128 KV cache physical page size parameter
    (--c128-page-size) takes effect. A valid value (positive multiple of 16)
    lets the DeepSeek-V4-Flash W8A8 server start and serve an inference request.

    [Test Category] Parameter
    [Test Target] --c128-page-size
    """

    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        env = os.environ.copy()
        env.update(DEEPSEEK_V4_FLASH_W8A8_ENVS)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=DEEPSEEK_V4_FLASH_W8A8_OTHER_ARGS,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_c128_page_size_server_and_inference(self):
        response = requests.get(f"{self.base_url}/health_generate")
        self.assertEqual(response.status_code, 200)

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()