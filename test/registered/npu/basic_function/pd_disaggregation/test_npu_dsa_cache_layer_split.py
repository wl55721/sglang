import os
import unittest
from time import sleep
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ascend.npu_eval_accuracy_kit import _is_pr_pipeline, run_npu_pr_smoke
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_pd_server,
)

register_npu_ci(est_time=3600, suite="full-16-npu-a3")
register_npu_ci(est_time=3600, suite="nightly-16-npu-a3", nightly=True)

# --- Device partitioning for PD disaggregation on a 16-NPU host ---
# prefill uses cards 0..TP_SIZE-1 (default base-gpu-id 0); decode starts at
# DECODE_BASE_GPU_ID so it owns cards TP_SIZE..2*TP_SIZE-1.
TP_SIZE = 8
NPU_COUNT = 16
DECODE_BASE_GPU_ID = TP_SIZE

# Environment for DeepSeek-V4-Flash W8A8 (reused from the verified perf case) plus the
# Ascend mooncake transfer backend required by --enable-dsa-cache-layer-split.
BASE_ENV = {
    # Restrict the visible NPU set to the 16 cards of the host for both workers.
    "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(i) for i in range(NPU_COUNT)),
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "8",
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    # Ascend mooncake transfer backend for the DSA cache layer split.
    "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
    # ASCEND_NPU_PHY_ID must be tuned per CI node/vendor; adjust on the real host.
    "ASCEND_NPU_PHY_ID": os.environ.get("ASCEND_NPU_PHY_ID", ""),
}

COMMON_ARGS = [
    "--tp-size",
    str(TP_SIZE),
    "--attn-cp-size",
    str(TP_SIZE),
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    "9000",
    "--mem-fraction-static",
    "0.7",
    "--chunked-prefill-size",
    "131072",
    "--max-running-requests",
    "160",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--kv-cache-dtype",
    "auto",
    "--disable-radix-cache",
]


class TestNPUDsaCacheLayerSplit(TestDisaggregationBase):
    """Testcase: Verify that --enable-dsa-cache-layer-split keeps GSM8K accuracy on NPU.

    The DSA GPU KV/indexer cache layers are split across context-parallel ranks in a PD
    disaggregation prefill worker. This parameter applies only to the prefill stage and
    requires: DSA model, PD prefill, --enable-prefill-cp + --cp-strategy interleave, the
    mooncake transfer backend and --pp-size 1 (default).

    [Test Category] Context Parallel + PD Disaggregation
    [Test Target] --enable-dsa-cache-layer-split
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()
        cls.url = urlparse(cls.lb_url)

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-dsa-cache-layer-split",
            "--disaggregation-transfer-backend",
            "mooncake",
            *COMMON_ARGS,
        ]

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env={**os.environ, **BASE_ENV},
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "mooncake",
            # Start decode on card TP_SIZE so it does not collide with prefill's 0..TP_SIZE-1.
            "--base-gpu-id",
            str(DECODE_BASE_GPU_ID),
            *COMMON_ARGS,
        ]

        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env={**os.environ, **BASE_ENV},
        )

    def test_gsm8k(self):
        if _is_pr_pipeline:
            run_npu_pr_smoke(self.lb_url)
            return
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.url.hostname}",
            port=int(self.url.port),
        )

        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            # 0.95 nominal, allow 0.02 fluctuation
            0.93,
        )

    @classmethod
    def tearDownClass(cls):
        cls.model = None
        super().tearDownClass()
        # wait for server release source
        sleep(10)


if __name__ == "__main__":
    if is_in_ci():
        unittest.main()
    else:
        unittest.main()