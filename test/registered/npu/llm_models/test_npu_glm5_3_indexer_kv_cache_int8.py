"""E2E tests for `--indexer-kv-cache-dtype int8` on NPU (NPU = Ascend Neural
Processing Unit).

The INT8 LightningIndexer recipe (PR #42224) quantizes the DSA (Dynamic
Sparse Attention) lightning-indexer key cache to int8 + a per-token fp16
scale, so the CANN (Compute Architecture for Neural Networks, Huawei Ascend)
`npu_quant_lightning_indexer` op runs the indexer scoring in INT8. It is an
NPU-only recipe: the indexer cache bytes per token drop from 256 (bf16 key)
to 130 (int8 key + fp16 scale), which raises the derived KV (Key/Value) cache
capacity, and the INT8 GEMM (General Matrix Multiply) speeds up the indexer
scoring that dominates long-context prefill.

This file launches GLM-5.3-w8a8 (W8A8 = 8-bit weight, 8-bit activation
quantization; GlmMoeDsa family on the generic ascend DSA path) twice - once
without the flag (legacy bf16 indexer path) and once with
`--indexer-kv-cache-dtype int8` - using the official single-node A3 recipe
(`--tp-size 16`, one node / 8 physical cards), and verifies:

1. Functionality: the recipe is active and normal generations complete.
   (Model-level accuracy is gated by the existing GLM-5.3-w8a8 16P nightly
   GPQA/GSM8K tests; the INT8 recipe's precision behavior was validated by
   the PR #42224 unit tests, so no extra accuracy gate is duplicated here.)
2. Effect: the derived KV capacity (max_total_num_tokens) strictly increases
   thanks to the halved indexer cache, and long-context prefill TTFT (Time
   To First Token) does not regress (target: >= 5% faster).

Note: speculative decoding (NEXTN) is deliberately omitted from the launch
args so prefill TTFT reflects the plain indexer path.

[Test Category] Model
[Test Target] --indexer-kv-cache-dtype int8
"""

import json
import random
import time
import unittest
from statistics import median

import requests

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_3_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=1800, suite="full-test-16-npu-a3", nightly=True)

MODEL_PATH = GLM_5_3_W8A8_MODEL_PATH
# Single-node A3 recipe from the official GLM-5.3-w8a8 tutorial
# (--tp-size 16, one node = 8 physical cards / 16 dies). Non-FP8 main KV
# cache is a hard requirement of the INT8 indexer recipe; dtype bfloat16
# satisfies it.
BASE_ARGS = [
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--tp-size",
    "16",
    "--nnodes",
    "1",
    "--chunked-prefill-size",
    "4096",
    "--max-prefill-tokens",
    "8192",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.75",
    "--max-running-requests",
    "32",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--disable-shared-experts-fusion",
]
INT8_ARGS = BASE_ARGS + ["--indexer-kv-cache-dtype", "int8"]
# Env vars from the same tutorial's single-node launch script.
GLM_5_3_SINGLE_NODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1000",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
}
LAUNCH_TIMEOUT = 3000

# Deterministic ~8k-token prompts; long enough that indexer scoring dominates
# prefill, unique enough that no radix-cache reuse hides the effect.
_WORDS = [
    "delta",
    "echo",
    "kilo",
    "lima",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
]


def build_long_prompt(seed: int, num_words: int = 6000) -> str:
    rng = random.Random(seed)
    body = " ".join(rng.choice(_WORDS) for _ in range(num_words))
    return f"Summarize the following text in one short sentence.\n\n{body}"


def stream_ttft(base_url: str, payload: dict) -> dict:
    """POST /generate with streaming; return TTFT, total latency, finish reason."""
    start = time.perf_counter()
    ttft = None
    finish_reason = None
    with requests.post(
        base_url + "/generate", json=payload, stream=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            body = line[len(b"data:") :].strip()
            if body == b"[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("text") and ttft is None:
                ttft = time.perf_counter() - start
            meta = chunk.get("meta_info") or {}
            if meta.get("finish_reason"):
                finish_reason = meta["finish_reason"]
    return {
        "ttft": ttft,
        "total": time.perf_counter() - start,
        "finish_reason": finish_reason,
    }


def measure_long_prefill_ttft(base_url: str, num_prompts: int = 12):
    """Sequential long-prompt generation; prefill dominates each request."""
    ttfts = []
    for seed in range(num_prompts):
        result = stream_ttft(
            base_url,
            {
                "text": build_long_prompt(seed),
                "sampling_params": {"max_new_tokens": 16, "temperature": 0},
            },
        )
        assert result["finish_reason"] in ("stop", "length"), result
        assert result["ttft"] is not None, "no token was streamed back"
        ttfts.append(result["ttft"])
    return ttfts


class TestNPUGLM53IndexerKVCacheInt8(CustomTestCase):
    """INT8 indexer cache recipe on GLM-5.3-w8a8 (TP16 single node):
    functionality + capacity/TTFT effect vs the bf16 indexer baseline.
    [Test Category] Model.
    """

    @classmethod
    def setUpClass(cls):
        # 1) Baseline: legacy bf16 NPU lightning indexer (no flag).
        cls.base_url = "http://127.0.0.1:30000"
        cls.model = MODEL_PATH
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            other_args=BASE_ARGS,
            env=GLM_5_3_SINGLE_NODE_ENVS,
        )
        info = requests.get(cls.base_url + "/get_server_info", timeout=30).json()
        cls.baseline_max_total_num_tokens = info["max_total_num_tokens"]
        cls.baseline_ttfts = measure_long_prefill_ttft(cls.base_url)
        terminate_and_kill_process_tree(cls.process)
        cls.process = None

        # 2) INT8 indexer cache recipe.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            other_args=INT8_ARGS,
            env=GLM_5_3_SINGLE_NODE_ENVS,
        )
        info = requests.get(cls.base_url + "/get_server_info", timeout=30).json()
        cls.int8_max_total_num_tokens = info["max_total_num_tokens"]
        cls.int8_ttfts = measure_long_prefill_ttft(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_int8_recipe_active(self):
        """The running server must report the requested indexer cache dtype."""
        info = requests.get(self.base_url + "/get_server_info", timeout=30).json()
        self.assertEqual(info["indexer_kv_cache_dtype"], "int8")

    def test_generation_functionality(self):
        """Normal short generations must work end to end under the recipe."""
        prompts = [
            "What is the capital of France? Answer in one word.",
            "Name the largest planet in the solar system.",
            "Write the number after 41.",
        ]
        for prompt in prompts:
            resp = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["text"], f"empty generation for: {prompt}")

    def test_indexer_cache_capacity_increases(self):
        """Effect (deterministic): halving the indexer cache bytes per token
        must raise the derived KV token capacity."""
        self.assertGreater(
            self.int8_max_total_num_tokens,
            self.baseline_max_total_num_tokens,
            f"INT8 indexer cache did not increase KV capacity: "
            f"{self.int8_max_total_num_tokens} vs baseline "
            f"{self.baseline_max_total_num_tokens}",
        )

    def test_long_prefill_ttft_no_regression(self):
        """Effect: INT8 indexer scoring must not regress (and is expected to
        speed up) long-context prefill TTFT."""
        baseline_median = median(self.baseline_ttfts)
        int8_median = median(self.int8_ttfts)
        self.assertLessEqual(
            int8_median,
            baseline_median * 0.95,
            f"INT8 indexer regressed long-context prefill TTFT: "
            f"median {int8_median:.3f}s vs baseline {baseline_median:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()
