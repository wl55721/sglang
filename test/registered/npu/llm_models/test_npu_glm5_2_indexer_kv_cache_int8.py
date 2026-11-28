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

This file launches GLM-5.2-w4a8 - the GLM-5.2 quantized checkpoint family the
PR itself validated (GSM8K accuracy parity + 25% TTFT reduction on A2/A3) -
twice: once without the flag (legacy bf16 indexer path) and once with
`--indexer-kv-cache-dtype int8`, using the single-node A3 recipe
(`--tp-size 16`, one node / 8 physical cards), and verifies:

1. Functionality: the recipe is active and normal generations complete.
2. Effect: the derived KV capacity (max_total_num_tokens) strictly increases
   thanks to the halved indexer cache, and long-context prefill TTFT (Time
   To First Token) does not regress (target: >= 5% faster).

Note: PR #42224 was authored against GLM5.2-w4a8-int; the GLM-5.3-w8a8
checkpoint was measured NOT to work with the recipe on A3 (first-prefill
segfault inside npu_quant_lightning_indexer even in eager mode, and a KV
capacity regression instead of the expected increase), so this test pins the
carrier to the GLM-5.2 family. Speculative decoding (NEXTN) is deliberately
omitted from the launch args so prefill TTFT reflects the plain indexer path.

[Test Category] Model
[Test Target] --indexer-kv-cache-dtype int8
"""

import json
import os
import random
import time
import unittest
from statistics import median

import requests

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_2_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=1800, suite="full-test-16-npu-a3", nightly=True)

MODEL_PATH = GLM_5_2_W4A8_MODEL_PATH
# Single-node A3 recipe (--tp-size 16, one node = 8 physical cards / 16
# dies). Non-FP8 main KV cache is a hard requirement of the INT8 indexer
# recipe; dtype bfloat16 satisfies it.
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
    # The effective prompt ceiling is min(context_len, max_total_num_tokens);
    # the KV pool on a TP16 single node is small for DSA checkpoints, so keep
    # the model side from binding first and let the pool be the real limit.
    "--context-length",
    "32768",
    "--mem-fraction-static",
    "0.8",
    # Run decode eagerly: keeps the measured path free of graph-capture side
    # effects, mirroring the DeepSeek-V3.2 NPU recipe. Prefill TTFT (the
    # measured metric here) is unaffected: prefill graph capture is already
    # auto-disabled on this path.
    "--disable-decode-cuda-graph",
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
# Env vars commonly used by single-node A3 launch scripts.
GLM_SINGLE_NODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1000",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
}
LAUNCH_TIMEOUT = 3000

# Deterministic ~5k-token prompts; long enough that indexer scoring dominates
# prefill (still 2 chunks at --chunked-prefill-size 4096), unique enough that
# no radix-cache reuse hides the effect. 2900 words tokenize to ~5.1k tokens
# on the GLM tokenizer (~1.75 tokens per word), safely under the KV pool
# ceiling that the ingress length check clamps to on a TP16 single node.
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


def build_long_prompt(seed: int, num_words: int = 2900) -> str:
    rng = random.Random(seed)
    body = " ".join(rng.choice(_WORDS) for _ in range(num_words))
    return f"Summarize the following text in one short sentence.\n\n{body}"


def stream_ttft(base_url: str, payload: dict) -> dict:
    """POST /generate with streaming; return TTFT, total latency, finish reason."""
    start = time.perf_counter()
    ttft = None
    finish_reason = None
    finish_reason_raw = None
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
                # Newer sglang returns a structured dict: {"type": "abort",
                # "message": ..., ...}; keep both the normalized type and the
                # raw dict (the abort message pinpoints the scheduler path).
                fr = meta["finish_reason"]
                finish_reason_raw = fr
                finish_reason = fr.get("type") if isinstance(fr, dict) else fr
    return {
        "ttft": ttft,
        "total": time.perf_counter() - start,
        "finish_reason": finish_reason,
        "finish_reason_raw": finish_reason_raw,
    }


def measure_long_prefill_ttft(base_url: str, num_prompts: int = 20, warmup: int = 2):
    """Sequential long-prompt generation; prefill dominates each request.

    Fires `warmup` throwaway requests first (distinct seeds) so the measured
    loop does not pay first-request costs (JIT compile, HCCL rendezvous,
    allocator growth) that differ between the two server launches and would
    otherwise bias whichever config happens to run first.
    """
    for seed in range(10_000, 10_000 + warmup):
        result = stream_ttft(
            base_url,
            {
                "text": build_long_prompt(seed),
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "stream": True,
            },
        )
        assert result["finish_reason"] in ("stop", "length"), result
    ttfts = []
    for seed in range(num_prompts):
        result = stream_ttft(
            base_url,
            {
                "text": build_long_prompt(seed),
                "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                "stream": True,
            },
        )
        assert result["finish_reason"] in ("stop", "length"), result
        assert result["ttft"] is not None, "no token was streamed back"
        ttfts.append(result["ttft"])
    return ttfts


@unittest.skipUnless(
    os.path.isfile(os.path.join(GLM_5_2_W4A8_MODEL_PATH, "config.json")),
    f"GLM-5.2-w4a8 model config not found at {GLM_5_2_W4A8_MODEL_PATH}",
)
class TestNPUGLM52IndexerKVCacheInt8(CustomTestCase):
    """INT8 indexer cache recipe on GLM-5.2-w4a8 (TP16 single node):
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
            env=GLM_SINGLE_NODE_ENVS,
        )
        # Note: /get_server_info is deprecated, use /server_info instead.
        info = requests.get(cls.base_url + "/server_info", timeout=30).json()
        cls.baseline_max_total_num_tokens = info["max_total_num_tokens"]
        # Pool size is the first suspect whenever a long probe FINISH_ABORTs:
        # print it so a failure is diagnosable without re-running.
        print(f"[indexer-int8] baseline max_total_num_tokens = {cls.baseline_max_total_num_tokens}")
        cls.baseline_ttfts = measure_long_prefill_ttft(cls.base_url)
        terminate_and_kill_process_tree(cls.process)
        cls.process = None

        # 2) INT8 indexer cache recipe.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            other_args=INT8_ARGS,
            env=GLM_SINGLE_NODE_ENVS,
        )
        info = requests.get(cls.base_url + "/server_info", timeout=30).json()
        cls.int8_max_total_num_tokens = info["max_total_num_tokens"]
        print(f"[indexer-int8] int8 max_total_num_tokens = {cls.int8_max_total_num_tokens}")
        cls.int8_ttfts = measure_long_prefill_ttft(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_int8_recipe_active(self):
        """The running server must report the requested indexer cache dtype."""
        info = requests.get(self.base_url + "/server_info", timeout=30).json()
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
        speed up) long-context prefill TTFT.

        The baseline and INT8 servers are measured in two separate sequential
        runs, so a raw median-vs-median comparison is exposed to whole-run
        machine drift (clock/thermal state, co-tenants) that is larger than
        the 5% margin being asserted. Instead, prompts use identical seeds in
        both runs, so the comparison is paired: per-seed INT8/baseline TTFT
        ratios are computed and their median is asserted. Run-wide
        multiplicative drift cancels in the per-seed ratios; only per-prompt
        noise, which the median of 20 samples suppresses, remains.
        """
        self.assertEqual(
            len(self.baseline_ttfts), len(self.int8_ttfts), "paired runs diverged"
        )
        ratios = [
            i / b for i, b in zip(self.int8_ttfts, self.baseline_ttfts) if b > 0
        ]
        median_ratio = median(ratios)
        print(
            f"[indexer-int8] per-seed INT8/baseline TTFT ratios = "
            f"{[round(r, 3) for r in ratios]}"
        )
        self.assertLessEqual(
            median_ratio,
            0.95,
            f"INT8 indexer regressed long-context prefill TTFT: paired median "
            f"ratio {median_ratio:.3f} (INT8 {median(self.int8_ttfts):.3f}s vs "
            f"baseline {median(self.baseline_ttfts):.3f}s)",
        )


if __name__ == "__main__":
    unittest.main()
