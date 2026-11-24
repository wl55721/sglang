"""E2E tests for the UniBoost schedule policy on NPU (NPU = Ascend Neural
Processing Unit).

UniBoost (PR #35637) is a tail-aware prefill admission policy: every queued
request gets priority key Phi = arrival_time - b(uncached_work), so cheap
(prefix-cache-hot) requests are admitted ahead of expensive cold prompts with
a bounded boost. The parameters under test are scheduler-side only (device
agnostic); this file verifies on NPU that:

1. Functionality: `--schedule-policy uniboost` (+ `--uniboost-gamma`,
   `--enable-uniboost-gamma-ada`) serves every request correctly and the
   running config is visible through /server_info.
2. Effect: in a mixed burst (one long cold prompt + several cache-hot short
   requests), the TTFT (Time To First Token, first-token latency) of the
   cache-hot requests is materially lower than under the FCFS (First Come
   First Served) baseline.

[Test Category] Interface
[Test Target] --schedule-policy uniboost / --uniboost-gamma / --enable-uniboost-gamma-ada
"""

import json
import random
import re
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from statistics import median

import requests

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=1200, suite="full-test-2-npu-a3", nightly=True)

BASE_ARGS = [
    "--attention-backend",
    "ascend",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    # Small chunks make queueing dominate: a long cold prompt occupies several
    # prefill iterations, so the scheduling difference is measurable.
    "--chunked-prefill-size",
    "2048",
    # A real admission bottleneck is REQUIRED for the policy to matter: the
    # schedule policy only reorders requests waiting for admission. With a
    # cap >= the burst size every request is admitted at once and both
    # policies behave identically (chunked prefill interleaves them the same
    # way). Cap 2 = cold + 1 hot run concurrently; the rest must queue, so
    # FCFS keeps the queued hots behind the whole cold prompt while uniboost
    # admits the cache-hot requests first.
    "--max-running-requests",
    "2",
    "--mem-fraction-static",
    "0.8",
]
FCFS_ARGS = list(BASE_ARGS)
UNIBOOST_ARGS = BASE_ARGS + [
    "--schedule-policy",
    "uniboost",
    "--uniboost-gamma",
    "10",
]
GAMMA_ADA_ARGS = BASE_ARGS + [
    "--schedule-policy",
    "uniboost",
    "--uniboost-gamma",
    "10",
    "--enable-uniboost-gamma-ada",
]

# ~800-token shared passage; every "hot" request reuses it so the radix cache
# makes its uncached work tiny (a few dozen tokens).
SHARED_PASSAGE = (
    "The industrial revolution reshaped Europe in the nineteenth century. "
    "Steam power replaced water wheels, railways connected port cities with "
    "inland factories, and telegraph lines carried market prices faster than "
    "any courier. Workers migrated from villages into growing urban centers, "
    "where new social classes emerged alongside mechanized production. "
) * 8

# A long prompt with no reusable prefix, so its uncached work dominates the
# burst (~12k tokens against the 2048-token chunk size).
_COLD_WORDS = [
    "harbor",
    "meadow",
    "quartz",
    "lantern",
    "orbit",
    "canvas",
    "thicket",
    "beacon",
    "granite",
    "willow",
    "compass",
    "cobalt",
    "tundra",
    "sparrow",
    "marble",
    "cinder",
]


def build_cold_prompt(num_words: int = 9000) -> str:
    rng = random.Random(20260101)
    return " ".join(rng.choice(_COLD_WORDS) for _ in range(num_words))


def stream_generate_ttft(base_url: str, payload: dict) -> dict:
    """POST /generate with streaming; return TTFT, total latency and finish reason."""
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
                # Newer sglang returns a structured dict: {"type": ..., ...}
                fr = meta["finish_reason"]
                finish_reason = fr.get("type") if isinstance(fr, dict) else fr
    total = time.perf_counter() - start
    return {"ttft": ttft, "total": total, "finish_reason": finish_reason}


def run_mixed_burst(base_url: str, cold_prompt: str, num_hot: int = 8, rounds: int = 3):
    """One round = a long cold request submitted first, then `num_hot`
    cache-hot short requests submitted right after (all streaming). Under
    FCFS the hot requests queue behind the cold prompt; under uniboost they
    are admitted ahead of it (bounded boost). Returns (hot_ttfts, cold_ttfts)."""
    hot_ttfts, cold_ttfts = [], []
    for round_idx in range(rounds):
        futures = []

        def _cold():
            payload = {
                "text": cold_prompt,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                "stream": True,
            }
            return stream_generate_ttft(base_url, payload)

        def _hot(i: int):
            payload = {
                "text": SHARED_PASSAGE
                + f"\n\nQuestion {round_idx}-{i}: What invention carried market prices faster than any courier? Answer in one word.",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                "stream": True,
            }
            return stream_generate_ttft(base_url, payload)

        with ThreadPoolExecutor(max_workers=num_hot + 1) as pool:
            futures.append(pool.submit(_cold))
            time.sleep(0.05)  # cold request reaches the queue first
            for i in range(num_hot):
                futures.append(pool.submit(_hot, i))
            results = [f.result() for f in futures]

        for r in results:
            assert r["ttft"] is not None, "no token was streamed back"
            assert r["finish_reason"] in ("stop", "length"), (
                f"unexpected finish_reason: {r['finish_reason']}"
            )
        cold_ttfts.append(results[0]["ttft"])
        hot_ttfts.extend(r["ttft"] for r in results[1:])
    return hot_ttfts, cold_ttfts


class TestNPUUniboostSchedulePolicy(CustomTestCase):
    """FCFS baseline vs UniBoost: functionality + cache-hot TTFT improvement.

    Launches the same 1B model twice (FCFS first, then uniboost) and runs the
    identical mixed burst against both. [Test Category] Interface.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.cold_prompt = build_cold_prompt()

        # 1) FCFS baseline
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=FCFS_ARGS,
        )
        cls.fcfs_hot_ttfts, cls.fcfs_cold_ttfts = run_mixed_burst(
            cls.base_url, cls.cold_prompt
        )
        terminate_and_kill_process_tree(cls.process)
        cls.process = None

        # 2) UniBoost policy
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=UNIBOOST_ARGS,
        )
        cls.uniboost_hot_ttfts, cls.uniboost_cold_ttfts = run_mixed_burst(
            cls.base_url, cls.cold_prompt
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_uniboost_config_active(self):
        """The running server must report the uniboost policy and gamma."""
        info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertEqual(info["schedule_policy"], "uniboost")
        self.assertEqual(float(info["uniboost_gamma"]), 10.0)

    def test_all_requests_complete_correctly(self):
        """Every burst request finished with a normal reason and non-empty
        output (results were collected in setUpClass; the assert here makes
        the contract explicit)."""
        self.assertTrue(self.uniboost_hot_ttfts, "no hot request results")
        self.assertTrue(self.uniboost_cold_ttfts, "no cold request results")

    def test_hot_request_ttft_improves(self):
        """Effect: median TTFT of cache-hot requests must be at least 20%
        lower under uniboost than under FCFS."""
        fcfs_median = median(self.fcfs_hot_ttfts)
        uniboost_median = median(self.uniboost_hot_ttfts)
        self.assertLess(
            uniboost_median,
            fcfs_median * 0.8,
            f"UniBoost did not improve cache-hot TTFT: "
            f"median {uniboost_median:.3f}s vs FCFS {fcfs_median:.3f}s",
        )


class TestNPUUniboostGammaAda(CustomTestCase):
    """Functionality + effect of --enable-uniboost-gamma-ada (Gamma Ada =
    online automatic tuning of gamma from the observed latency tail).

    Fires >= 256 completed requests so at least one gamma refit window
    (256 samples) elapses, then asserts the server is healthy, the config is
    reported, at least one refit actually fired (its log line was emitted)
    and gamma moved away from its initial value. [Test Category] Interface.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        # Capture server output so the "UniBoost gamma-Ada refit #N: gamma=..."
        # log lines (the only external view of the live gamma, which lives in
        # scheduler-process memory) can be asserted afterwards.
        cls.out_log_file = open("./uniboost_gamma_ada_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./uniboost_gamma_ada_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=GAMMA_ADA_ARGS,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        cls.out_log_file.close()
        cls.err_log_file.close()

    def test_uniboost_config_active(self):
        # Note: /get_server_info is deprecated, use /server_info instead.
        info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertEqual(info["schedule_policy"], "uniboost")
        self.assertEqual(info["enable_uniboost_gamma_ada"], True)

    def test_sustained_load_across_refit_windows(self):
        """>= 256 completions cross at least one gamma-Ada refit window; all
        requests must succeed, the server must stay responsive afterwards,
        and at least one refit must have fired with a gamma that actually
        moved away from the initial value (10.0)."""
        waves = 10
        concurrency = 32
        for wave in range(waves):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(
                        requests.post,
                        self.base_url + "/generate",
                        timeout=120,
                        json={
                            "text": f"Count from {wave} to {wave + 3}.",
                            "sampling_params": {
                                "max_new_tokens": 8,
                                "temperature": 0,
                            },
                        },
                    )
                    for _ in range(concurrency)
                ]
                for future in futures:
                    resp = future.result()
                    self.assertEqual(resp.status_code, 200)
                    self.assertTrue(resp.json()["text"], "empty generation")
        # Server still healthy after refit(s).
        health = requests.get(self.base_url + "/health", timeout=30)
        self.assertEqual(health.status_code, 200)

        # Effect: the refit must actually have run and moved gamma. The live
        # gamma is scheduler-internal, so assert on the refit log lines.
        self.out_log_file.seek(0)
        self.err_log_file.seek(0)
        content = self.out_log_file.read() + self.err_log_file.read()
        refits = re.findall(
            r"UniBoost gamma-Ada refit #(\d+): gamma=([0-9.eE+-]+)", content
        )
        self.assertGreaterEqual(
            len(refits),
            1,
            "no 'UniBoost gamma-Ada refit' line was logged; the online "
            "tuner did not fire despite >= 256 completed requests",
        )
        moved_gamma = float(refits[-1][1])
        self.assertNotAlmostEqual(
            moved_gamma,
            10.0,
            delta=0.5,
            msg=f"gamma-Ada fired {len(refits)} time(s) but gamma never left "
            f"its initial value (last gamma={moved_gamma})",
        )


if __name__ == "__main__":
    unittest.main()
