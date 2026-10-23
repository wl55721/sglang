"""E2E test for ``--enable-kv-cache-sharding`` on NPU (Ascend backend).

KV-cache sharding strips the at-rest KV pool at page granularity across the
attention group that would otherwise replicate it. On NPU it runs only on PD
prefill workers:

* GQA/MHA models shard across attention-CP ranks (needs ``--enable-prefill-cp``
  with ``attn_cp_size > 1``);
* MLA models without prefill CP shard across attention-TP ranks (needs
  ``tp_size > 1`` and attention DP disabled).

Each scenario launches a full PD disaggregation (prefill + decode + LB) using
the Ascend transfer backend and exercises the sharded read path against a real
NPU attention kernel. Two checks per scenario:

* a plain generation goes through the owner-major scratch (``begin_shard_extend``
  -> HCCL gather -> page-major reshape -> NPU attention kernel);
* a repeated prompt hits the prefill prefix cache, proving the
  ``flatten_prefix_block_tables`` -> ``translate_loc_to_scratch`` remapping and
  the split ``k_buffer``/``v_buffer`` read path round-trip correctly.

The FIA-NZ scenario additionally forces the NZ physical cache layout
(``SGLANG_USE_FIA_NZ``) together with the FIA kernel (``ASCEND_USE_FIA``) on top
of sharding, locking the ``_set_fia_nz_kv_buffer`` / ``gather_mla_cache_pages(
is_nz=True)`` invariant on real hardware.

Each scenario needs 4 NPUs: prefill ``tp=2`` on devices 0-1, decode ``tp=2`` on
devices 2-3 (``--base-gpu-id 2``).
"""

import unittest

import requests

from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=1200, suite="base-b-test-4-npu-a3")
register_npu_ci(est_time=1200, suite="nightly-4-npu-a3", nightly=True)

# NPU memory allocator tuning shared by prefill and decode (same as the existing
# NPU PD test in basic_function/pd_disaggregation).
_NPU_ALLOC_CONF = "expandable_segments:True,pin_memory_expandable_segments:True"
# AscendTransferEngine reads its central store URL from this env var; the value
# matches the existing NPU PD test node's local metadata store.
_ASCEND_MF_STORE_URL = "tcp://127.0.0.1:24667"

# Must exceed ``--page-size 64``: the prefix cache matches at page granularity,
# so a sub-page prompt would floor to 0 tokens and never produce a cache hit.
_PREFIX_PROMPT = (
    "The theory of relativity, developed by Albert Einstein in the early "
    "twentieth century, fundamentally changed how physicists understand space, "
    "time, gravity, and the structure of the universe itself. Before Einstein, "
    "physicists believed that space and time were separate and absolute, "
    "providing a fixed stage on which the events of the universe played out. "
    "Einstein showed that space and time are in fact woven together into a "
    "single four-dimensional fabric known as spacetime, and that this fabric "
    "can be bent and curved by the presence of mass and energy. According to "
    "general relativity, what we experience as the force of gravity is actually "
    "the curvature of spacetime caused by massive objects such as stars, "
    "planets, and galaxies. This insight not only explained the motion of "
    "planets more accurately than Newton's laws, but also predicted remarkable "
    "new phenomena including black holes, gravitational waves, and the bending "
    "of light around the sun."
)


class KVCacheShardingPDBase(PDDisaggregationServerBase):
    """Shared PD deployment: Ascend transfer + KV-cache sharding on prefill.

    Subclasses pick the model and the attention-parallel scheme the shard
    stripes across (prefill CP for GQA, plain TP for MLA), plus per-side env for
    FIA-NZ.
    """

    prefill_tp_size = 2
    decode_tp_size = 2
    decode_base_gpu_id = 2

    # Scenario knobs (overridden by subclasses).
    model = None
    prefill_parallel_args = []
    prefill_only_env = {}
    shard_env = {}

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--nccl-port",
            cls.prefill_nccl_port,
            "--tp",
            str(cls.prefill_tp_size),
            "--page-size",
            "64",
            "--chunked-prefill-size",
            "4096",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--enable-kv-cache-sharding",
        ] + cls.prefill_parallel_args
        env = {
            "PYTORCH_NPU_ALLOC_CONF": _NPU_ALLOC_CONF,
            "ASCEND_MF_STORE_URL": _ASCEND_MF_STORE_URL,
            # Sharded rotation metadata requires the Python UnifiedTreeCore.
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
            **cls.prefill_only_env,
            **cls.shard_env,
        }
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--nccl-port",
            cls.decode_nccl_port,
            "--tp",
            str(cls.decode_tp_size),
            "--base-gpu-id",
            str(cls.decode_base_gpu_id),
            "--page-size",
            "64",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
        ]
        env = {
            "PYTORCH_NPU_ALLOC_CONF": _NPU_ALLOC_CONF,
            "ASCEND_MF_STORE_URL": _ASCEND_MF_STORE_URL,
            # Decode reads the same cache layout the prefill side produced.
            **cls.shard_env,
        }
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.launch_all()

    def _generate(self, text, max_new_tokens=32):
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=120,
        )
        self.assertEqual(
            response.status_code, 200, f"Request failed: {response.status_code}"
        )
        return response.json()

    def test_generate_ok(self):
        """A single decoding pass through the sharded read path produces text."""
        data = self._generate(_PREFIX_PROMPT)
        self.assertTrue(data.get("text"), f"empty output: {data!r}")

    def test_prefix_cache_hit(self):
        """A repeated prompt must hit the prefill prefix cache, proving the
        scratch-page remapping feeds a correct (cacheable) read."""
        self._generate(_PREFIX_PROMPT, max_new_tokens=8)
        second = self._generate(_PREFIX_PROMPT, max_new_tokens=8)
        cached = second.get("meta_info", {}).get("cached_tokens", 0)
        self.assertGreater(
            cached, 0, f"expected a prefix-cache hit, got cached_tokens={cached}"
        )


class TestKVCacheShardingGQA(KVCacheShardingPDBase):
    """GQA/MHA shards across prefill attention-CP ranks.

    The NPU GQA prefill-CP attention path only implements the FIA kernel
    (``do_cp_attn_fia``); the non-FIA branch raises ``NotImplementedError``.
    """

    model = QWEN3_8B_WEIGHTS_PATH
    prefill_only_env = {"ASCEND_USE_FIA": "1"}
    prefill_parallel_args = [
        "--enable-prefill-cp",
        "--attn-cp-size",
        "2",
        "--cp-strategy",
        "zigzag",
    ]


class TestKVCacheShardingMLA(KVCacheShardingPDBase):
    """MLA (no prefill CP) shards across attention-TP ranks."""

    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    prefill_parallel_args = [
        "--attn-cp-size",
        "1",
        "--attn-dp-size",
        "1",
    ]


class TestKVCacheShardingMLAFIANZ(KVCacheShardingPDBase):
    """MLA sharding under the FIA NZ cache layout + FIA attention kernel."""

    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    prefill_parallel_args = [
        "--attn-cp-size",
        "1",
        "--attn-dp-size",
        "1",
    ]
    shard_env = {
        "ASCEND_USE_FIA": "1",
        "SGLANG_USE_FIA_NZ": "1",
    }


if __name__ == "__main__":
    unittest.main()