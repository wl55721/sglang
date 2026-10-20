"""Unit tests for NPU logical-page KV-cache sharding (MHA + MLA paths).

These tests verify the NPU adaptation of ``--enable-kv-cache-sharding``
without launching a server:

1. ``NPUPageInterleaveMHATokenToKVPool`` and
   ``NPUPageInterleaveMLATokenToKVPool`` resolve the diamond inheritance so
   the shard mechanics (``PageInterleaveKVPoolMixin``) win over the NPU base;
2. their ``_create_gather_comm`` swap the CUDA ``PyNcclCommunicator`` for a
   dedicated HCCL ``NpuKVGatherCommunicator``;
3. the MLA subclass overrides the fused-``kv_buffer`` accessors into the NPU
   split ``k_buffer``/``v_buffer`` layout instead of inheriting the CUDA
   triton path;
4. ``validate_kv_shard_attention_backend`` now admits the ``ascend`` prefill
   backend alongside ``fa3``.

These are pure import-time/structural checks; the backend gate exercises the
arg-validation relaxation directly at the module boundary.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="base-a-test-npu")

# Mock NPU-only modules before importing the source modules, matching the
# sibling NPU attention tests (memory_pool_npu does `if is_npu(): import
# torch_npu`, and page_interleave_pool pulls pynccl through memory_pool).
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(_, MagicMock())

from sglang.srt.arg_groups import kv_shard_hook
from sglang.srt.arg_groups.kv_shard_hook import validate_kv_shard_attention_backend
from sglang.srt.distributed.device_communicators.npu_communicator import (
    NpuKVGatherCommunicator,
)
from sglang.srt.hardware_backend.npu.attention.mla_cache import gather_mla_cache_pages
from sglang.srt.hardware_backend.npu.memory_pool_npu import (
    NPUMHATokenToKVPool,
    NPUMLATokenToKVPool,
    NPUPageInterleaveMHATokenToKVPool,
    NPUPageInterleaveMLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveKVPoolMixin,
    PageInterleaveMHATokenToKVPool,
    PageInterleaveMLATokenToKVPool,
)


class TestNPUPageInterleaveMHAPoolMRO(unittest.TestCase):
    def test_inherits_both_bases(self):
        self.assertTrue(
            issubclass(
                NPUPageInterleaveMHATokenToKVPool, PageInterleaveMHATokenToKVPool
            )
        )
        self.assertTrue(
            issubclass(NPUPageInterleaveMHATokenToKVPool, NPUMHATokenToKVPool)
        )
        self.assertTrue(issubclass(NPUPageInterleaveMHATokenToKVPool, MHATokenToKVPool))

    def test_mro_orders_shard_mechanics_before_npu_base(self):
        """The shard mixin must precede the NPU base so shard getters/writes win."""
        mro = NPUPageInterleaveMHATokenToKVPool.__mro__
        self.assertIs(mro[0], NPUPageInterleaveMHATokenToKVPool)
        self.assertLess(
            mro.index(PageInterleaveMHATokenToKVPool),
            mro.index(PageInterleaveKVPoolMixin),
        )
        self.assertLess(
            mro.index(PageInterleaveKVPoolMixin),
            mro.index(NPUMHATokenToKVPool),
        )
        self.assertLess(
            mro.index(NPUMHATokenToKVPool),
            mro.index(MHATokenToKVPool),
        )

    def test_gather_comm_override_is_defined_on_subclass(self):
        self.assertIn("_create_gather_comm", NPUPageInterleaveMHATokenToKVPool.__dict__)


class TestNPUPageInterleaveMHAGatherComm(unittest.TestCase):
    def test_create_gather_comm_returns_npu_gather_communicator(self):
        pool = object.__new__(NPUPageInterleaveMHATokenToKVPool)
        shard_group = SimpleNamespace(device_group=object())
        comm = pool._create_gather_comm(shard_group)
        self.assertIsInstance(comm, NpuKVGatherCommunicator)


class TestNPUPageInterleaveMLAPoolMRO(unittest.TestCase):
    def test_inherits_both_bases(self):
        self.assertTrue(
            issubclass(
                NPUPageInterleaveMLATokenToKVPool, PageInterleaveMLATokenToKVPool
            )
        )
        self.assertTrue(
            issubclass(NPUPageInterleaveMLATokenToKVPool, NPUMLATokenToKVPool)
        )
        self.assertTrue(
            issubclass(NPUPageInterleaveMLATokenToKVPool, MLATokenToKVPool)
        )

    def test_mro_orders_shard_mechanics_before_npu_base(self):
        """The shard mixin must precede the NPU base so shard getters/writes win."""
        mro = NPUPageInterleaveMLATokenToKVPool.__mro__
        self.assertIs(mro[0], NPUPageInterleaveMLATokenToKVPool)
        self.assertLess(
            mro.index(PageInterleaveMLATokenToKVPool),
            mro.index(PageInterleaveKVPoolMixin),
        )
        self.assertLess(
            mro.index(PageInterleaveKVPoolMixin),
            mro.index(NPUMLATokenToKVPool),
        )
        self.assertLess(
            mro.index(NPUMLATokenToKVPool),
            mro.index(MLATokenToKVPool),
        )

    def test_split_layout_accessors_are_overridden_on_subclass(self):
        # NPU keeps latent/RoPE in separate page-major buffers while the CUDA
        # base reads a fused ``kv_buffer``. These overrides must stay on the
        # NPU subclass so the sharded read/write path never falls back to the
        # fused triton path.
        for method in (
            "_create_gather_comm",
            "_scratch_tensor_specs",
            "_gather_pairs",
            "get_kv_buffer",
            "get_key_buffer",
            "get_value_buffer",
            "get_kv_buffer_shape",
            "set_kv_buffer",
        ):
            with self.subTest(method=method):
                self.assertIn(method, NPUPageInterleaveMLATokenToKVPool.__dict__)


class TestNPUPageInterleaveMLAGatherComm(unittest.TestCase):
    def test_create_gather_comm_returns_npu_gather_communicator(self):
        pool = object.__new__(NPUPageInterleaveMLATokenToKVPool)
        shard_group = SimpleNamespace(device_group=object())
        comm = pool._create_gather_comm(shard_group)
        self.assertIsInstance(comm, NpuKVGatherCommunicator)


class TestNPUPageInterleaveMLAPrefixRead(unittest.TestCase):
    """End-to-end prefill prefix-cache read path for the sharded MLA pool.

    The layer-ahead gather is an HCCL collective that cannot run on CPU, so we
    construct the *already-assembled* scratch directly (the owner-major page
    layout the gather would have produced) and then drive the real readers:
    ``get_key_buffer``/``get_value_buffer`` -> page-major reshape, the
    ``flatten_prefix_block_tables`` translation -> scratch page ids, and
    ``gather_mla_cache_pages``. The invariant under test: reading any logical
    page vector through the sharded scratch returns the identical KV as the
    unsharded, logical page order.
    """

    PS = 2
    SHARD_SIZE = 4
    KV_DIM = 4
    KR_DIM = 2
    # FIA-NZ requires a head dimension divisible by 16 (tile width).
    NZ_KV_DIM = 32
    NZ_KR_DIM = 16

    def _logical_page(self, lp: int, dim: int, offset: float) -> torch.Tensor:
        """Encode logical token ``lp*PS + slot`` as its KV feature value."""
        t = torch.arange(self.PS, dtype=torch.float32) + lp * self.PS + offset
        return t.view(self.PS, 1, 1).expand(self.PS, 1, dim)

    @staticmethod
    def _nz_encode_page(page_std: torch.Tensor) -> torch.Tensor:
        """Encode a logical ``[PS, 1, D]`` page into FIA-NZ physical layout.

        Mirrors ``NPUMLATokenToKVPool._set_fia_nz_kv_buffer``: the public
        page-major ``[PS, 1, D]`` storage is read by FIA as
        ``[1, D // 16, PS, 16]``, so logical (slot, tile, sub) scatters to the
        16-wide row ``tile * PS + slot``.
        """
        ps, dim = page_std.shape[0], page_std.shape[-1]
        num_tiles = dim // 16
        assert dim % 16 == 0
        out = torch.zeros(1, ps, 1, dim)
        dst = out.view(-1, 1, num_tiles, ps, 16).view(-1, 16)
        src = page_std.contiguous().view(-1, num_tiles, 16).reshape(-1, 16)
        indices = [
            tile * ps + slot for slot in range(ps) for tile in range(num_tiles)
        ]
        dst[torch.tensor(indices, dtype=torch.long)] = src
        return out.view(ps, 1, dim)

    def _make_pool(self, logical_pages, nz=False):
        P = len(logical_pages)
        pool = object.__new__(NPUPageInterleaveMLATokenToKVPool)
        pool.page_size = self.PS
        pool.shard_spec = SimpleNamespace(page_size=self.PS)
        pool.shard_size = self.SHARD_SIZE
        pool.store_dtype = torch.float32
        pool.dtype = torch.float32
        pool._shard_extend_active = True

        kv_dim = self.NZ_KV_DIM if nz else self.KV_DIM
        kr_dim = self.NZ_KR_DIM if nz else self.KR_DIM

        # Rebuild the per-batch owner-major plan (mirrors begin_shard_extend):
        # sort unique pages by (owner, local page); owner r's pages land at
        # r * block_pages .. in local-page order. The owner-major scratch span
        # is shard_size * block_pages slots (padded to a regular allgather
        # block); slots with no page are the never-referenced trash page.
        pool._page_pos = torch.full(
            (self.SHARD_SIZE * (P + 2),), 0, dtype=torch.int32
        )
        order = sorted(
            range(P),
            key=lambda i: (
                logical_pages[i] % self.SHARD_SIZE,
                logical_pages[i] // self.SHARD_SIZE,
            ),
        )
        block_pages = (P + self.SHARD_SIZE - 1) // self.SHARD_SIZE
        num_slots = self.SHARD_SIZE * block_pages
        within = {}
        for i in order:
            lp = logical_pages[i]
            o = lp % self.SHARD_SIZE
            w = within.get(o, 0)
            pool._page_pos[lp] = o * block_pages + w
            within[o] = w + 1

        # Scratch slot s holds logical page inv[s], token-major [rows, 1, dim].
        inv = [-1] * num_slots
        for i in order:
            inv[int(pool._page_pos[logical_pages[i]])] = logical_pages[i]

        scratch_k = torch.zeros(num_slots * self.PS, 1, kv_dim)
        scratch_v = torch.zeros(num_slots * self.PS, 1, kr_dim)
        k_pages = scratch_k.view(num_slots, self.PS, 1, kv_dim)
        v_pages = scratch_v.view(num_slots, self.PS, 1, kr_dim)
        for s in range(num_slots):
            if inv[s] >= 0:
                k_page = self._logical_page(inv[s], kv_dim, 0.0)
                v_page = self._logical_page(inv[s], kr_dim, 0.5)
                if nz:
                    k_page = self._nz_encode_page(k_page)
                    v_page = self._nz_encode_page(v_page)
                k_pages[s] = k_page
                v_pages[s] = v_page
        return pool, scratch_k, scratch_v, (kv_dim, kr_dim)

    def test_prefix_cache_read_matches_logical_page_order(self):
        logical_pages = [0, 1, 2, 3, 4, 5]
        pool, scratch_k, scratch_v, (kv_dim, kr_dim) = self._make_pool(logical_pages)
        slot = SimpleNamespace(tensors={"k": scratch_k, "v": scratch_v})

        # The getters route through _acquire_slot_for_read (a CUDA stream/event
        # wait on real hardware); stub it here and exercise the real layout +
        # reshape path it feeds into.
        with patch.object(pool, "_acquire_slot_for_read", return_value=slot):
            k_buf = pool.get_key_buffer(0)
            v_buf = pool.get_value_buffer(0)
            k_buf_fused, v_buf_fused = pool.get_kv_buffer(0)

        num_slots = scratch_k.shape[0] // self.PS
        self.assertEqual(k_buf.shape, (num_slots, self.PS, 1, kv_dim))
        self.assertEqual(v_buf.shape, (num_slots, self.PS, 1, kr_dim))
        self.assertTrue(torch.equal(k_buf_fused, k_buf))
        self.assertTrue(torch.equal(v_buf_fused, v_buf))

        # A request's MLA prefix block tables, deliberately non-sorted.
        prefix_block_tables = torch.tensor([3, 0, 5, 1, 4, 2], dtype=torch.int32)
        scratch_page_ids = (
            pool.translate_loc_to_scratch(prefix_block_tables * self.PS) // self.PS
        )

        k_cached = gather_mla_cache_pages(k_buf, scratch_page_ids, is_nz=False)
        k_rope_cached = gather_mla_cache_pages(
            v_buf, scratch_page_ids, is_nz=False
        ).flatten(0, 1)

        expected_k = torch.stack(
            [self._logical_page(lp, kv_dim, 0.0) for lp in prefix_block_tables.tolist()]
        )
        expected_v = torch.stack(
            [self._logical_page(lp, kr_dim, 0.5) for lp in prefix_block_tables.tolist()]
        )

        self.assertTrue(torch.equal(k_cached, expected_k))
        self.assertTrue(torch.equal(k_rope_cached, expected_v.flatten(0, 1)))

    def test_prefix_cache_read_matches_logical_page_order_nz(self):
        logical_pages = [0, 1, 2, 3, 4, 5]
        pool, scratch_k, scratch_v, (kv_dim, kr_dim) = self._make_pool(
            logical_pages, nz=True
        )
        slot = SimpleNamespace(tensors={"k": scratch_k, "v": scratch_v})

        with patch.object(pool, "_acquire_slot_for_read", return_value=slot):
            k_buf = pool.get_key_buffer(0)
            v_buf = pool.get_value_buffer(0)

        prefix_block_tables = torch.tensor([3, 0, 5, 1, 4, 2], dtype=torch.int32)
        scratch_page_ids = (
            pool.translate_loc_to_scratch(prefix_block_tables * self.PS) // self.PS
        )

        k_cached = gather_mla_cache_pages(k_buf, scratch_page_ids, is_nz=True)
        k_rope_cached = gather_mla_cache_pages(
            v_buf, scratch_page_ids, is_nz=True
        ).flatten(0, 1)

        expected_k = torch.stack(
            [self._logical_page(lp, kv_dim, 0.0) for lp in prefix_block_tables.tolist()]
        )
        expected_v = torch.stack(
            [self._logical_page(lp, kr_dim, 0.5) for lp in prefix_block_tables.tolist()]
        )

        self.assertTrue(torch.equal(k_cached, expected_k))
        self.assertTrue(torch.equal(k_rope_cached, expected_v.flatten(0, 1)))


class TestValidateKVShardAttentionBackend(unittest.TestCase):
    def _assert_backend(self, backend):
        # A truthy cfg avoids the resolving_view(server_args) call; only
        # resolved_view (for attention_backends_of) is exercised.
        with (
            patch.object(
                kv_shard_hook, "attention_backends_of", return_value=(backend, None)
            ),
            patch.object(kv_shard_hook, "resolved_view", return_value=None),
        ):
            validate_kv_shard_attention_backend(None, cfg=SimpleNamespace())

    def test_ascend_backend_is_accepted(self):
        self._assert_backend("ascend")

    def test_fa3_backend_is_accepted(self):
        self._assert_backend("fa3")

    def test_unsupported_backend_raises(self):
        with (
            patch.object(
                kv_shard_hook,
                "attention_backends_of",
                return_value=("flashinfer", None),
            ),
            patch.object(kv_shard_hook, "resolved_view", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "fa3"):
                validate_kv_shard_attention_backend(None, cfg=SimpleNamespace())


if __name__ == "__main__":
    unittest.main()