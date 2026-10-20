# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Minimal correctness probe for the KV-cache-sharding read path on NPU.

KV-cache sharding (``--enable-kv-cache-sharding``) serves attention from an
*assembled scratch* buffer whose rows are a shuffled, owner-major permutation
of the logical KV (see ``PageInterleaveKVPoolMixin`` in
``mem_cache/page_interleave_pool.py``). The NPU "contiguous" attention path is
``AscendTorchNativeAttnBackend.run_sdpa_forward_extend``, which is pure torch
(``k_cache[req_to_token[...]]`` + SDPA). It therefore has no notion of paged
layout; it reads KV by *row index*.

This probe verifies the exact premise the adapter relies on: a shuffled
scratch together with a correctly re-mapped ``req_to_token`` (row index in the
scratch) must produce the *identical* attention output as an identity
(locally-ordered) layout. If this holds, the device-independent part of the
risk — the ``req_to_token`` re-mapping (risk #2) and the contiguous-vs-paged
layout of the read path (risk #1, contiguous side) — is a pure bookkeeping
problem, not an operator problem.

The backend under test only imports ``torch``, so this runs on CPU.
"""

import sys

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
    AscendTorchNativeAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _run_extend(
    backend: AscendTorchNativeAttnBackend,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_len: int,
    chunk_len: int,
) -> torch.Tensor:
    """Run the contiguous extend path for a single request."""
    seq_len = prefix_len + chunk_len
    output = torch.empty_like(query)
    backend.run_sdpa_forward_extend(
        query=query,
        output=output,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([seq_len]),
        extend_prefix_lens=torch.tensor([prefix_len]),
        extend_seq_lens=torch.tensor([chunk_len]),
        encoder_lens=None,
        is_cross_attention=False,
        scaling=None,
        enable_gqa=False,
        causal=True,
        sliding_window_size=-1,
        logit_cap=0.0,
    )
    return output


@pytest.mark.parametrize("prefix_len,chunk_len", [(3, 2), (5, 1), (4, 4)])
def test_shuffled_scratch_gather_matches_identity(prefix_len: int, chunk_len: int):
    backend = AscendTorchNativeAttnBackend()
    torch.manual_seed(0)

    heads, dim = 4, 8
    seq_len = prefix_len + chunk_len
    max_ctx = 16

    # Logical KV order: token i is kv_seq[i].
    kv_seq_k = torch.randn(seq_len, heads, dim)
    kv_seq_v = torch.randn(seq_len, heads, dim)
    query = torch.randn(chunk_len, heads, dim)

    # Reference: identity layout (locally-ordered scratch).
    req_ref = torch.full((1, max_ctx), -1, dtype=torch.int64)
    req_ref[0, :seq_len] = torch.arange(seq_len)
    out_ref = _run_extend(
        backend, query, kv_seq_k, kv_seq_v, req_ref, prefix_len, chunk_len
    )

    # Probe: token i is stored at scratch row perm[i] (shuffled scratch).
    perm = torch.randperm(seq_len)
    k_scratch = torch.zeros(seq_len, heads, dim)
    v_scratch = torch.zeros(seq_len, heads, dim)
    k_scratch[perm] = kv_seq_k
    v_scratch[perm] = kv_seq_v

    # req_to_token stores the scratch row index of each token.
    req_shuffled = torch.full((1, max_ctx), -1, dtype=torch.int64)
    req_shuffled[0, :seq_len] = perm
    out_shuffled = _run_extend(
        backend, query, k_scratch, v_scratch, req_shuffled, prefix_len, chunk_len
    )

    torch.testing.assert_close(out_shuffled, out_ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))