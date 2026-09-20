from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future
from queue import Empty, Queue

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            raise RuntimeError("NpuMemcache layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class NpuMemcacheDirectLinker(UnifiedCacheLinker):
    """External KV store reached directly from the device pools via Ascend MemCache.

    Mirrors :class:`MooncakeDirectLinker`'s thread/queue framework, but uses
    ``memcache_hybrid``'s layered object model instead of object-internal byte
    ranges: one page object is stored as ``num_layers`` layer buffers and read
    back layer-by-layer through ``batch_get_into_layers`` / ``batch_put_from_layers``.
    """

    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers

        tp_rank = 0
        tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store import (
                NpuMemcacheStore,
            )

            self.storage = NpuMemcacheStore(storage_config, mem_pool=None)
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        logger.info(
            "NpuMemcache direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )

        self.register_buffers()
        self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        # Env escape hatch to whole-object layered reads: keeps a single
        # batch_get_into_layers call per pool (no compute overlap) instead of the
        # layer-by-layer default, for validating the layered object layout.
        self.load_layerwise = os.getenv(
            "SGLANG_NPU_MEMCACHE_LINKER_LAYERWISE", "1"
        ) not in ("0", "false", "False", "no", "off")
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self.completed_loads: Queue[list[str]] = Queue()
        self.offload_queue: Queue[tuple[list[PoolTransfer], int, object] | None] = (
            Queue()
        )
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"npu-memcache-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"npu-memcache-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register NPU KV buffer with Ascend MemCache, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "NpuMemcache direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"NpuMemcache load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # Already-published loads cannot be safely canceled without tree rollback.
        return False

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self.completed_loads.get_nowait()

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        freeze_gc("NpuMemcache direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        ready_event = device_module.Event()
        ready_event.record()
        self.load_queue.put((counter_index, pending, ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                try:
                    ready_event.synchronize()
                    self.load_layer_wise(counter_index, list(pending.values()))
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("NpuMemcache layer-wise load batch failed")
                finally:
                    self.completed_loads.put(list(pending))
            finally:
                self.load_queue.task_done()

    def _build_batches(
        self, request_transfers: list[list[PoolTransfer]]
    ) -> dict[PoolName, tuple[list[str], list[int]]]:
        batches: dict[PoolName, tuple[list[str], list[int]]] = {}
        for transfers in request_transfers:
            for transfer in transfers:
                keys, locations = batches.setdefault(transfer.name, ([], []))
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys or []), transfer
                )
                keys.extend(self.storage._tag_keys(component_keys))
                locations.extend(
                    self.pools[transfer.name].prepare_locations(transfer.host_indices)
                )
        return batches

    def _full_layer_lists(
        self,
        entry,
        locations: list[int],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Per-object buffer lists covering every logical layer, in layer order.

        This is the exact reverse of the layered read in ``load_layer_wise``; a
        page object is written as ``num_layers`` layer buffers and must be read
        back with the same layering.
        """
        buffer_ptrs_list: list[list[int]] = [[] for _ in locations]
        sizes_list: list[list[int]] = [[] for _ in locations]
        for layer in range(self.num_layers):
            meta = entry.get_prepared_layer_range_meta(locations, layer)
            if meta is None:
                continue
            ptrs, sizes, _offsets = meta
            for index in range(len(locations)):
                buffer_ptrs_list[index].extend(ptrs[index])
                sizes_list[index].extend(sizes[index])
        return buffer_ptrs_list, sizes_list

    def load_layer_wise(
        self, counter_index: int, request_transfers: list[list[PoolTransfer]]
    ) -> None:
        try:
            batches = self._build_batches(request_transfers)
            if self.load_layerwise:
                for layer in range(self.num_layers):
                    self._load_one_layer(batches, layer)
                    self.layer_done_counter.complete(counter_index, layer)
            else:
                self._load_whole_objects(batches)
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("NpuMemcache layer-wise load batch failed")

    def _load_one_layer(
        self, batches: dict[PoolName, tuple[list[str], list[int]]], layer: int
    ) -> None:
        for name, (keys, locations) in batches.items():
            entry = self.pools[name]
            meta = entry.get_prepared_layer_range_meta(locations, layer)
            if meta is None:
                continue
            ptrs, sizes, _offsets = meta
            # ``ptrs[i]`` / ``sizes[i]`` already carry the per-component buffers of
            # this single layer; read exactly that layer for every page object.
            if keys and not self.storage.batch_get_into_layers(keys, ptrs, sizes):
                raise RuntimeError(
                    f"NpuMemcache layered get failed for pool={name}, layer={layer}."
                )

    def _load_whole_objects(
        self, batches: dict[PoolName, tuple[list[str], list[int]]]
    ) -> None:
        for name, (keys, locations) in batches.items():
            if not keys:
                continue
            entry = self.pools[name]
            buffer_ptrs_list, sizes_list = self._full_layer_lists(entry, locations)
            if not self.storage.batch_get_into_layers(keys, buffer_ptrs_list, sizes_list):
                raise RuntimeError(
                    f"NpuMemcache layered get failed for pool={name} (whole object)."
                )

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys or []) * self.page_size
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put((expanded, tokens, ready_event))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready_event = task
                ready_event.synchronize()
                results = self._run_offload(expanded)
                if results:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info(
                            "NpuMemcache direct linker offload: tokens=%d", tokens
                        )
                self.offload_results.put(results)
            except BaseException:
                logger.exception("NpuMemcache offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def _run_offload(self, expanded: list[PoolTransfer]) -> bool:
        batches = self._build_batches([expanded])
        for name, (keys, locations) in batches.items():
            if not keys:
                continue
            entry = self.pools[name]
            buffer_ptrs_list, sizes_list = self._full_layer_lists(entry, locations)
            if not self.storage.batch_put_from_layers(
                keys, buffer_ptrs_list, sizes_list
            ):
                logger.warning(
                    "NpuMemcache offload failed: pool=%s keys=%d",
                    name,
                    len(keys),
                )
                return False
        return True

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        self.pending_loads.clear()
        self.load_queue.join()
        self.offload_queue.join()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.completed_loads.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("NpuMemcache direct linker stats: %s", self.stats)
        self.storage.close()