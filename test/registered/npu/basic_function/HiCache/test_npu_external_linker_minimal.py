"""Minimal NPU e2e test proving the ``npu_memcache`` unified-cache external linker works.

Purpose: unlike the full KL suite (which stresses decode cache-hit branching and can
hit DP=16 ``all_gather`` deadlocks), this test only proves on DeepSeek-V4-Flash W8A8
that KV actually gets loaded back from an Ascend MemCache through the
``--unified-cache-external-linker`` direct linker. It self-launches a MetaService,
starts the server with the linker enabled, warms a multi-page prefix twice (the 2nd
hit triggers the lazy write-through offload), flushes the device radix tree, and
asserts a 3rd request reports ``cached_tokens_details["host"] > 0`` (tokens pulled
back from the remote MemCache).

Run:
    python3 -m pytest test/registered/npu/basic_function/HiCache/\
        test_npu_external_linker_minimal.py -v

Prerequisites: ``memcache_hybrid`` installed, 16 NPU cards, and the DSV4 W8A8
(modelslim) checkpoint present (default set via
``DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH``).
"""

import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

logger = logging.getLogger(__name__)

DSV4_FLASH_LAUNCH_TIMEOUT = 1800
META_SERVICE_SETUP_TIMEOUT = 300

# The radix tree only caches page-aligned prefixes (`page_size` = 128, configured
# below). A prompt shorter than one page is truncated to 0 tokens by
# `RadixKey.page_aligned(128)`, so nothing is ever inserted, offloaded, or loaded
# back. This prompt is several pages of English prose so the device KV for a whole
# page (several pages, really) is cached and offloaded. The exact token count is
# model-tokenizer dependent; this is comfortably above 2 pages for every common
# English tokenizer.
PROMPT_LONG = (
    "A large language model is a statistical system trained on a very large "
    "corpus of text drawn from the public internet. During training the model "
    "learns to predict the next token in a sequence given the tokens that came "
    "before it. This simple objective, repeated across trillions of tokens, is "
    "enough to produce a system that can answer questions, summarize documents, "
    "translate between languages, write computer programs, and follow complex "
    "multi-step instructions. The model represents text as a sequence of tokens, "
    "where each token is a short fragment of text, typically a few characters or "
    "a single word. The Transformer architecture processes the whole sequence in "
    "parallel using the attention mechanism, which lets every position attend to "
    "every other position according to a learned similarity score. The attention "
    "score for a pair of positions is computed as a dot product between a query "
    "vector derived from one position and a key vector derived from the other, "
    "followed by a softmax normalization and a weighted sum over value vectors. "
    "In practice the computation is split into many attention heads so that the "
    "model can track different kinds of relationship in parallel. The output of "
    "each layer is added to its input through a residual connection and passed "
    "through layer normalization, which keeps the activations on a stable scale. "
    "All of these layers are stacked deeply, and the final layer produces a "
    "distribution over the vocabulary from which the next token is sampled. "
    "Inference proceeds autoregressively: the model generates one token, appends "
    "it to the sequence, and repeats the process until a stop condition is met. "
    "To serve these models efficiently, systems cache the intermediate keys and "
    "values that the attention layers compute, so that repeated computations over "
    "a shared prefix can be avoided. A radix tree over token sequences is a "
    "natural data structure for this cache, because it lets the server share the "
    "cached computation across requests that begin with the same prompt. When a "
    "request shares a prefix with an earlier request, the server reuses the "
    "already computed keys and values for the shared part and only computes the "
    "remainder. This reduces latency and increases throughput for workloads that "
    "reuse long system prompts or few-shot examples. The keys and values can also "
    "be moved out of the accelerator memory to CPU memory or to a remote store, "
    "which is useful when the working set is larger than the local memory. The "
    "difficulty is that remote memory has higher latency and lower bandwidth, so "
    "the system must decide carefully which entries to keep locally and which to "
    "evict. A common policy is write-through caching, in which an entry is copied "
    "to the remote store once it has been accessed enough times to justify the "
    "cost of the transfer. This describes the general design of the feature that "
    "this test exercises on the NPU, and the paragraph is intentionally long so "
    "that the prompt spans multiple cache pages and triggers a real offload of "
    "the cached keys and values to the remote memory store, followed by a load "
    "back from the remote store after the local cache has been flushed."
)

# Aligned with the CI case's DSV4 W8A8 env block and its deepep + DP attention env.
# DeepSeek-V4-Flash needs DP attention at tp_size=16 (attn_tp_size collapses to 1),
# otherwise the sparse o_proj reshapes by n_local_groups = o_groups // attn_tp_size
# -> 0 and forward crashes. The SGLANG_OPT_* flags disable CUDA/ROCm fast-paths whose
# quantized layouts do not match the modelslim W8A8 checkpoint.
DEEPSEEK_V4_FLASH_W8A8_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
    # Stop /health_generate from issuing a real one-token generation probe. Under
    # DP=16 that probe stalls (unstable HCCL/comm ports) and flips the server into
    # ServerStatus.UnHealthy with a persistent 503; bypassing it makes health return
    # 200 so the driver can issue its own /generate requests.
    "SGLANG_DIAG_BYPASS_HEALTH_GENERATE": "1",
    # Avoid device-stream wedge when the direct linker does DMA collide with the
    # compute stream during prefill offload (cold prefill works without the linker,
    # hangs with it). Turn off layer-wise overlap, a known mitigation for this.
    # NOTE: read directly by the direct linker via os.getenv, so it belongs in the
    # server process env below.
    "SGLANG_NPU_MEMCACHE_LINKER_LAYERWISE": "0",
    # deepep
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
}

DEFAULT_SGLANG_HICACHE_CONFIG_ENV = "SGLANG_HICACHE_MEMCACHE_CONFIG_PATH"


class _MemcacheServices:
    """Ascend MemCache MetaService lifecycle for a local single-node config.

    Mirrors ``NpuMemcacheTestServices`` from the full HiCache KL case: it reuses an
    already-running MetaService when one exists (memcache holds a node-global lock so
    only one may run per host), and otherwise launches one and waits until its TCP
    config-store / metrics endpoints are reachable before returning. Skipping the
    readiness wait made the server's ``DistributedObjectStore`` client connect before
    the MetaService was up, failing with ``Failed to connect ... after tried 60 times``.
    """

    def __init__(self):
        self.config_path = os.environ.get(DEFAULT_SGLANG_HICACHE_CONFIG_ENV)
        self._owned_config_path = None
        self._log_path = None
        self.process = None

    @staticmethod
    def is_available():
        return importlib.util.find_spec("memcache_hybrid") is not None

    @property
    def _config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def meta_service_url(self):
        return self._config.get("meta_service_url", "tcp://127.0.0.1:5000")

    @property
    def config_store_url(self):
        return self._config.get("config_store_url", "tcp://127.0.0.1:6000")

    @property
    def metrics_url(self):
        return self._config.get("metrics_url", "http://127.0.0.1:8000")

    def _generate_config(self):
        cfg = {
            "meta_service_url": f"tcp://127.0.0.1:{find_available_port(5000)}",
            "config_store_url": f"tcp://127.0.0.1:{find_available_port(6000)}",
            "metrics_url": f"http://127.0.0.1:{find_available_port(8000)}",
            "log_level": "info",
            "protocol": os.environ.get(
                "SGLANG_NPU_MEMCACHE_LINKER_PROTOCOL", "device_sdma"
            ),
            "dram_size": os.environ.get("SGLANG_NPU_MEMCACHE_LINKER_DRAM_SIZE", "1GB"),
            "hbm_size": 0,
            "world_size": int(
                os.environ.get("SGLANG_NPU_MEMCACHE_LINKER_WORLD_SIZE", "256")
            ),
        }
        fd, path = tempfile.mkstemp(prefix="npu_memcache_config_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        logger.info("Generated MemCache config at %s: %s", path, cfg)
        return path

    @staticmethod
    def _discover_running_meta_service_config():
        """Return the ``--config_path`` of a start_meta_service already running on
        this host, or None.

        memcache_hybrid keeps a node-global lock in C++ that cannot be read from
        Python, so when no explicit config is provided we discover an already-running
        MetaService from the process table and reuse its config instead of launching a
        competing instance and tripping the lock.
        """
        try:
            out = subprocess.run(
                ["ps", "-eo", "pid,args"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.splitlines():
            if "start_meta_service" not in line or "--config_path" not in line:
                continue
            tokens = line.split()
            for i, tok in enumerate(tokens):
                if tok == "--config_path" and i + 1 < len(tokens):
                    path = tokens[i + 1].strip()
                    if path.startswith("http") or not os.path.exists(path):
                        continue
                    return path
        return None

    def start(self):
        if not self.is_available():
            raise RuntimeError("memcache_hybrid is not installed")

        if self.config_path is None:
            logger.info(
                "SGLANG_HICACHE_MEMCACHE_CONFIG_PATH is unset; generating a local "
                "single-node MemCache config."
            )
            existing = self._discover_running_meta_service_config()
            if existing is not None:
                logger.info("Reusing running MetaService config %s", existing)
                self.config_path = existing
                self._owned_config_path = None
                self.process = None
                return
            self.config_path = self._generate_config()
            self._owned_config_path = self.config_path
        elif not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"MetaService config not found at {self.config_path}"
            )

        # If a service is already reachable at the configured endpoints, reuse it
        # rather than launching a competing instance and tripping the lock.
        if self._probe_services():
            logger.info("Reusing already-running MetaService from %s", self.config_path)
            self.process = None
            return

        cmd = [
            sys.executable,
            "-m",
            "sglang.srt.mem_cache.storage.npu_memcache.start_meta_service",
            "--config_path",
            self.config_path,
        ]
        fd, self._log_path = tempfile.mkstemp(
            prefix="npu_memcache_metaservice_", suffix=".log"
        )
        os.close(fd)
        log_stream = open(self._log_path, "w", encoding="utf-8")
        logger.info("Starting Ascend MemCache MetaService: %s", " ".join(cmd))
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_stream.close()
        try:
            self._wait_until_ready()
        except Exception:
            self._dump_log()
            raise

    def _wait_until_ready(self):
        deadline = time.monotonic() + META_SERVICE_SETUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self._log_tail()
                raise RuntimeError(
                    f"Ascend MemCache MetaService exited with code "
                    f"{self.process.returncode}.\n{tail}"
                )
            if self._probe_services():
                logger.info("Ascend MemCache MetaService is ready.")
                return
            time.sleep(3)
        raise TimeoutError(
            f"Timed out after {META_SERVICE_SETUP_TIMEOUT}s waiting for "
            f"Ascend MemCache MetaService"
        )

    def _probe_services(self):
        for url in (self.meta_service_url, self.config_store_url, self.metrics_url):
            parsed = urlparse(url)
            host = parsed.hostname or "127.0.0.1"
            if parsed.scheme in ("http", "https"):
                try:
                    requests.get(url, timeout=3)
                    return True
                except requests.RequestException:
                    continue
            if self._is_port_open(host, parsed.port or 5000):
                return True
        return False

    @staticmethod
    def _is_port_open(host, port):
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    def _log_tail(self):
        if not self._log_path or not os.path.exists(self._log_path):
            return "(no meta service log)"
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                return "MetaService log ({}):\n{}".format(
                    self._log_path, ("".join(f.readlines()[-50:])).rstrip() or "(empty)"
                )
        except OSError as e:
            return f"(failed to read MetaService log: {e})"

    def _dump_log(self):
        if self._log_path and os.path.exists(self._log_path):
            logger.error("%s", self._log_tail())

    def server_env(self):
        return {DEFAULT_SGLANG_HICACHE_CONFIG_ENV: self.config_path}

    def stop(self):
        if self.process is not None:
            try:
                self.process.kill()
            except Exception:
                pass
            try:
                self.process.wait(timeout=10)
            except Exception:
                pass
            self.process = None
        if self._owned_config_path and os.path.exists(self._owned_config_path):
            os.remove(self._owned_config_path)
            self._owned_config_path = None


register_npu_ci(est_time=900, suite="base-b-test-1-npu-a3")


class TestNpuExternalLinkerMinimal(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _MemcacheServices.is_available():
            raise unittest.SkipTest("memcache_hybrid is not installed")
        # The MemCache config's ``protocol`` field is read by _generate_config()
        # from THIS (pytest) process's os.environ, not from the server env dict
        # passed to popen_launch_server. Setting it here (before start()) is what
        # actually overrides device_sdma. host_shm keeps offload/load off the
        # device DMA path to avoid the prefill stream wedge; override externally
        # via SGLANG_NPU_MEMCACHE_LINKER_PROTOCOL to try host_rdma/device_sdma.
        os.environ.setdefault("SGLANG_NPU_MEMCACHE_LINKER_PROTOCOL", "host_shm")
        cls.memcache = _MemcacheServices()
        cls.memcache.start()

        tp_size = 16
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.process = popen_launch_server(
            DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
            cls.base_url,
            timeout=DSV4_FLASH_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--tp-size",
                str(tp_size),
                "--attention-backend",
                "dsv4",
                "--disable-cuda-graph",
                "--quantization",
                "modelslim",
                "--dp-size",
                str(tp_size),
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--page-size",
                "128",
                "--chunked-prefill-size",
                "131072",
                "--max-prefill-tokens",
                "80000",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                str(tp_size),
                "--mem-fraction-static",
                "0.68",
                "--enable-cache-report",
                # Skip the server warmup run: under DP=16 the warmup generation
                # request stalls (unstable HCCL/comm ports), so server_status stays
                # ServerStatus.Starting and /health_generate returns 503 forever.
                # Skipping it sets server_status to Up immediately (http_server.py
                # launches warmup only when skip_server_warmup is False).
                "--skip-server-warmup",
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-backend",
                "npu_memcache",
            ],
            env={**DEEPSEEK_V4_FLASH_W8A8_ENVS, **cls.memcache.server_env()},
            device="npu",
        )
        # Wait until the HTTP server is healthy.
        for _ in range(int(DSV4_FLASH_LAUNCH_TIMEOUT)):
            try:
                if requests.get(f"{cls.base_url}/health_generate", timeout=5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("server did not become healthy")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        if getattr(cls, "memcache", None) is not None:
            cls.memcache.stop()

    def _generate(self, text):
        payload = {
            "text": text,
            "max_new_tokens": 8,
            "sampling_params": {"temperature": 0},
        }
        r = requests.post(f"{self.base_url}/generate", json=payload, timeout=300)
        r.raise_for_status()
        meta = r.json()["meta_info"]
        return meta, meta.get("cached_tokens_details") or {}

    def test_remote_host_kv_loadback(self):
        # write-through is lazy under the external linker: a prefix node is
        # offloaded to MemCache only after it has been hit (--page-size 256 makes
        # the prompt several pages, so plain /generate does insert real nodes).
        # The reference pattern (flexkv verify_outputs.py) is:
        #   R1 warm -> R2 hit (backs up to MemCache, async) -> flush device tree
        #   -> R3 misses device, hits MemCache, loads KV back.
        logger.info("Request 1 (cold: insert device KV)...")
        meta1, _ = self._generate(PROMPT_LONG)
        logger.info(
            "req1 cached_tokens=%s details=%s",
            meta1.get("cached_tokens"),
            meta1.get("cached_tokens_details"),
        )

        logger.info("Request 2 (warm: full device hit -> write-through offload)...")
        meta2, _ = self._generate(PROMPT_LONG)
        logger.info(
            "req2 cached_tokens=%s details=%s",
            meta2.get("cached_tokens"),
            meta2.get("cached_tokens_details"),
        )

        # Let the async offload thread finish the batch_put to MemCache.
        logger.info("Waiting for async offload to settle...")
        time.sleep(5)

        logger.info("Flushing device radix tree (MemCache keeps the offloaded KV)...")
        flush_resp = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            timeout=60,
        )
        flush_resp.raise_for_status()

        logger.info("Request 3 (device empty -> should load KV back from MemCache)...")
        meta3, details3 = self._generate(PROMPT_LONG)
        host_tokens = int(details3.get("host", 0))
        logger.info(
            "req3 cached_tokens=%s cached_tokens_details=%s",
            meta3.get("cached_tokens"),
            details3,
        )
        self.assertGreater(
            host_tokens, 0, "linker did not load any KV back from MemCache"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()