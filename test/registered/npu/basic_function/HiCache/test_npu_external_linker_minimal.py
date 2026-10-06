"""Minimal NPU e2e test proving the ``npu_memcache`` unified-cache external linker works.

Purpose: unlike the full KL suite (which stresses decode cache-hit branching and can
hit DP=16 ``all_gather`` deadlocks), this test only proves on DeepSeek-V4-Flash W8A8
that KV actually gets loaded back from an Ascend MemCache through the
``--unified-cache-external-linker`` direct linker. It self-launches a MetaService,
starts the server with the linker enabled, warms the same prefix twice, and asserts
the 2nd request reports ``cached_tokens_details["host"] > 0`` (tokens pulled from the
remote MemCache).

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
    # deepep
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
}

DEFAULT_SGLANG_HICACHE_CONFIG_ENV = "SGLANG_HICACHE_MEMCACHE_CONFIG_PATH"


class _MemcacheServices:
    """Ascend MemCache MetaService lifecycle for a local single-node config."""

    def __init__(self):
        self.config_path = os.environ.get(DEFAULT_SGLANG_HICACHE_CONFIG_ENV)
        self._owned_config_path = None
        self.process = None

    @staticmethod
    def is_available():
        return importlib.util.find_spec("memcache_hybrid") is not None

    @staticmethod
    def _port_open(host, port):
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

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
        logger.info("Generated MemCache config at %s", path)
        return path

    def start(self):
        if not self.is_available():
            raise RuntimeError("memcache_hybrid is not installed")
        if self.config_path is None:
            self.config_path = self._generate_config()
            self._owned_config_path = self.config_path
        self.log_path = os.path.join(
            tempfile.gettempdir(), f"meta_service_{os.getpid()}.log"
        )
        log_fh = open(self.log_path, "w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.mem_cache.storage.npu_memcache.start_meta_service",
                "--config_path",
                self.config_path,
            ],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        logger.info("Started MetaService pid=%s (log=%s)", self.process.pid, self.log_path)

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


@register_npu_ci(est_time=900, suite="base-b-test-1-npu-a3")
class TestNpuExternalLinkerMinimal(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _MemcacheServices.is_available():
            raise unittest.SkipTest("memcache_hybrid is not installed")
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
                "256",
                "--chunked-prefill-size",
                "2048",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                str(tp_size),
                "--mem-fraction-static",
                "0.62",
                "--enable-cache-report",
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

    def test_remote_host_kv_loadback(self):
        payload = {
            "text": "The capital of France is ",
            "max_new_tokens": 8,
            "sampling_params": {"temperature": 0},
        }
        logger.info("Request 1 (cold / warm into MemCache)...")
        r1 = requests.post(f"{self.base_url}/generate", json=payload, timeout=300)
        r1.raise_for_status()
        meta1 = r1.json()["meta_info"]
        logger.info(
            "req1 cached_tokens=%s details=%s",
            meta1.get("cached_tokens"),
            meta1.get("cached_tokens_details"),
        )

        logger.info("Request 2 (should hit remote MemCache)...")
        r2 = requests.post(f"{self.base_url}/generate", json=payload, timeout=300)
        r2.raise_for_status()
        meta2 = r2.json()["meta_info"]
        details2 = meta2.get("cached_tokens_details") or {}
        host_tokens = int(details2.get("host", 0))
        logger.info(
            "req2 cached_tokens=%s cached_tokens_details=%s",
            meta2.get("cached_tokens"),
            details2,
        )
        self.assertGreater(host_tokens, 0, "linker did not load any KV back from MemCache")


if __name__ == "__main__":
    import unittest

    unittest.main()