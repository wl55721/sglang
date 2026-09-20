"""NPU e2e test for the ``npu_memcache`` unified-cache external linker.

Mirrors the CUDA :mod:`test_unified_cache_linker_kl_dsv4` case: it self-launches an
Ascend MemCache MetaService, then runs the multi-turn KL log-probability consistency
suite against DeepSeek-V4-Flash. Each KL case warms KV into MemCache once, then
reloads it through the ``--unified-cache-external-linker-backend=npu_memcache``
direct linker (layer-by-layer ``batch_get_into_layers``), asserting both that the
log-probs stay consistent and that remote direct load-back actually happened.

Prerequisites for the worker running this test:
  - ``memcache_hybrid`` installed (otherwise the class is skipped). When
    ``SGLANG_HICACHE_MEMCACHE_CONFIG_PATH`` is unset, a local single-node config
    is generated automatically and its MetaService is launched by the test; set
    that env var to point at a deployable JSON to use an external MemCache. The
    same JSON feeds the MetaService process (``meta_service_url`` /
    ``config_store_url`` / ``metrics_url`` / ``log_level``) and the SGLang
    server's client ``DistributedObjectStore`` (``protocol`` / ``dram_size`` /
    ``world_size``).
  - 16 NPU cards, matching ``tp_size``.
"""

import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.environ import envs
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

logger = logging.getLogger(__name__)

DSV4_FLASH_LAUNCH_TIMEOUT = 3600
META_SERVICE_SETUP_TIMEOUT = 300

# Defaults used when the test generates its own local MetaService config because
# SGLANG_HICACHE_MEMCACHE_CONFIG_PATH is unset. Override via the SGLANG_NPU_MEMCACHE_*
# env vars if the deployment needs a different protocol / pool sizing.
DEFAULT_MEMCACHE_PROTOCOL = "device_sdma"
DEFAULT_MEMCACHE_DRAM_SIZE = "1GB"
DEFAULT_MEMCACHE_WORLD_SIZE = 256

# Minimal env set to run DeepSeek-V4-Flash W8A8 on NPU without MTP/deepep.
DEEPSEEK_V4_FLASH_W8A8_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
}

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class NpuMemcacheTestServices:
    """Lifecycle helper for an Ascend MemCache MetaService subprocess.

    The MetaService and the SGLang server's client-side ``DistributedObjectStore``
    share one JSON config file. ``start_meta_service.py`` applies its ``MetaConfig``
    keys while the server store applies the ``LocalConfig`` keys; unknown keys are
    logged and ignored on each side.
    """

    def __init__(self, config_path=None):
        self.config_path = config_path or envs.SGLANG_HICACHE_MEMCACHE_CONFIG_PATH.get()
        self._owned_config_path = None
        self.process = None

    @staticmethod
    def is_available():
        """Whether ``memcache_hybrid`` (required to launch MetaService) is installed."""
        return importlib.util.find_spec("memcache_hybrid") is not None

    def _generate_default_config(self):
        """Build a local MetaService config when none is provided externally.

        Ports are picked at runtime and the client-side ``LocalConfig`` fields
        (``protocol`` / ``dram_size`` / ``world_size`` / ``hbm_size``) default to
        single-node NPU values; override them with the ``SGLANG_NPU_MEMCACHE_LINKER_*``
        env vars when the deployment needs something else.
        """
        protocol = os.environ.get(
            "SGLANG_NPU_MEMCACHE_LINKER_PROTOCOL", DEFAULT_MEMCACHE_PROTOCOL
        )
        dram_size = os.environ.get(
            "SGLANG_NPU_MEMCACHE_LINKER_DRAM_SIZE", DEFAULT_MEMCACHE_DRAM_SIZE
        )
        world_size = int(
            os.environ.get(
                "SGLANG_NPU_MEMCACHE_LINKER_WORLD_SIZE", DEFAULT_MEMCACHE_WORLD_SIZE
            )
        )
        cfg = {
            "meta_service_url": f"tcp://127.0.0.1:{find_available_port(5000)}",
            "config_store_url": f"tcp://127.0.0.1:{find_available_port(6000)}",
            "metrics_url": f"http://127.0.0.1:{find_available_port(8000)}",
            "log_level": "info",
            "protocol": protocol,
            "dram_size": dram_size,
            "hbm_size": 0,
            "world_size": world_size,
        }
        fd, path = tempfile.mkstemp(prefix="npu_memcache_config_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        logger.info("Generated local MemCache config at %s: %s", path, cfg)
        return path

    @property
    def metrics_url(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("metrics_url", "http://127.0.0.1:8000")

    def start(self):
        if not self.config_path:
            logger.info(
                "SGLANG_HICACHE_MEMCACHE_CONFIG_PATH is unset; generating a local "
                "single-node MemCache config."
            )
            self.config_path = self._generate_default_config()
            self._owned_config_path = self.config_path
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"MetaService config not found at {self.config_path}")

        cmd = [
            sys.executable,
            "-m",
            "sglang.srt.mem_cache.storage.npu_memcache.start_meta_service",
            "--config_path",
            self.config_path,
        ]
        logger.info("Starting Ascend MemCache MetaService: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_until_ready()

    def _wait_until_ready(self):
        parsed = urlparse(self.metrics_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8000
        probe = f"http://{host}:{port}"
        deadline = time.monotonic() + META_SERVICE_SETUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Ascend MemCache MetaService exited with code {self.process.returncode}"
                )
            try:
                if requests.get(probe, timeout=3).status_code == 200:
                    logger.info("Ascend MemCache MetaService is ready at %s", probe)
                    return
            except requests.RequestException:
                pass
            time.sleep(3)
        raise TimeoutError(
            f"Timed out after {META_SERVICE_SETUP_TIMEOUT}s waiting for MetaService at {probe}"
        )

    def stop(self):
        if self.process is not None:
            logger.info("Stopping Ascend MemCache MetaService...")
            try:
                terminate_and_kill_process_tree(self.process)
            except Exception as e:
                logger.warning("Failed to stop MetaService: %s", e)
            finally:
                self.process = None
        if self._owned_config_path is not None:
            try:
                os.remove(self._owned_config_path)
                logger.info("Removed generated MemCache config %s", self._owned_config_path)
            except OSError as e:
                logger.warning("Failed to remove generated MemCache config: %s", e)
            finally:
                self._owned_config_path = None

    def server_env(self):
        """Env to pass to the SGLang server's client-side ``NpuMemcacheStore``."""
        return {
            envs.SGLANG_HICACHE_MEMCACHE_CONFIG_PATH.name: self.config_path,
        }


class TestNpuDeepSeekV4FlashUnifiedCacheLinkerKL(
    UnifiedRadixTreeTestMixin, CustomTestCase
):
    page_size = 256
    kl_threshold = 0.01
    sampling_temperature = 0
    max_new_tokens = 64
    prefix_len = 2048
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5

    tp_size = 16

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.memcache = NpuMemcacheTestServices()
        if not NpuMemcacheTestServices.is_available():
            raise unittest.SkipTest(
                "memcache_hybrid is not installed; skipping npu_memcache "
                "unified-cache external linker e2e."
            )
        cls.memcache.start()
        cls.process = None
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DSV4_FLASH_LAUNCH_TIMEOUT,
                other_args=[
                    "--trust-remote-code",
                    "--device",
                    "npu",
                    "--tp-size",
                    str(cls.tp_size),
                    "--attention-backend",
                    "dsv4",
                    "--quantization",
                    "modelslim",
                    "--page-size",
                    str(cls.page_size),
                    "--chunked-prefill-size",
                    "8192",
                    "--mem-fraction-static",
                    "0.62",
                    "--disable-shared-experts-fusion",
                    "--swa-full-tokens-ratio",
                    "0.25",
                    "--max-total-tokens",
                    "8192",
                    "--max-running-requests",
                    "1",
                    "--enable-cache-report",
                    "--enable-unified-cache-external-linker",
                    "--unified-cache-external-linker-backend",
                    "npu_memcache",
                ],
                env={
                    **DEEPSEEK_V4_FLASH_W8A8_ENVS,
                    **cls.memcache.server_env(),
                },
                device="npu",
            )
            cls.input_ids = get_input_ids(cls.model, num_samples=18)
        except Exception:
            try:
                if cls.process is not None:
                    terminate_and_kill_process_tree(cls.process)
            finally:
                cls.memcache.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.process is not None:
                terminate_and_kill_process_tree(cls.process)
        finally:
            cls.memcache.stop()

    @unittest.skip("Linker CI targets Direct load-back KL accuracy")
    def test_gsm8k(self):
        pass

    @unittest.skip("Linker CI targets Direct load-back KL accuracy")
    def test_mmlu(self):
        pass

    def prefill_cache_assert(self, result, prefix_len, label):
        self._record_cache_result(result, prefix_len, label)

    def decode_cache_assert(self, result, history_len, output_len, label):
        self._record_cache_result(result, history_len + output_len, label)

    def _record_cache_result(self, result, expected_cached_tokens, label):
        meta_info = result["meta_info"]
        cached_tokens = int(meta_info["cached_tokens"])
        minimum = max(0, expected_cached_tokens - self.page_size)
        self.assertGreaterEqual(
            cached_tokens,
            minimum,
            f"{label}: expected cached_tokens >= {minimum}, got {cached_tokens}",
        )
        details = meta_info.get("cached_tokens_details") or {}
        remote_tokens = int(details.get("host", 0))
        self._direct_remote_tokens += remote_tokens
        if remote_tokens:
            print(f"{label}: Direct load-back confirmed for {remote_tokens} tokens")

    def _run_linker_kl_case(self, test_case):
        self._direct_remote_tokens = 0
        test_case()
        print(f"Direct load-back total: {self._direct_remote_tokens} tokens")
        self.assertGreater(
            self._direct_remote_tokens,
            0,
            "Expected this KL case to load KV through the npu_memcache Direct Linker",
        )

    def test_multiturn_logprobs_match(self):
        self._run_linker_kl_case(super().test_multiturn_logprobs_match)

    def test_multiturn_prefill_cache_hit_branching(self):
        self._run_linker_kl_case(super().test_multiturn_prefill_cache_hit_branching)

    def test_multiturn_decode_cache_hit_branching(self):
        self._run_linker_kl_case(super().test_multiturn_decode_cache_hit_branching)


if __name__ == "__main__":
    unittest.main()