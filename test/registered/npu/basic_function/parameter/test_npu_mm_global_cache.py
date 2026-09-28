import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from urllib.parse import urlparse

import openai

from sglang.test.ascend.test_ascend_utils import (
    IMAGES_023_PATH,
    IMAGES_LOGO_PATH,
    KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_launch_server,
    popen_with_error_check,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="base-b-test-3-npu-a3")


def _data_url_from_image(image_path: str, mime: str = "image/png") -> str:
    """Build an inline data URL for a local image so the OpenAI client can send it."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class TestNpuMmGlobalCache(CustomTestCase):
    """NPU e2e for `--enable-mm-global-cache --mm-global-cache-backend npu_memcache`.

    Launches an EPD topology (encoder + prefill + decode + LB) on NPU:
      - encoder: --encoder-only --encoder-transfer-backend zmq_to_scheduler,
        --enable-mm-global-cache --mm-global-cache-backend npu_memcache
      - prefill/decode KV transfer: --disaggregation-transfer-backend ascend
      - card split: encode(gpu0) + prefill(gpu1) + decode(gpu2), each tp=1

    The global embed cache relies on Ascend MemCache (MetaService/LocalService).
    When SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH is not provided, the test
    self-launches a local MetaService and generates its own LocalService config
    so the cache-hit path runs end-to-end (no external MemCache deployment).

    [Test Category] Functional
    [Test Target] --enable-mm-global-cache / --mm-global-cache-backend=npu_memcache
    """

    @classmethod
    def _ensure_memcache(cls):
        """Make a MemCache (MetaService + LocalConfig) available for the test.

        Reuses an external deployment when SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH
        is set; otherwise spins up a local MetaService via
        ``sglang.srt.mem_cache.storage.npu_memcache.start_meta_service`` and writes a
        LocalService JSON config, leaking neither the meta process nor the config.
        """
        cls._meta_proc = None
        cls._meta_tmpdir = None
        if cls.cache_cfg_path:
            return

        cls._meta_tmpdir = tempfile.mkdtemp(prefix="npu_mm_global_cache_meta_")
        meta_port, cfg_port = 25037, 25038
        meta_url = f"tcp://127.0.0.1:{meta_port}"
        cfg_store_url = f"tcp://127.0.0.1:{cfg_port}"

        meta_cfg_path = os.path.join(cls._meta_tmpdir, "metaservice_config.json")
        with open(meta_cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta_service_url": meta_url,
                    "config_store_url": cfg_store_url,
                    "log_level": "info",
                },
                f,
            )

        cls.cache_cfg_path = os.path.join(
            cls._meta_tmpdir, "localservice_config.json"
        )
        # Make the generated config visible to the encoder subprocess. memcache_env
        # is snapshotted after this, so every worker reads the local MemCache config.
        os.environ["SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH"] = cls.cache_cfg_path
        with open(cls.cache_cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta_service_url": meta_url,
                    "config_store_url": cfg_store_url,
                    "log_level": "info",
                    # Values below mirror the HiCache npu_memcache LocalService
                    # config (see npu_memcache/README.md). device_sdma runs over
                    # device SDMA channels and needs no host RDMA library
                    # (libhcom.so), so it works on a bare NPU node; host_* protocol
                    # would try to load the missing host RDMA extend lib during BM
                    # init. device_id/init_bm are control keys consumed by SGLang.
                    "world_size": 256,
                    "protocol": "device_sdma",
                    "dram_size": "1GB",
                    "device_id": 0,
                    "init_bm": True,
                },
                f,
            )

        meta_log = open(
            os.path.join(cls._meta_tmpdir, "meta.out"), "w", encoding="utf-8"
        )
        cls._meta_err = open(
            os.path.join(cls._meta_tmpdir, "meta.err"), "w", encoding="utf-8"
        )
        cmd = [
            "python3",
            "-m",
            "sglang.srt.mem_cache.storage.npu_memcache.start_meta_service",
            "--config_path",
            meta_cfg_path,
        ]
        print("Starting MetaService:", " ".join(cmd))
        cls._meta_proc = subprocess.Popen(
            cmd, stdout=meta_log, stderr=cls._meta_err, env=os.environ.copy()
        )
        # Give MetaService time to bind its listeners before the encoder connects.
        time.sleep(5)

    @classmethod
    def setUpClass(cls):
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        base_port = str(parsed.port)
        cls.lb_port = base_port
        cls.encode_port = f"{int(base_port) + 300}"
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.bootstrap_port = f"{int(base_port) + 500}"
        cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"

        cls.model = KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH
        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.lb_url}/v1"
        cls.image_url = _data_url_from_image(IMAGES_LOGO_PATH)
        # A different image used to verify cache-key isolation (must miss).
        cls.other_image_url = _data_url_from_image(IMAGES_023_PATH)

        cls.cache_cfg_path = os.environ.get(
            "SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH"
        )

        # Ensure MemCache config is generated before snapshotting the env so the
        # SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH we just set is inherited by
        # every worker subprocess.
        cls._ensure_memcache()
        cls.memcache_env = dict(os.environ)
        # AscendTransferEngine (PD KV transfer on prefill/decode) reads
        # ASCEND_MF_STORE_URL; point it at the environment-provided MemFabric
        # ConfigStore so the store_url is not null when initializing the C++ engine.
        cls.memcache_env["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24667"

        cls.encode_stdout = io.StringIO()
        cls.encode_stderr = io.StringIO()
        cls.start_encode()
        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(cls.encode_url + "/health", process=cls.process_encode)
        cls.wait_server_ready(
            cls.prefill_url + "/health", process=cls.process_prefill
        )
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_router()
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)
        time.sleep(5)

    @staticmethod
    def wait_server_ready(url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=None):
        from sglang.utils import wait_for_http_ready

        wait_for_http_ready(url=url, timeout=timeout, process=process)
        print(f"Server {url} is ready")

    @classmethod
    def start_encode(cls):
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--base-gpu-id",
            "0",
            "--tp-size",
            "1",
            "--port",
            cls.encode_port,
            "--disable-cuda-graph",
        ]
        encode_args += [
            "--enable-mm-global-cache",
            "--mm-global-cache-backend",
            "npu_memcache",
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=cls.memcache_env,
            return_stdout_stderr=(cls.encode_stdout, cls.encode_stderr),
        )

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--language-only",
            "--encoder-urls",
            cls.encode_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--base-gpu-id",
            "1",
            "--tp-size",
            "1",
            "--port",
            cls.prefill_port,
            "--disable-cuda-graph",
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=cls.memcache_env,
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
            "--base-gpu-id",
            "2",
            "--tp-size",
            "1",
            "--port",
            cls.decode_port,
            "--disable-cuda-graph",
        ]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=cls.memcache_env,
        )

    @classmethod
    def launch_router(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", " ".join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)

    @classmethod
    def tearDownClass(cls):
        for proc in [
            cls.process_lb,
            cls.process_decode,
            cls.process_prefill,
            cls.process_encode,
        ]:
            if proc:
                try:
                    terminate_and_kill_process_tree(proc)
                except Exception as e:
                    print(f"Error killing process {proc.pid}: {e}")

        if getattr(cls, "_meta_proc", None) and cls._meta_proc.poll() is None:
            try:
                terminate_and_kill_process_tree(cls._meta_proc)
            except Exception as e:
                print(f"Error killing MetaService process: {e}")
        meta_err = getattr(cls, "_meta_err", None)
        if meta_err is not None:
            try:
                meta_err.close()
            except Exception:
                pass
        if getattr(cls, "_meta_tmpdir", None) and os.path.exists(cls._meta_tmpdir):
            shutil.rmtree(cls._meta_tmpdir, ignore_errors=True)

    def _client(self):
        return openai.Client(api_key=self.api_key, base_url=f"{self.lb_url}/v1")

    def _parse_cache_log(self):
        """Parse '=== Multi-Level Cache Check ===' lines from the encode server."""
        log = self.encode_stdout.getvalue() + self.encode_stderr.getvalue()
        pattern = re.compile(
            r"Multi-Level Cache Check.*?"
            r"Local Hits:\s*(\d+).*?"
            r"Global Hits:\s*(\d+).*?"
            r"Misses.*?:\s*(\d+)"
        )
        return [(int(m[1]), int(m[2]), int(m[3])) for m in pattern.finditer(log)]

    def _chat(self, client, image_url):
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ],
            temperature=0,
            max_tokens=128,
        )
        text = response.choices[0].message.content
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)
        return text

    def test_image_cache_hit(self):
        client = self._client()
        baseline = len(self._parse_cache_log())

        # 1) cold request: the image embedding must be computed (miss).
        cold_text = self._chat(client, self.image_url)
        time.sleep(1)
        # 2) warm request: the same image must be served from the global mm cache.
        warm_text = self._chat(client, self.image_url)
        time.sleep(1)
        # 3) isolation: a different image must NOT hit the cached key (miss again).
        self._chat(client, self.other_image_url)
        time.sleep(1)

        entries = self._parse_cache_log()[baseline:]
        print(f"[NPU mm-global-cache] cache log entries: {entries}")
        self.assertEqual(len(entries), 3, "Expected one cache check per request")

        # Cold: nothing cached yet -> local=0, global=0, miss>0.
        first_local, first_global, first_miss = entries[0]
        self.assertEqual(
            first_local, 0, f"First request must not hit local cache: {entries[0]}"
        )
        self.assertEqual(
            first_global, 0, f"First request must not hit global cache: {entries[0]}"
        )
        self.assertGreater(
            first_miss,
            0,
            f"First request should be a cold miss (embedding computed): {entries[0]}",
        )

        # Warm: served from cache with zero GPU recompute.
        second_local, second_global, second_miss = entries[1]
        self.assertGreater(
            second_local + second_global,
            0,
            f"Second request must hit the mm cache: {entries[1]}",
        )
        self.assertEqual(
            second_miss,
            0,
            f"Second request must have 0 GPU-work misses: {entries[1]}",
        )

        # Reuse correctness proxy: greedy (temperature=0) cold vs warm must be
        # semantically consistent about the same image. Byte-equality is NOT
        # asserted on purpose: VLM decoding is not guaranteed deterministic, so
        # exact string match would conflate "cache reused" with "backend stable".
        # Requiring overlapping content words guards against a stale/corrupt
        # cached embedding (which would describe something unrelated) while
        # tolerating benign rephrasing.
        def _content_words(text: str) -> set:
            stopwords = {
                "a", "an", "the", "and", "or", "of", "in", "on", "to", "for",
                "with", "this", "that", "is", "are", "was", "were", "be", "by",
                "at", "it", "its", "from",
            }
            words = set()
            for w in text.lower().split():
                w = w.strip(".,;:!?()\"'[]-")
                if w and w not in stopwords and len(w) > 2:
                    words.add(w)
            return words

        shared = _content_words(cold_text) & _content_words(warm_text)
        self.assertTrue(
            shared,
            "Cold and warm requests should describe the same image subject: "
            f"cold={cold_text!r} warm={warm_text!r}",
        )

        # Isolation: the different image was never cached -> cold miss.
        third_local, third_global, third_miss = entries[2]
        self.assertEqual(
            third_local, 0, f"Different image must not hit local cache: {entries[2]}"
        )
        self.assertEqual(
            third_global, 0, f"Different image must not hit global cache: {entries[2]}"
        )
        self.assertGreater(
            third_miss,
            0,
            f"Different image should be a distinct-cache miss: {entries[2]}",
        )


if __name__ == "__main__":
    unittest.main()
