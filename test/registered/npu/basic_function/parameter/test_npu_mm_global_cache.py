import base64
import io
import os
import re
import time
import unittest
from urllib.parse import urlparse

import openai

from sglang.test.ascend.test_ascend_utils import (
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

    The global embed cache relies on a deployed Ascend MemCache instance, whose
    connection settings are provided via the SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH
    environment variable. The cache-hit test is skipped when that config is absent.

    [Test Category] Functional
    [Test Target] --enable-mm-global-cache / --mm-global-cache-backend=npu_memcache
    """

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

        cls.cache_cfg_path = os.environ.get(
            "SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH"
        )

        cls.memcache_env = {**os.environ, "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667"}
        if cls.cache_cfg_path:
            cls.memcache_env["SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH"] = (
                cls.cache_cfg_path
            )

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
        if cls.cache_cfg_path:
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

    def test_image_cache_hit(self):
        if not self.cache_cfg_path:
            self.skipTest(
                "mm-global-cache not enabled: SGLANG_MM_GLOBAL_CACHE_MEMCACHE_CONFIG_PATH not set"
            )
        client = self._client()
        baseline = len(self._parse_cache_log())
        for _ in range(2):
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": self.image_url}},
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
            time.sleep(1)

        entries = self._parse_cache_log()[baseline:]
        print(f"[NPU mm-global-cache] cache log entries: {entries}")
        self.assertGreaterEqual(
            len(entries), 2, "Expected at least 2 cache-check log entries"
        )
        local_hits, global_hits, _ = entries[-1]
        self.assertGreater(
            local_hits + global_hits,
            0,
            f"Second image request should hit the global mm cache: {entries[-1]}",
        )


if __name__ == "__main__":
    unittest.main()