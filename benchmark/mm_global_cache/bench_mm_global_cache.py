# SPDX-License-Identifier: Apache-2.0
"""mm-global-cache（Mooncake 后端）A/B 性能对比 benchmark。

用途
----
在 EPD 拓扑（encoder + prefill + decode + LB）上比较全局多模态 embedding 缓存的
性能收益，量化三组场景：

- A baseline：不开缓存，同一批图片反复请求 -> 每次都是 miss（全部重算 ViT）。
- B 高命中：开缓存（``--enable-mm-global-cache --mm-global-cache-backend mooncake``），
  同一批图片反复请求 -> 初次 miss 之后全部命中。
- C 零命中：开缓存、全部互异图片 -> 每次 miss，仅隔离出缓存带来的固定开销。

收益口径
--------
- A vs B = 净收益，收益公式 ``(miss_TTFT - hit_TTFT) / miss_TTFT``。
- A vs C = 固定成本（开缓存但从不命中的额外开销）。

指标
----
- TTFT：流式请求 + ``time.perf_counter()`` 取首个 content chunk 的时间，报告 P50/P90。
- 吞吐：并发 N 压满，报告 req/s 与 token/s。
- 每组先做 warmup（丢弃前几轮计时再统计）。

变量控制
--------
同模型、同 EPD 拓扑、同并发、同 max_tokens、``temperature=0``、保留
``--disable-cuda-graph``。A 组不加缓存 flag 且不起 mooncake 服务。

运行方式
--------
    python3 bench_mm_global_cache.py --group all --concurrency 8 --num-requests 40
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

import openai
import requests

from sglang.test.ascend.test_ascend_utils import (
    IMAGES_023_PATH,
    IMAGES_LOGO_PATH,
    KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_pd_server,
    popen_launch_server,
    popen_with_error_check,
    terminate_and_kill_process_tree,
)

DEFAULT_PROMPT = "What is shown in this image?"


# ---------------------------------------------------------------------------
# 环境与设备探测
# ---------------------------------------------------------------------------
def _strip_proxy_env() -> None:
    """去掉 http(s)/all proxy 环境变量，避免本地回环请求被代理拦截。

    NPU 场景下 127.0.0.1 的 loopback 请求若被 http_proxy 等变量劫持会直接失败；
    这里在脚本进程内清除，同时影响后续子进程 env（``os.environ.copy()``）与
    openai/httpx（默认读取 os.environ）。
    """
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    ):
        os.environ.pop(key, None)


def _detect_device() -> str:
    """探测当前计算设备（npu / cuda / cpu），用于分支 EPD 启动参数。"""
    try:
        import torch  # noqa: F401
    except Exception:
        return "cpu"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 图片辅助
# ---------------------------------------------------------------------------
def _data_url_from_image(image_path: str, mime: str = "image/png") -> str:
    """把本地图片编码成 OpenAI 客户端可直接内联的 data URL。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _synthetic_image_data_urls(n: int, reference_path: str) -> List[str]:
    """生成 n 张与参考图同分辨率的互异合成图 data URL（用于 C 组零命中）。

    关键：必须与 A/B 组使用同一分辨率（保证相同的动态 tiling / ViT 计算量），
    仅用不同 seed 的随机噪声保证内容哈希两两不同，从而每次请求都 miss。
    早期用 64x64 纯色图会因分辨率/复杂度差异导致编码成本不可比，已弃用。
    """
    from PIL import Image
    import random as _random

    with Image.open(reference_path) as ref:
        size = ref.size  # (宽, 高)

    urls: List[str] = []
    for i in range(n):
        rng = _random.Random(i)
        raw = bytes(rng.getrandbits(8) for _ in range(size[0] * size[1] * 3))
        img = Image.frombytes("RGB", size, raw)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:image/png;base64,{b64}")
    return urls


def _cycle_urls(pool: List[str], n: int) -> List[str]:
    """把固定图片池循环扩展到 n 张（用于 A/B 组的“同批图反复请求”）。"""
    if not pool:
        raise ValueError("图片池为空，请通过 --image-paths 提供至少一张真实图片")
    return [pool[i % len(pool)] for i in range(n)]


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _percentile(sorted_vals: List[float], p: float) -> float:
    """线性插值百分位（输入需已排序）。"""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


@dataclass
class GroupStats:
    """一组 benchmark 的汇总指标。"""

    group: str
    num_requests: int = 0
    # 顺序（并发 1）测量
    seq_ttft_p50_s: float = 0.0
    seq_ttft_p90_s: float = 0.0
    seq_ttft_mean_s: float = 0.0
    seq_req_per_s: float = 0.0
    seq_tok_per_s: float = 0.0
    # 并发 N 测量
    conc_ttft_p50_s: float = 0.0
    conc_ttft_p90_s: float = 0.0
    conc_ttft_mean_s: float = 0.0
    conc_req_per_s: float = 0.0
    conc_tok_per_s: float = 0.0
    # 用于收益计算（顺序 TTFT，秒）
    _seq_ttfts: List[float] = field(default_factory=list)


def _build_stats(
    ttfts: List[float], latencies: List[float], total_tokens: int, wall_s: float
) -> dict:
    sorted_ttfts = sorted(ttfts)
    n = len(ttfts)
    return {
        "ttft_p50_s": _percentile(sorted_ttfts, 50),
        "ttft_p90_s": _percentile(sorted_ttfts, 90),
        "ttft_mean_s": sum(ttfts) / n if n else 0.0,
        "req_per_s": n / wall_s if wall_s > 0 else 0.0,
        "tok_per_s": total_tokens / wall_s if wall_s > 0 else 0.0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# 请求客户端
# ---------------------------------------------------------------------------
def _make_messages(image_url: str, prompt: str) -> List[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _sync_request(client: "openai.Client", image_url: str, prompt: str, max_tokens: int):
    """同步流式请求一次，返回 (ttft_s, 总延迟_s, 输出 token 数)。"""
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model="default",
        messages=_make_messages(image_url, prompt),
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
    )
    ttft: Optional[float] = None
    out_tokens = 0
    for chunk in stream:
        choices = chunk.choices
        if choices and choices[0].delta and choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            out_tokens += 1
    total = time.perf_counter() - t0
    if ttft is None:
        # 极端情况下没有任何 content chunk，用总延迟兜底。
        ttft = total
    return ttft, total, out_tokens


async def _async_request(
    client: "openai.AsyncClient", image_url: str, prompt: str, max_tokens: int
):
    """异步流式请求一次，返回 (ttft_s, 总延迟_s, 输出 token 数)。"""
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model="default",
        messages=_make_messages(image_url, prompt),
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
    )
    ttft: Optional[float] = None
    out_tokens = 0
    async for chunk in stream:
        choices = chunk.choices
        if choices and choices[0].delta and choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            out_tokens += 1
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    return ttft, total, out_tokens


async def _run_concurrent(
    client: "openai.AsyncClient",
    image_urls: List[str],
    prompt: str,
    max_tokens: int,
    concurrency: int,
):
    """用信号量限制并发、一次性压满 ``image_urls``，返回 (结果列表, 墙钟秒)。"""
    sem = asyncio.Semaphore(concurrency)

    async def worker(url: str):
        async with sem:
            return await _async_request(client, url, prompt, max_tokens)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(worker(u) for u in image_urls))
    wall = time.perf_counter() - t0
    return results, wall


# ---------------------------------------------------------------------------
# EPD + Mooncake 拓扑编排
# ---------------------------------------------------------------------------
class Topology:
    """一次 benchmark 组的 encoder + prefill + decode + LB +（可选）mooncake 生命周期。"""

    def __init__(self, args: argparse.Namespace, enable_cache: bool, device: str):
        self.args = args
        self.enable_cache = enable_cache
        self.device = device

        self.base_host = args.host
        self.base_port = args.port
        self.lb_port = args.port
        self.encode_port = args.port + 300
        self.prefill_port = args.port + 100
        self.decode_port = args.port + 200
        self.bootstrap_port = args.port + 500
        self.encode_url = f"http://{self.base_host}:{self.encode_port}"
        self.prefill_url = f"http://{self.base_host}:{self.prefill_port}"
        self.decode_url = f"http://{self.base_host}:{self.decode_port}"
        self.lb_url = f"http://{self.base_host}:{self.lb_port}"

        self.model = args.model
        self.api_key = "sk-123456"

        # 进程句柄
        self.process_lb = None
        self.process_decode = None
        self.process_prefill = None
        self.process_encode = None
        self.master_service_process = None
        self.metadata_service_process = None
        self._mooncake_logs: dict = {}
        self.encode_stdout = io.StringIO()
        self.encode_stderr = io.StringIO()

    # -- mooncake -----------------------------------------------------------
    def _start_mooncake_services(self) -> None:
        self.mooncake_master_port = find_available_port(50051)
        self.mooncake_metrics_port = find_available_port(9003)
        self.mooncake_metadata_port = find_available_port(8080)
        self._mooncake_logs = {}

        # mooncake >= 0.3.13 拒绝在无 auth token 时绑定 0.0.0.0（CVE-2026-103765）。
        # 本 benchmark 无外部 metadata 暴露，仅环回访问，保留旧版开放行为。
        self.metadata_service_process = self._launch_mooncake_service(
            "metadata",
            [
                "python3",
                "-m",
                "mooncake.http_metadata_server",
                "--port",
                str(self.mooncake_metadata_port),
            ],
            extra_env={"MC_METADATA_HTTP_ALLOW_INSECURE": "1"},
        )
        self.master_service_process = self._launch_mooncake_service(
            "master",
            [
                "mooncake_master",
                "--port",
                str(self.mooncake_master_port),
                "--metrics_port",
                str(self.mooncake_metrics_port),
            ],
        )
        self._wait_for_mooncake_services_ready()

    def _launch_mooncake_service(
        self, name: str, command: List[str], extra_env: Optional[dict] = None
    ):
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        log = tempfile.NamedTemporaryFile(
            prefix=f"mooncake-{name}-", suffix=".log", delete=False
        )
        self._mooncake_logs[name] = log.name
        with log:
            return subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env
            )

    def _mooncake_log_tail(self, name: str) -> str:
        path = self._mooncake_logs.get(name)
        if not path or not os.path.exists(path):
            return "(no log)"
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-30:])

    def _wait_for_mooncake_services_ready(self, timeout: int = 30) -> None:
        deadline = time.time() + timeout
        metadata_url = f"http://127.0.0.1:{self.mooncake_metadata_port}/metadata"
        print(f"等待 Mooncake 服务就绪：{metadata_url} ...")
        while time.time() < deadline:
            for name, proc in (
                ("metadata", self.metadata_service_process),
                ("master", self.master_service_process),
            ):
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"mooncake_{name} 提前退出 (code={proc.returncode}); "
                        f"日志尾部:\n{self._mooncake_log_tail(name)}"
                    )
            try:
                # GET /metadata 缺 key 时返回 400；任何 HTTP 响应都说明 metadata
                # HTTP server 已在监听，正是我们要探测的目标。
                requests.get(metadata_url, timeout=2)
                print("Mooncake 服务已就绪")
                return
            except requests.RequestException as e:
                print(f"metadata server 尚未可达：{e}")
            time.sleep(2)
        raise RuntimeError(
            f"Mooncake 服务 {timeout}s 内未就绪；"
            f"metadata 日志尾部:\n{self._mooncake_log_tail('metadata')}"
            f"\nmaster 日志尾部:\n{self._mooncake_log_tail('master')}"
        )

    # -- EPD 节点 -----------------------------------------------------------
    def _build_env(self) -> dict:
        env = dict(os.environ)
        if self.enable_cache:
            env.update(
                {
                    "MOONCAKE_MASTER": f"127.0.0.1:{self.mooncake_master_port}",
                    "MOONCAKE_PROTOCOL": "tcp",
                    "MC_MS_AUTO_DISC": "0",
                    "MOONCAKE_DEVICE": "",
                    "MOONCAKE_TE_META_DATA_SERVER": (
                        f"http://127.0.0.1:{self.mooncake_metadata_port}/metadata"
                    ),
                    "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",
                }
            )
        if self.device == "npu":
            # AscendTransferEngine（prefill/decode 的 PD KV 传输）读取该变量。
            env["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24667"
        return env

    def start(self) -> None:
        if self.enable_cache:
            self._start_mooncake_services()
        env = self._build_env()

        self._start_encode(env)
        self._start_prefill(env)
        self._start_decode(env)
        self._wait_server_ready(self.encode_url + "/health", self.process_encode)
        self._wait_server_ready(self.prefill_url + "/health", self.process_prefill)
        self._wait_server_ready(self.decode_url + "/health", self.process_decode)
        self._launch_router(env)
        self._wait_server_ready(self.lb_url + "/health", self.process_lb)
        time.sleep(5)

    def _start_encode(self, env: dict) -> None:
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
            str(self.encode_port),
            "--disable-cuda-graph",
        ]
        if self.enable_cache:
            encode_args += [
                "--enable-mm-global-cache",
                "--mm-global-cache-backend",
                "mooncake",
            ]
        self.process_encode = popen_launch_server(
            self.model,
            base_url=self.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=env,
            return_stdout_stderr=(self.encode_stdout, self.encode_stderr),
        )

    def _start_prefill(self, env: dict) -> None:
        prefill_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            self.encode_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            str(self.bootstrap_port),
            "--base-gpu-id",
            "1",
            "--tp-size",
            "1",
            "--port",
            str(self.prefill_port),
            "--disable-cuda-graph",
        ]
        if self.device == "npu":
            prefill_args[1:1] = ["--attention-backend", "ascend"]
            prefill_args += ["--disaggregation-transfer-backend", "ascend"]
        self.process_prefill = popen_launch_pd_server(
            self.model,
            self.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )

    def _start_decode(self, env: dict) -> None:
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            str(self.bootstrap_port),
            "--base-gpu-id",
            "2",
            "--tp-size",
            "1",
            "--port",
            str(self.decode_port),
            "--disable-cuda-graph",
        ]
        if self.device == "npu":
            decode_args[1:1] = ["--attention-backend", "ascend"]
            decode_args += ["--disaggregation-transfer-backend", "ascend"]
        self.process_decode = popen_launch_pd_server(
            self.model,
            self.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )

    def _launch_router(self, env: dict) -> None:
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            self.prefill_url,
            "--decode",
            self.decode_url,
            "--host",
            self.base_host,
            "--port",
            str(self.lb_port),
        ]
        print("启动 LB：", " ".join(lb_command))
        self.process_lb = popen_with_error_check(lb_command)

    @staticmethod
    def _wait_server_ready(url: str, process: Optional[subprocess.Popen]) -> None:
        from sglang.utils import wait_for_http_ready

        wait_for_http_ready(
            url=url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, process=process
        )
        print(f"服务 {url} 已就绪")

    def stop(self) -> None:
        for proc in (
            self.process_lb,
            self.process_decode,
            self.process_prefill,
            self.process_encode,
        ):
            if proc:
                try:
                    terminate_and_kill_process_tree(proc)
                except Exception as e:  # noqa: BLE001
                    print(f"清理进程 {proc.pid} 失败：{e}")

        for proc in (self.master_service_process, self.metadata_service_process):
            if proc and proc.poll() is None:
                try:
                    terminate_and_kill_process_tree(proc)
                except Exception as e:  # noqa: BLE001
                    print(f"清理 Mooncake 服务 {proc.pid} 失败：{e}")

    def make_client(self) -> "openai.Client":
        return openai.Client(api_key=self.api_key, base_url=f"{self.lb_url}/v1")

    def make_async_client(self) -> "openai.AsyncClient":
        return openai.AsyncClient(api_key=self.api_key, base_url=f"{self.lb_url}/v1")


# ---------------------------------------------------------------------------
# 单组 benchmark 流程
# ---------------------------------------------------------------------------
def run_group(topology: Topology, group: str, args: argparse.Namespace) -> GroupStats:
    client = topology.make_client()
    aclient = topology.make_async_client()

    total_requests = args.num_requests + args.warmup
    if group in ("A", "B"):
        all_urls = _cycle_urls(args.real_image_urls, total_requests)
    else:  # C：全部互异，但与 A/B 同分辨率，保证编码成本可比
        all_urls = _synthetic_image_data_urls(
            total_requests, args.real_image_paths[0]
        )

    warmup_urls = all_urls[: args.warmup]
    measure_urls = all_urls[args.warmup :]

    # warmup（丢弃计时）
    for u in warmup_urls:
        _sync_request(client, u, args.prompt, args.max_tokens)

    # 顺序（并发 1）测量：TTFT + 并发 1 吞吐
    seq_ttfts: List[float] = []
    seq_latencies: List[float] = []
    seq_tokens = 0
    seq_t0 = time.perf_counter()
    for u in measure_urls:
        ttft, total, tok = _sync_request(client, u, args.prompt, args.max_tokens)
        seq_ttfts.append(ttft)
        seq_latencies.append(total)
        seq_tokens += tok
    seq_wall = time.perf_counter() - seq_t0

    seq_stats = _build_stats(seq_ttfts, seq_latencies, seq_tokens, seq_wall)

    # 并发 N 测量：吞吐 + 负载下的 TTFT
    results, conc_wall = asyncio.run(
        _run_concurrent(
            aclient, measure_urls, args.prompt, args.max_tokens, args.concurrency
        )
    )
    conc_ttfts: List[float] = []
    conc_tokens = 0
    for ttft, total, tok in results:
        conc_ttfts.append(ttft)
        conc_tokens += tok
    conc_stats = _build_stats(
        conc_ttfts, [r[1] for r in results], conc_tokens, conc_wall
    )

    return GroupStats(
        group=group,
        num_requests=len(measure_urls),
        seq_ttft_p50_s=seq_stats["ttft_p50_s"],
        seq_ttft_p90_s=seq_stats["ttft_p90_s"],
        seq_ttft_mean_s=seq_stats["ttft_mean_s"],
        seq_req_per_s=seq_stats["req_per_s"],
        seq_tok_per_s=seq_stats["tok_per_s"],
        conc_ttft_p50_s=conc_stats["ttft_p50_s"],
        conc_ttft_p90_s=conc_stats["ttft_p90_s"],
        conc_ttft_mean_s=conc_stats["ttft_mean_s"],
        conc_req_per_s=conc_stats["req_per_s"],
        conc_tok_per_s=conc_stats["tok_per_s"],
        _seq_ttfts=seq_ttfts,
    )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _ms(s: float) -> str:
    return f"{s * 1000:8.1f}"


def _print_results(results: List[GroupStats]) -> None:
    header = (
        f"{'组':>2} | {'顺序TTFT P50':>12} | {'顺序TTFT P90':>12} | "
        f"{'顺序 req/s':>10} | {'顺序 tok/s':>10} | "
        f"{'并发TTFT P50':>12} | {'并发TTFT P90':>12} | "
        f"{'并发 req/s':>10} | {'并发 tok/s':>10}"
    )
    print("\n========== mm-global-cache A/B 性能对比 ==========")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.group:>2} | "
            f"{_ms(r.seq_ttft_p50_s)} | {_ms(r.seq_ttft_p90_s)} | "
            f"{r.seq_req_per_s:10.3f} | {r.seq_tok_per_s:10.3f} | "
            f"{_ms(r.conc_ttft_p50_s)} | {_ms(r.conc_ttft_p90_s)} | "
            f"{r.conc_req_per_s:10.3f} | {r.conc_tok_per_s:10.3f}"
        )

    by_group = {r.group: r for r in results}
    if "A" in by_group and "B" in by_group:
        a = by_group["A"]
        b = by_group["B"]
        ttft_gain = (
            (a.seq_ttft_p50_s - b.seq_ttft_p50_s) / a.seq_ttft_p50_s
            if a.seq_ttft_p50_s > 0
            else 0.0
        )
        print("\n净收益（A vs B，顺序 TTFT P50）：")
        print(f"  miss_TTFT = {_ms(a.seq_ttft_p50_s)} ms")
        print(f"  hit_TTFT  = {_ms(b.seq_ttft_p50_s)} ms")
        print(f"  收益 = (miss - hit) / miss = {ttft_gain * 100:.1f}%")

    if "A" in by_group and "C" in by_group:
        c = by_group["C"]
        # 固定成本 = C（零命中）相对 A（baseline）的 TTFT 额外开销。
        overhead = c.seq_ttft_p50_s - by_group["A"].seq_ttft_p50_s
        print("\n固定成本（A vs C，顺序 TTFT P50）：")
        print(f"  C 零命中 TTFT - A baseline TTFT = {_ms(overhead)} ms")
    print("=================================================\n")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="mm-global-cache（Mooncake 后端）A/B 性能对比 benchmark"
    )
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["A", "B", "C", "all"],
        help="跑哪一组：A=baseline无缓存 / B=高命中 / C=零命中 / all=全部",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="吞吐压测的并发数（并发 N）",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=40,
        help="计入统计的有效请求数（不含 warmup）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="每组丢弃计时的 warmup 请求数",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="每个请求的最大输出 token 数",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
        help="模型权重路径",
    )
    parser.add_argument(
        "--image-paths",
        type=str,
        default=f"{IMAGES_LOGO_PATH},{IMAGES_023_PATH}",
        help="A/B 组复用的真实图片路径，逗号分隔",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="随图片一起发送的文本提示",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="LB 监听端口；缺省时自动寻找可用端口",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.port is None:
        args.port = find_available_port(30000)

    # 本地 loopback 场景下规避代理拦截。
    _strip_proxy_env()

    args.real_image_paths = [
        p.strip() for p in args.image_paths.split(",") if p.strip()
    ]
    args.real_image_urls = [_data_url_from_image(p) for p in args.real_image_paths]

    device = _detect_device()
    print(f"检测到设备：{device}")
    if device == "cpu":
        raise RuntimeError("未检测到可用计算设备（npu/cuda），无法进行 benchmark")

    groups = ["A", "B", "C"] if args.group == "all" else [args.group]
    results: List[GroupStats] = []

    for group in groups:
        print(f"\n===== 开始 {group} 组 benchmark =====")
        # A 组不开缓存、不启动 mooncake；B/C 开缓存并启动 mooncake。
        enable_cache = group in ("B", "C")
        topology = Topology(args, enable_cache=enable_cache, device=device)
        try:
            topology.start()
            stats = run_group(topology, group, args)
            results.append(stats)
        finally:
            topology.stop()

    _print_results(results)


if __name__ == "__main__":
    main()