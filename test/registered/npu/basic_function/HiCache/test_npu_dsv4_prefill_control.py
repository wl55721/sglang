"""诊断脚本：判定 DeepSeek-V4-Flash NPU prefill 设备挂起与 external-linker 的关系。

背景
----
``test_npu_unified_cache_external_linker_dsv4.py`` 在当前 NPU 上 watchdog 超时。
py-spy 显示调度 MainThread 停在 prefill 的 D2H 同步点（`.item()` / `.cpu()`）上，
说明是**主计算流设备级 wedge**，而非元数据逻辑问题。需要判别该 wedge 是否由
``--enable-unified-cache-external-linker``(npu_memcache) 的加载 DMA 引入。

本脚本复用该测试同款启动参数，提供一个**对照组**（默认不带 linker），
并与 linker 版（``--with-linker``）做对比：

- 实验1：默认，**去掉** linker 两个标志，发一个普通 prefill。
  - 正常 → 卡死是 linker 加载路径引入，去查 ``batch_get_into_layers`` / ``_load_one_layer``。
  - 仍卡 → 与 linker 无关，是 DSV4 prefill 在 NPU 上的设备级问题。
- 实验2：``--with-linker``（需外部已配好 ``SGLANG_HICACHE_MEMCACHE_CONFIG_PATH``
  及 MetaService），可叠加 ``--layerwise 0`` 隔离逐层加载。

用法示例
--------
  # 对照组（无 linker）
  python -m test.registered.npu.basic_function.HiCache.test_npu_dsv4_prefill_control

  # linker 版 + 关闭逐层加载
  SGLANG_HICACHE_MEMCACHE_CONFIG_PATH=/path/to/config.json \\
    python -m test.registered.npu.basic_function.HiCache.test_npu_dsv4_prefill_control \\
      --with-linker --layerwise 0

对每组参数，若无响应会在 ``--req-timeout`` 后判为 HANG，并打印 py-spy 采集命令，
便于确认 MainThread 现在的停靠位置。
"""

import argparse
import logging
import sys
import time

import requests

from sglang.srt.environ import envs
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.test_utils import (
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger(__name__)


# DeepSeek-V4-Flash W8A8 (modelslim) 在 NPU 上必需的 env，与测试用例保持一致。
DEEPSEEK_V4_FLASH_W8A8_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
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
}

# 与测试 setUpClass 同一套启动参数；据此去掉/加上 linker 标志。
def _build_server_args(with_linker: bool, layerwise: str) -> list[str]:
    args = [
        "--trust-remote-code",
        "--device", "npu",
        "--tp-size", "16",
        "--attention-backend", "dsv4",
        "--disable-cuda-graph",
        "--quantization", "modelslim",
        "--page-size", "256",
        "--chunked-prefill-size", "8192",
        "--mem-fraction-static", "0.62",
        "--disable-shared-experts-fusion",
        "--swa-full-tokens-ratio", "0.25",
        "--max-total-tokens", "8192",
        "--max-running-requests", "1",
        "--enable-cache-report",
    ]
    if with_linker:
        args += [
            "--enable-unified-cache-external-linker",
            "--unified-cache-external-linker-backend", "npu_memcache",
        ]
    return args


def _wait_for_generate(base_url: str, text: str, max_new_tokens: int,
                       timeout: float):
    """发一个普通 prefill 请求并等待响应；超时即判为 HANG。"""
    deadline = time.monotonic() + timeout
    resp = None
    while time.monotonic() < deadline:
        try:
            r = requests.post(
                f"{base_url}/generate", json={
                    "text": text,
                    "max_new_tokens": max_new_tokens,
                },
                timeout=max(1, min(5.0, deadline - time.monotonic())),
            )
            resp = r
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return resp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=find_available_port(30000))
    ap.add_argument("--with-linker", action="store_true",
                    help="带上 external-linker 两个启动标志（实验2，需外部配好 "
                         "SGLANG_HICACHE_MEMCACHE_CONFIG_PATH 与 MetaService）")
    ap.add_argument("--layerwise", default="1", choices=["0", "1"],
                    help="仅 --with-linker 时生效：SGLANG_NPU_MEMCACHE_LINKER_LAYERWISE")
    ap.add_argument("--launch-timeout", type=float, default=1800,
                    help="server 启动/加载模型超时")
    ap.add_argument("--req-timeout", type=float, default=300,
                    help="单个 prefill 请求的等待超时；超时判为 HANG")
    ap.add_argument("--text", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    mode = "with-external-linker(layerwise=%s)" % args.layerwise if args.with_linker else "control(no linker)"

    if args.with_linker and not envs.SGLANG_HICACHE_MEMCACHE_CONFIG_PATH.get():
        logger.error("--with-linker 需要 SGLANG_HICACHE_MEMCACHE_CONFIG_PATH 指向有效 "
                     "MemCache JSON，并已启动对应 MetaService。未设置，中止。")
        return 2

    server_args = _build_server_args(args.with_linker, args.layerwise)
    server_env = dict(DEEPSEEK_V4_FLASH_W8A8_ENVS)
    if args.with_linker:
        # 逐层加载开关是 server 侧 os.getenv 读取的 env，非 CLI 参数。
        server_env["SGLANG_NPU_MEMCACHE_LINKER_LAYERWISE"] = args.layerwise
    logger.info("launching server (mode=%s) model=%s", mode,
                DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH)
    logger.info("server args: %s", " ".join(server_args))

    proc = None
    try:
        # 阶段 1：拉起到 healthy。首次 forward 前返回；真正 wedge 发生在首个请求。
        logger.info("[phase1] waiting for server healthy (timeout=%.0fs)...",
                    args.launch_timeout)
        proc = popen_launch_server(
            DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
            base_url,
            timeout=args.launch_timeout,
            other_args=server_args,
            env=server_env,
            device="npu",
        )
        logger.info("[phase1] server healthy. url=%s", base_url)

        # 阶段 2：首个 prefill 请求。
        logger.info("[phase2] sending prefill request (timeout=%.0fs)...",
                    args.req_timeout)
        t0 = time.monotonic()
        resp = _wait_for_generate(base_url, args.text, args.max_new_tokens,
                                  args.req_timeout)
        elapsed = time.monotonic() - t0

        if resp is not None and resp.status_code == 200:
            logger.info("[phase2] SUCCESS in %.1fs -> %s", elapsed,
                        resp.json().get("text", "")[:80])
        else:
            status = resp.status_code if resp is not None else "NO_RESPONSE"
            logger.error("[phase2] FAILURE/HANG after %.1fs (status=%s). "
                         "This indicates a device-level wedge.", elapsed, status)
            _print_pyspy_hint(proc)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("error during run: %s", exc)

    # 到这说明异常退出；附 py-spy 提示以便定位卡点。
    _print_pyspy_hint(proc)
    return 1


def _print_pyspy_hint(proc) -> None:
    # py-spy 采集调度进程栈，确认 MainThread 当前停在哪一步。
    pids = []
    if proc is not None and proc.pid:
        pids.append(str(proc.pid))
    logger.info(
        "=== 若主机已卡住，请用 py-spy 抓调度进程栈确认 MainThread 停靠点：===\n"
        "  pids=$(pgrep -f 'sglang.srt.launch_server' | tr '\\n' ' ')\n"
        "  for p in $pids; do py-spy dump --native --pid $p; done\n"
        "  (调度进程是其中带 scheduler 的那个)"
    )


if __name__ == "__main__":
    sys.exit(main())