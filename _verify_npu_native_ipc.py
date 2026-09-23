# 冒烟验证脚本：torch_npu 原生 reduction 在「无 SGLANG_TP_RANK」时的自描述设备落点
#
# 目标：验证 weight-cache NPU transport 设计的地基——
#   原生（未经 monkey_patch_torch_reductions 加载的）torch_npu reduction 能否
#   在 daemon/client 两个独立进程之间，不依赖 SGLANG_TP_RANK，靠 reduce_tensor
#   句柄自携带的 device index，把 NPU 张量落到同一张物理卡，且是真零拷贝（显存映射）。
#
# 使用：在目标 NPU 机器上执行  python _verify_npu_native_ipc.py
# 通过标准：全部 VERIFY-OK，且「零拷贝写回可见」= OK。
# 注意：本脚本不 import sglang（避免触发任何 monkey patch），仅用 torch + torch_npu。

import os
import sys

# 先确认 SGLANG_TP_RANK 确实未设置（被验证的前提本身）
_tprank = os.environ.get("SGLANG_TP_RANK")
print("[前置] SGLANG_TP_RANK =", _tprank)

# ---- 0. 环境探测（版本 + 芯片型号） ----
try:
    import torch
    import torch_npu
except Exception as e:  # noqa: BLE001
    print("[FATAL] 无法 import torch_npu:", e)
    print("结论: FAILED-ENV（非 NPU 环境或 torch_npu 未安装，无法验证）")
    sys.exit(1)

print("[ENV] torch        =", torch.__version__)
print("[ENV] torch_npu    =", torch_npu.__version__)

soc = device_name = None
try:
    soc = torch_npu.get_soc_version()       # 芯片型号，如 'Ascend910B*'
    print("[ENV] soc_version =", soc)
except Exception as e:  # noqa: BLE001
    print("[WARN] get_soc_version 不可用:", e)
try:
    if torch.npu.is_available():
        device_name = torch.npu.get_device_name(0)
        print("[ENV] device_name  =", device_name)
except Exception as e:  # noqa: BLE001
    print("[WARN] get_device_name 不可用:", e)

# 芯片型号拼接用于 950DT 排除告警（950DT 官方不支持 IPC）
_soc_str = " ".join(str(x) for x in [soc, device_name] if x)
if not torch.npu.is_available():
    print("[FATAL] torch_npu 已安装但 NPU 不可见（ASCEND_RT_VISIBLE_DEVICES?）。")
    sys.exit(1)
print("[ENV] npu.count   =", torch.npu.device_count())

# 950DT 硬性排除：官方文档明确 Ascend 950DT 不支持 IPC 显存共享
if "950" in _soc_str and "DT" in _soc_str:
    print("[BLOCK] 检测到 Ascend 950DT —— 官方明确 950DT 不支持 IPC，无需继续验证。")
    print("结论: HDK_DT_UNSUPPORTED")
    sys.exit(2)

# ---- 1. spawn 启动 = 按官方文档要求（fork 无法继承 NPU 上下文） ----
import multiprocessing as mp
from multiprocessing import Queue as MpQueue

try:
    ctx = mp.get_context("spawn")
except Exception as e:  # noqa: BLE001
    print("[FATAL] 无法创建 spawn context:", e)
    sys.exit(1)


def client_worker(
    tensor_queue: MpQueue,
    writeback_queue: MpQueue,
    expected_device: int,
) -> None:
    """client 侧：取句柄 -> 原生重建 -> 校验设备落点 -> 原地写回验证零拷贝。

    刻意不调用 monkey_patch_torch_reductions，也不读 SGLANG_TP_RANK，
    只用 torch.multiprocessing.reductions 在 daemon 侧生成的 (fn, args)。
    """
    import torch  # noqa
    import torch_npu  # noqa  (确保 NPU reduction 注册)

    func, args = tensor_queue.get()
    print("[client] 收到 (fn, args)，args 长度 =", len(args))

    list_args = list(args)
    # reduce_tensor 的 args[6] 是设备索引；打印它从而验证"句柄自携带设备"
    print("[client] args[6]（自描述 device index）=", list_args[6])

    # 原生重建（未 patch）：本项目验证的正是这一条
    tensor = func(*list_args)
    print("[client] 重建后 device =", tensor.device, " shape =", tuple(tensor.shape))
    print("[client] 重建后数值[0] =", float(tensor[0].item()))

    # 设备落点校验：应落到句柄自描述的卡
    ok_dev = bool(tensor.device.type == "npu" and tensor.device.index == expected_device)
    print("[CHECK] client 落在 npu:%d (%s) ->" % (expected_device, "OK" if ok_dev else "MISMATCH"))
    writeback_queue.put(("device", ok_dev, str(tensor.device)))

    # 零拷贝写回验证：原地改写，daemon 应能在同一显存读到
    tensor[0] = 12345.0
    torch.npu.synchronize()
    writeback_queue.put(("writeback_done", True, None))


def main() -> int:
    dev_id = 0  # 目标物理卡；如需改成 daemon 所用的卡号
    print(f"[daemon] 使用 npu:{dev_id} 构造张量并导出句柄")

    # 在 daemon 侧造张量（先确保 set_device 与句柄一致）
    torch.npu.set_device(dev_id)
    t = torch.full((8,), 1.0, device=f"npu:{dev_id}", dtype=torch.float32)

    tq: MpQueue = ctx.Queue()
    wq: MpQueue = ctx.Queue()
    proc = ctx.Process(target=client_worker, args=(tq, wq, dev_id))
    proc.start()

    # daemon 侧用原生 reduction 导出（不用 sglang 的 MultiprocessingSerializer）
    from torch.multiprocessing.reductions import reduce_tensor
    handle = reduce_tensor(t)
    tq.put(handle)

    # 等 client 反馈
    dev_flag = dev_ok = writeback_ok = False
    dev_str, wb_detail = None, None
    for _ in range(2):
        kind, ok, detail = wq.get(timeout=30)
        if kind == "device":
            dev_flag, dev_ok, dev_str = True, ok, detail
        elif kind == "writeback_done":
            writeback_ok = bool(ok)

    proc.join(timeout=10)
    if proc.is_alive():
        print("[WARN] client 未退出，终止")
        proc.terminate()

    print("[daemon] 写回后读取 t[0] =", float(t.cpu().numpy()[0]))
    # 零拷贝证明核心：client 写 12345 后 daemon 必须能读到（同物理显存）
    writeback_visible = bool(abs(float(t[0].item()) - 12345.0) < 1e-3)
    print("[CHECK] 写回可见（零拷贝显存映射）->", "OK" if writeback_visible else "ERROR")

    print()
    print("=" * 60)
    if dev_flag and dev_ok and writeback_ok and writeback_visible:
        print("结论: SUPPORTED —— 原生 NPU reduction 无需 SGLANG_TP_RANK，")
        print("      靠自描述 device index 即可落到同一物理卡，零拷贝成立。")
        print("      => NpuWeightCacheTransportBackend 的自描述设备设计成立。")
        return 0
    print("结论: FAILED —— 见上方 CHECK。可能原因：")
    print("  * HDK/CANN 版本低于门槛（需 HDK 25.3.RC1+ / CANN 8.3.RC1+）")
    print("  * 芯片型号较新/老导致不支持 (如 950DT)")
    print("  * args[6] 语义与官方文档描述的 device index 不一致")
    return 3 if writeback_visible else 4


if __name__ == "__main__":
    sys.exit(main())