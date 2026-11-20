"""TeaCache 风格的轻量步间输出缓存。

按相邻步输入的相对 L1 距离决定是否复用上一次 DiT/UNet 输出。
这是通用近似实现，不是某个第三方 TeaCache 仓库的逐行移植。

为什么有"分道"这回事
--------------------
批量化 CFG（SD1.5 / SDXL 等）每步**只调一次** UNet（`torch.cat([latents]*2)`,
batch=2），一条 lane 就够。但有些管线是 uncond / cond **两次前向** ——
那时单 lane 会把"cond(t)"和"uncond(t+1)"当成相邻步比较，距离必然很大，
于是命中率恒为 0（用户看到的就是"勾了加速但一点没快"）。

所以这里做成**自适应的**：先按单 lane 跑；如果连续若干步一次都没命中，
就自动改成按调用序号奇偶分两条 lane（奇/偶各自比较同角色的相邻步）。
分道只可能让命中变少、不会让保真变差 —— 它比较的是同角色、更远的步。
如果分道之后仍然 0 命中，下一张图会退回单 lane，避免一直背着它。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_ACTIVE = None

#: 连续算过这么多步还一次没命中，就怀疑是"每步两次前向"，改分道试试。
SPLIT_AFTER_COMPUTES = 6


def _sample_distance(previous, current) -> float:
    if previous is None or current is None or previous.shape != current.shape:
        return float("inf")
    left = previous.detach().reshape(-1)
    right = current.detach().reshape(-1)
    stride = max(1, left.numel() // 4096)
    left = left[::stride].float()
    right = right[::stride].float()
    denominator = left.abs().mean().clamp_min(1e-6)
    return float(((left - right).abs().mean() / denominator).item())


def _new_lane() -> dict:
    return {"prev_in": None, "prev_out": None, "acc": 0.0, "skips": 0}


class _StepCache:
    def __init__(self, model, threshold: float, max_skips: int):
        self.model = model
        self.threshold = max(0.01, float(threshold))
        self.max_skips = max(1, int(max_skips))
        #: 是否按调用序号奇偶分道（自动探测，见模块 docstring）
        self.split = False
        self._split_hits = 0
        self._calls = 0
        self._last_distance = None
        self._original = model.forward
        self._lanes = [_new_lane(), _new_lane()]
        self.reset()
        model.forward = self._forward

    def reset(self):
        # 分道是"学到的"：上一次分道一步都没命中，说明这个管线不是两次前向，
        # 这张图退回单 lane —— 不能让一个错误的判断一直拖累命中率。
        if self.split and self._split_hits == 0:
            self.split = False
            logger.info("步间缓存：上次分道后仍无命中，退回单 lane")
        self._split_hits = 0
        self._lanes = [_new_lane(), _new_lane()]
        self._calls = 0
        self._last_distance = None
        self.computed = 0
        self.reused = 0

    def restore(self):
        """把 forward 换回"我们包装之前"的那个实现。

        ⚠️ 这里曾经是 `del self.model.__dict__["forward"]`。它只在
        "没有别人比我们更早包装 forward" 时才等价于还原：一旦
        `utils/video_gen.py` 之类的模块先给同一个 unet 打了 forward 补丁、
        我们再叠一层，`del` 会把**别人的补丁一起删掉**，退回类上的原始
        forward —— 表现是关掉步间缓存后视频插帧/其它 forward 补丁悄悄失效。
        直接赋值回去才是正确的栈式还原。
        """
        try:
            self.model.forward = self._original
        except Exception as exc:
            logger.warning("步间缓存还原 forward 失败: %s", exc)
        # 缓存里攥着上一张图整份 UNet 输出（分辨率高时几十 MB），
        # 不清掉会一直占着显存。
        for lane in self._lanes:
            lane["prev_in"] = None
            lane["prev_out"] = None

    def _input_tensor(self, args, kwargs):
        import torch
        for key in ("hidden_states", "sample"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                return value
        return args[0] if args and isinstance(args[0], torch.Tensor) else None

    def _lane(self) -> dict:
        return self._lanes[(self._calls % 2) if self.split else 0]

    def _maybe_split(self):
        """连续算了若干步一次没命中 → 试试"每步两次前向"的分道模式。"""
        if self.split or self.reused or self.computed < SPLIT_AFTER_COMPUTES:
            return
        self.split = True
        self._lanes = [_new_lane(), _new_lane()]
        self._split_hits = 0
        logger.info(
            "步间缓存：连续 %d 步 0 命中，怀疑每步有 uncond/cond 两次前向 → "
            "改按奇偶分道缓存", self.computed)

    def _forward(self, *args, **kwargs):
        lane = self._lane()
        self._calls += 1
        current = self._input_tensor(args, kwargs)
        can_reuse = lane["prev_out"] is not None and current is not None
        if can_reuse:
            distance = _sample_distance(lane["prev_in"], current)
            self._last_distance = distance
            lane["acc"] += distance
            can_reuse = (lane["acc"] < self.threshold
                         and lane["skips"] < self.max_skips)
        if current is not None:
            lane["prev_in"] = current
        if can_reuse:
            lane["skips"] += 1
            self.reused += 1
            if self.split:
                self._split_hits += 1
            return lane["prev_out"]

        output = self._original(*args, **kwargs)
        lane["prev_out"] = output
        lane["acc"] = 0.0
        lane["skips"] = 0
        self.computed += 1
        self._maybe_split()
        return output


def configure_step_cache(pipe, arch_id: str, enabled: bool,
                         threshold: float = 0.18, max_skips: int = 2):
    """配置当前管线的缓存；禁用或换管线时恢复原 forward。

    支持范围：**任何带 `transformer` 或 `unet` 的管线**。以前这里是架构白
    名单（`flux → transformer`、`sdxl → unet`），其它架构打一行 INFO 就
    `return None` —— 于是 SD1.5 / SD2 / SD3 的用户在设置里勾了"步间缓存"，
    界面显示已启用，实际一步都没省（用户反馈的"几乎不省"）。
    现在按管线**有没有这个模块**判定，不再看架构名。
    """
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.restore()
        _ACTIVE = None
    if not enabled:
        return None
    attr = "transformer"
    model = getattr(pipe, "transformer", None)
    if model is None:
        attr = "unet"
        model = getattr(pipe, "unet", None)
    if model is None or not callable(getattr(model, "forward", None)):
        logger.info("步间缓存：管线（%s）没有可用的 transformer/unet，按原采样执行",
                    arch_id or "?")
        return None
    _ACTIVE = _StepCache(model, threshold, max_skips)
    logger.info("步间缓存已启用：架构=%s 模块=%s threshold=%.3f max_skips=%s",
                arch_id or "?", attr, _ACTIVE.threshold, _ACTIVE.max_skips)
    return _ACTIVE


def reset_step_cache():
    if _ACTIVE is not None:
        _ACTIVE.reset()


def step_cache_stats() -> dict:
    if _ACTIVE is None:
        return {"enabled": False, "computed": 0, "reused": 0,
                "calls": 0, "reuse_ratio": 0.0, "threshold": 0.0,
                "split": False, "last_distance": None}
    calls = _ACTIVE.computed + _ACTIVE.reused
    return {
        "enabled": True,
        "computed": _ACTIVE.computed,
        "reused": _ACTIVE.reused,
        "calls": calls,
        "reuse_ratio": (_ACTIVE.reused / calls) if calls else 0.0,
        "threshold": _ACTIVE.threshold,
        "consecutive_skips": _ACTIVE._lane()["skips"],
        "split": bool(_ACTIVE.split),
        "last_distance": _ACTIVE._last_distance,
    }


def step_cache_summary() -> str:
    """一行中文结论：这次到底省没省。

    为什么专门做这个：`step_cache_stats()["enabled"] is True` 只说明"配置
    装上了"，跟"真的跳过了步"无关。用户勾了加速看到 `enabled: true`
    却感觉没变快，就会怀疑功能坏了。说清楚 computed/reused 才诚实。
    """
    st = step_cache_stats()
    if not st["enabled"]:
        return "步间缓存：未启用"
    calls, reused = st["calls"], st["reused"]
    if calls == 0:
        return "步间缓存：本次没有采样步可统计"
    extra = "，分道模式" if st["split"] else ""
    if reused == 0:
        return (f"步间缓存：{calls} 步里 0 步命中（阈值 "
                f"{st['threshold']:.2f}{extra}）—— 相邻步变化都超过了阈值，"
                f"把阈值调大才会真的省时间")
    return (f"步间缓存：{calls} 步里复用 {reused} 步"
            f"（{reused / calls * 100:.0f}%，阈值 {st['threshold']:.2f}{extra}）")


def step_cache_status_text() -> str:
    """给界面用的短句（设置对话框里直接显示，能一眼看出"0 命中"）。"""
    st = step_cache_stats()
    if not st["enabled"]:
        return "未启用"
    if st["calls"] == 0:
        return "已启用，本次启动还没有采样数据（生成一张图后再看）"
    pct = st["reuse_ratio"] * 100
    if st["reused"] == 0:
        return (f"已启用但 0 步命中（{st['calls']} 步全部重算，阈值 "
                f"{st['threshold']:.2f}）—— 把阈值调大才会真的省时间")
    tail = "，分道模式" if st["split"] else ""
    return (f"复用 {st['reused']} / {st['calls']} 步（{pct:.0f}%），"
            f"阈值 {st['threshold']:.2f}{tail}")
