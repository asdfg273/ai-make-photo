"""TeaCache 风格的轻量步间输出缓存。

按相邻步输入的相对 L1 距离决定是否复用上一次 DiT/UNet 输出。
这是通用近似实现，不是某个第三方 TeaCache 仓库的逐行移植。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_ACTIVE = None


def _sample_distance(previous, current) -> float:
    import torch
    if previous is None or previous.shape != current.shape:
        return float("inf")
    left = previous.detach().reshape(-1)
    right = current.detach().reshape(-1)
    stride = max(1, left.numel() // 4096)
    left = left[::stride].float()
    right = right[::stride].float()
    denominator = left.abs().mean().clamp_min(1e-6)
    return float(((left - right).abs().mean() / denominator).item())


class _StepCache:
    def __init__(self, model, threshold: float, max_skips: int):
        self.model = model
        self.threshold = max(0.01, float(threshold))
        self.max_skips = max(1, int(max_skips))
        self._original = model.forward
        self.reset()
        model.forward = self._forward

    def reset(self):
        self._previous_input = None
        self._previous_output = None
        self._accumulated = 0.0
        self._consecutive_skips = 0
        self.computed = 0
        self.reused = 0

    def restore(self):
        try:
            del self.model.__dict__["forward"]
        except KeyError:
            self.model.forward = self._original

    def _input_tensor(self, args, kwargs):
        import torch
        for key in ("hidden_states", "sample"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                return value
        return args[0] if args and isinstance(args[0], torch.Tensor) else None

    def _forward(self, *args, **kwargs):
        current = self._input_tensor(args, kwargs)
        can_reuse = self._previous_output is not None and current is not None
        if can_reuse:
            self._accumulated += _sample_distance(self._previous_input, current)
            can_reuse = (self._accumulated < self.threshold
                         and self._consecutive_skips < self.max_skips)
        if current is not None:
            self._previous_input = current
        if can_reuse:
            self._consecutive_skips += 1
            self.reused += 1
            return self._previous_output

        output = self._original(*args, **kwargs)
        self._previous_output = output
        self._accumulated = 0.0
        self._consecutive_skips = 0
        self.computed += 1
        return output


def configure_step_cache(pipe, arch_id: str, enabled: bool,
                         threshold: float = 0.18, max_skips: int = 2):
    """配置当前管线的缓存；禁用或换管线时恢复原 forward。"""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.restore()
        _ACTIVE = None
    if not enabled:
        return None
    attr = "transformer" if arch_id == "flux" else "unet" if arch_id == "sdxl" else ""
    if not attr:
        logger.info("步间缓存不支持架构 %s，按原采样执行", arch_id)
        return None
    model = getattr(pipe, attr, None)
    if model is None:
        raise RuntimeError(f"步间缓存找不到当前管线的 {attr}。")
    _ACTIVE = _StepCache(model, threshold, max_skips)
    logger.info("步间缓存已启用：threshold=%.3f, max_skips=%s", threshold, max_skips)
    return _ACTIVE


def reset_step_cache():
    if _ACTIVE is not None:
        _ACTIVE.reset()


def step_cache_stats() -> dict:
    if _ACTIVE is None:
        return {"enabled": False, "computed": 0, "reused": 0}
    return {
        "enabled": True,
        "computed": _ACTIVE.computed,
        "reused": _ACTIVE.reused,
        "consecutive_skips": _ACTIVE._consecutive_skips,
    }
