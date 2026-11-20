# tests/test_step_cache_split.py — 步间缓存：分道 + 命中率必须可验证
#
# 为什么单独一个文件
# ------------------
# 用户的原话是"勾了加速但一点没快"。这类问题的麻烦之处在于：
#   · 界面显示"已启用"，`enabled: True`，看着一切正常；
#   · 实际一步都没复用，出图慢，但没有任何报错；
#   · 而"没有复用"有两种完全不同的原因：阈值太小（该调大），
#     或者管线每步调了两次前向（uncond/cond），单 lane 永远比不出相似度。
#
# 所以这里锁三件事：
#   1. 单前向管线（批量 CFG，SD1.5/SDXL 的真实形态）照旧能命中；
#   2. 两次前向的管线会自动改成"奇偶分道"，从 0 命中变成有命中；
#   3. 分道如果没能救回来，下一张图必须退回单 lane，而不是一直背着。
#
# 用法：venv\Scripts\python.exe tests\test_step_cache_split.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _unet(torch):
    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, sample):
            self.calls += 1
            return sample * 2

    return TinyUNet()


def _pipe(torch):
    from types import SimpleNamespace
    return SimpleNamespace(unet=_unet(torch))


def test_single_pass_still_hits_and_never_splits():
    """批量 CFG（每步一次前向）：必须照旧命中，且不该误判成两次前向。"""
    import torch
    from utils.step_cache import (configure_step_cache, reset_step_cache,
                                  step_cache_stats, step_cache_status_text)

    pipe = _pipe(torch)
    configure_step_cache(pipe, "sd15", True, threshold=0.5, max_skips=3)
    try:
        reset_step_cache()
        for i in range(8):
            pipe.unet(sample=torch.ones(1, 4) * (1.0 + 0.01 * i))
        st = step_cache_stats()
        assert st["split"] is False, "单前向管线不该分道"
        assert st["reused"] >= 2, f"相似步应该命中，实际 {st}"
        assert "复用" in step_cache_status_text()
    finally:
        configure_step_cache(pipe, "sd15", False)


def test_two_pass_cfg_auto_splits_and_hits():
    """uncond/cond 两次前向：单 lane 命中 0 次 → 自动分道后要有命中。"""
    import torch
    from utils.step_cache import (configure_step_cache, reset_step_cache,
                                  step_cache_stats, step_cache_status_text)

    pipe = _pipe(torch)
    configure_step_cache(pipe, "sd3", True, threshold=0.5, max_skips=3)
    try:
        reset_step_cache()
        for i in range(10):
            pipe.unet(sample=torch.ones(1, 4) * (1.0 + 0.01 * i))       # uncond
            pipe.unet(sample=torch.full((1, 4), 3.0) * (1.0 + 0.01 * i))  # cond
        st = step_cache_stats()
        assert st["split"] is True, f"两次前向应触发分道，实际 {st}"
        assert st["reused"] >= 1, f"分道后应能命中，实际 {st}"
        assert "分道" in step_cache_status_text()
    finally:
        configure_step_cache(pipe, "sd3", False)


def test_split_is_dropped_when_it_does_not_help():
    """分道没救回来（阈值本来就太小）→ 下一张图退回单 lane。"""
    import torch
    from utils.step_cache import configure_step_cache, reset_step_cache, step_cache_stats

    pipe = _pipe(torch)
    configure_step_cache(pipe, "sd3", True, threshold=0.001, max_skips=3)
    try:
        reset_step_cache()
        # 每个 lane 内部也大步变化（每步 +10），确保分道后同样一步都不像，
        # 这样"分道没救回来"才是真的 0 命中。
        for i in range(10):
            pipe.unet(sample=torch.ones(1, 4) * (1.0 + 10.0 * i))
            pipe.unet(sample=torch.full((1, 4), 3.0) * (1.0 + 10.0 * i))
        assert step_cache_stats()["split"] is True, "先用两次前向触发分道"
        assert step_cache_stats()["reused"] == 0, "阈值 0.001 不该有命中"
        reset_step_cache()                      # 等价于"下一张图开始"
        assert step_cache_stats()["split"] is False, "分道没命中就该退回单 lane"
    finally:
        configure_step_cache(pipe, "sd3", False)


def test_status_text_distinguishes_three_cases():
    """界面上必须能一眼分清：未启用 / 还没数据 / 已启用但 0 命中。"""
    import torch
    from utils.step_cache import (configure_step_cache, step_cache_status_text)

    pipe = _pipe(torch)
    configure_step_cache(pipe, "sd15", False)
    assert step_cache_status_text() == "未启用"

    configure_step_cache(pipe, "sd15", True, threshold=0.001, max_skips=1)
    try:
        assert "还没有采样数据" in step_cache_status_text()
        for i in range(4):
            pipe.unet(sample=torch.ones(1, 4) * (1.0 + 0.5 * i))
        text = step_cache_status_text()
        assert "0 步命中" in text, text
        assert "阈值" in text
    finally:
        configure_step_cache(pipe, "sd15", False)


def test_settings_dialog_shows_hit_rate():
    """设置页必须真的把这个数字显示出来（源码级；不建窗口）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "ui", "settings_dialog.py"),
              encoding="utf-8") as f:
        src = f.read()
    assert "step_cache_status_text" in src, "设置页没显示命中率"
    assert "_refresh_step_cache_stats" in src, "命中率没有刷新入口"
    assert "lbl_step_cache_stats" in src


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            import traceback
            failed.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("=" * 60)
    print(f"总计 {len(fns)} 项：PASS {len(fns) - len(failed)} / FAIL {len(failed)}")
    for f in failed:
        print(f"  失败: {f}")
    sys.exit(1 if failed else 0)
