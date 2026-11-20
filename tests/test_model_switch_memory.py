# -*- coding: utf-8 -*-
"""换模型必须真的把旧模型放掉（MOSS-Image 之后切别的模型曾经不释放）。

背景：用户实测「先使用 moss 模型后使用 anime 模型，没有把前面的模型给放开，
占用大量内存/显存」。根因有三层：
1. `clear_memory()` 只 gc + empty_cache，MOSS 的 mmap 权重留下的文件页留在
   进程工作集里（任务管理器显示 21GB）—— 必须 EmptyWorkingSet。
2. `_scheduler_base_cfg` 等状态残留，会污染下一个模型的采样器。
3. 用户只是在下拉框里换了模型，而加载是"生成时才做"，于是旧模型一直占着。
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: E402,F401

from utils import mem  # noqa: E402

PASS = FAIL = 0
FAILED = []


def check(name):
    def deco(fn):
        global PASS, FAIL
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            FAIL += 1
            FAILED.append(name)
            print("[FAIL] %s -> %r" % (name, exc))
        else:
            PASS += 1
            print("[PASS] %s" % name)
        return fn
    return deco


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM_SRC = _read(os.path.join(ROOT, "core", "model_manager.py"))
MAIN_SRC = _read(os.path.join(ROOT, "main.py"))
SHELL_SRC = _read(os.path.join(ROOT, "ui", "shell.py"))


@check("mem_helpers_are_safe")
def test_helpers():
    assert isinstance(mem.trim_working_set(), bool)
    assert mem.rss_mb() > 0
    alloc, reserved, free = mem.vram_mb()
    assert isinstance(alloc, float) and isinstance(reserved, float) and isinstance(free, float)
    delta = mem.release_memory()
    for key in ("rss_before", "rss_after", "vram_before", "vram_after"):
        assert key in delta, key
    assert isinstance(mem.describe(delta), str)


@check("clear_memory_trims_working_set_and_resets_scheduler_base")
def test_clear_memory_src():
    # 找到 clear_memory 函数体
    start = MM_SRC.index("def clear_memory(self):")
    end = MM_SRC.index("def apply_optimizations(", start)
    body = MM_SRC[start:end]
    assert "_mem.release_memory()" in body, "clear_memory 必须调 release_memory（含 EmptyWorkingSet）"
    assert "self._scheduler_base_cfg = None" in body, "clear_memory 必须清采样器基线"


@check("unload_all_reports_freed_memory")
def test_unload_all_src():
    start = MM_SRC.index("def unload_all(self):")
    end = MM_SRC.index("def sync_ipa_to_controlnet(", start)
    body = MM_SRC[start:end]
    assert "release_memory()" in body
    assert "所有模型已释放" in body


@check("shell_memory_actions_show_numbers")
def test_shell_src():
    assert "_mem.release_memory()" in SHELL_SRC
    assert "进程内存" in SHELL_SRC
    assert "显存" in SHELL_SRC


@check("selecting_another_model_releases_previous")
def test_release_previous():
    from main import AIDesktopApp

    calls = []

    class FakeAI:
        current_model_name = "moss/MOSS-Image-Base"

        def unload_all(self):
            calls.append(1)

    stub = types.SimpleNamespace(ai=FakeAI(), is_generating=False)
    fn = types.MethodType(AIDesktopApp._release_previous_model_for, stub)

    # 同一个模型 → 不卸载
    assert fn({"path": os.path.join("models", "moss", "MOSS-Image-Base")}) is False
    assert calls == []

    # 换到别的模型 → 卸载一次
    assert fn({"path": os.path.join("models", "anima", "pornmaster.ckpt")}) is True
    assert len(calls) == 1

    # 正在生成 → 不动
    stub2 = types.SimpleNamespace(ai=FakeAI(), is_generating=True)
    fn2 = types.MethodType(AIDesktopApp._release_previous_model_for, stub2)
    calls.clear()
    assert fn2({"path": "other.safetensors"}) is False
    assert calls == []


@check("model_changed_triggers_release")
def test_main_wiring():
    start = MAIN_SRC.index("def _on_model_changed(self, idx=None):")
    end = MAIN_SRC.index("def _release_previous_model_for", start)
    body = MAIN_SRC[start:end]
    assert "_release_previous_model_for(data)" in body


def main():
    print("=" * 60)
    print("总计 %d 项：PASS %d / FAIL %d" % (PASS + FAIL, PASS, FAIL))
    if FAILED:
        print("失败：%s" % FAILED)
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
