# tests/test_lora_plumbing.py — LoRA 挂载链路（无头，纯脚本，不需要 GPU）
#
# 锁住两个**静默失效**的真实 bug：
#
# 1) `LoadLoRA` 节点传的是 `[{"name":..,"weight":..}]`，而
#    `ModelManager.apply_multiple_loras` 内部是
#    `for (lora_name, weight) in lora_list`。对字典解包拿到的是**键**
#    （"name"/"weight"），于是它去找一个名叫 "name" 的文件、找不到就 continue
#    —— **LoRA 被完全跳过**，出图和不挂 LoRA 逐像素相同（实测 mean=0.0），
#    而节点里的 `except TypeError` 兜底也救不了（2 键字典解包不抛 TypeError）。
#    实测修复前 `pipe.get_active_adapters()` 是 `[]`，修复后是 `['lora_slot_0']`，
#    出图差异从 0.0 变成 33.3。
#
# 2) `LoadCheckpoint` 之后没有清 LoRA：`load_model()` 对同一模型会短路跳过，
#    于是"没有 LoadLoRA 节点的工作流"**继承**上一个工作流挂的 LoRA
#    （实测两张图差异 0.0，本该 33.3）。同一张图跑两次结果不同 = 不可复现。
#
# 用法：venv\Scripts\python.exe tests\test_lora_plumbing.py
import os
import sys

import _bootstrap  # noqa: F401  （放宽 stdout 编码，放最前）

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


class _FakeAI:
    """记录 LoRA 调用与 adapters 状态的最小 ModelManager 替身。"""

    def __init__(self, active=None):
        self.calls = []
        self._active = list(active or [])
        self.txt2img_pipe = self
        self.device = "cpu"
        self.current_arch_id = "sd15"
        self.is_sdxl = False
        self.current_model_name = ""
        self.current_model_path = ""

    # --- apply_multiple_loras 的替身：验证收到的形态 ---
    def apply_multiple_loras(self, lora_list, sub_dir=None):
        self.calls.append(list(lora_list))
        # 模仿真实实现：只有 (name, weight) 元组才会挂上
        ok = []
        for item in (lora_list or []):
            if (isinstance(item, (list, tuple)) and len(item) >= 2
                    and isinstance(item[0], str) and item[0] != "name"):
                ok.append(item)
        self._active = [f"lora_slot_{i}" for i in range(len(ok))]
        return True

    def get_active_adapters(self):
        return list(self._active)

    def load_model(self, name):
        self.current_model_name = name
        return True

    def switch_sampler(self, s):
        return True


def test_normalize_accepts_dict_and_tuple():
    """apply_multiple_loras 必须同时吃得下 dict 与 tuple，且不把键当文件名。"""
    from core.model_manager import ModelManager
    norm = ModelManager._normalize_lora_list
    assert norm([("a.safetensors", 0.8)]) == [("a.safetensors", 0.8)]
    # 关键：dict 要取 value，不能解包成 ("name","weight")
    got = norm([{"name": "b.safetensors", "weight": 0.5}])
    assert got == [("b.safetensors", 0.5)], got
    assert norm([{"lora": "c.safetensors"}]) == [("c.safetensors", 1.0)]
    # 字符串 "名字:权重"
    assert norm(["d.safetensors:0.3"]) == [("d.safetensors", 0.3)]
    # 脏数据丢弃但不抛
    assert norm([None, 123, {"weight": 0.5}]) == []
    assert norm([]) == []
    assert norm(None) == []
    # 绝不能出现 "name"/"weight" 这种键当名字
    for item in ({"name": "x", "weight": 1.0},):
        assert all(n not in ("name", "weight") for n, _ in norm([item]))


def test_load_lora_node_passes_tuples():
    """LoadLoRA 节点必须传 (name, weight) 元组，且挂完要核实。"""
    from workflow import registry
    from workflow.executor import NodeContext
    from workflow.schema import Node
    registry.load_builtin_nodes()
    spec = registry.get("LoadLoRA")
    assert spec is not None

    ai = _FakeAI()
    ctx = NodeContext(ai=ai)
    ctx.log = lambda m: None
    ctx.status = lambda m: None
    node = Node("l", "LoadLoRA", {},
                {"lora_1": "good.safetensors", "weight_1": 0.75})
    out = spec.run(ctx, node, {"model": _fake_model()})

    assert ai.calls, "应当调用 apply_multiple_loras"
    sent = ai.calls[0]
    assert sent and all(isinstance(x, tuple) for x in sent), \
        f"必须传元组，实际: {sent!r}"
    assert sent[0][0] == "good.safetensors" and sent[0][1] == 0.75, sent
    # 挂上了 → applied=True，且不应有 warning
    assert out["lora"] and out["lora"][0].applied is True
    assert ai.get_active_adapters(), "替身应认为挂上了"


def test_load_lora_reports_when_not_applied():
    """挂不上时必须如实上报，不能再静默。"""
    from workflow import registry
    from workflow.executor import NodeContext
    from workflow.schema import Node
    registry.load_builtin_nodes()
    spec = registry.get("LoadLoRA")

    ai = _FakeAI()
    # 让替身拒绝挂载（模拟文件不在当前架构目录下）
    ai.apply_multiple_loras = lambda lst, sub_dir=None: ai.calls.append(lst)
    ctx = NodeContext(ai=ai)
    ctx.log = lambda m: None
    ctx.status = lambda m: None
    node = Node("l", "LoadLoRA", {},
                {"lora_1": "missing.safetensors", "weight_1": 0.7})
    out = spec.run(ctx, node, {"model": _fake_model()})
    assert out["lora"][0].applied is False, "没挂上就该是 False"
    warns = list((ctx.metadata or {}).get("warnings") or [])
    assert warns, f"没挂上必须留 warning，实际: {ctx.metadata}"


def test_load_checkpoint_clears_residual_lora():
    """LoadCheckpoint 必须清掉上一轮残留的 LoRA（保证同一个图可复现）。"""
    from workflow import registry
    from workflow.executor import NodeContext
    from workflow.schema import Node
    registry.load_builtin_nodes()
    spec = registry.get("LoadCheckpoint")

    ai = _FakeAI(active=["lora_slot_0"])          # 假装上一轮挂了 LoRA
    ai.txt2img_pipe = object()                    # 让 pipe 属性存在即可
    ctx = NodeContext(ai=ai)
    ctx.log = lambda m: None
    ctx.status = lambda m: None
    node = Node("m", "LoadCheckpoint", {},
                {"path": "some_model.safetensors"})
    spec.run(ctx, node, {})

    assert ai.calls, "LoadCheckpoint 应当调用 apply_multiple_loras 清空"
    assert ai.calls[0] == [], \
        f"第一次调用应是清空（空列表），实际: {ai.calls[0]!r}"
    assert not ai.get_active_adapters(), "残留 LoRA 应被清掉"


def test_load_checkpoint_clears_even_when_model_reused():
    """同一模型被短路跳过时也必须清 —— 这正是泄漏的成因。"""
    from workflow import registry
    from workflow.executor import NodeContext
    from workflow.schema import Node
    registry.load_builtin_nodes()
    spec = registry.get("LoadCheckpoint")

    ai = _FakeAI(active=["lora_slot_0"])
    ai.txt2img_pipe = object()
    ai.current_model_name = "some_model.safetensors"   # 同一个模型
    ai.load_model = lambda n: True                     # 短路（不重载）
    ctx = NodeContext(ai=ai)
    ctx.log = lambda m: None
    ctx.status = lambda m: None
    node = Node("m", "LoadCheckpoint", {},
                {"path": "some_model.safetensors"})
    spec.run(ctx, node, {})
    assert ai.calls and ai.calls[0] == [], \
        "模型没重载时同样必须清 LoRA，否则会泄漏到本轮"
    assert not ai.get_active_adapters()


def _fake_model():
    from workflow.values import ModelRef
    return ModelRef(name="m", path="p", arch_id="sd15", loaded=True)


def test_real_manager_normalizes_the_old_dict_call():
    """用真实的 ModelManager（不加载模型）验证 dict 不再被静默跳过。"""
    from core.model_manager import ModelManager
    m = ModelManager()
    # 不改动管线，只看规范化结果：dict 必须变成 (name, weight)
    got = m._normalize_lora_list([{"name": "x.safetensors", "weight": 0.9}])
    assert got == [("x.safetensors", 0.9)], got
    assert "name" not in [n for n, _ in got]


if __name__ == "__main__":
    import main  # noqa: F401  （确保 UI 环境可用）
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failed.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"总计 {len(fns)} 项：PASS {len(fns) - len(failed)} / FAIL {len(failed)}")
    for f in failed:
        print(f"  失败: {f}")
    sys.exit(1 if failed else 0)
