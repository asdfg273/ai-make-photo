# -*- coding: utf-8 -*-
"""学习率调度 / 优化器 / 梯度累积的回归（core/lora_train.py + 训练对话框）。

为什么要单独测
--------------
"训练 400 步 loss 一直在降但出图糊"最常见的原因就是学习率曲线不对：
要么没有 warmup（前几步把权重带偏）、要么一路恒定（末期不收敛）、
要么调度器压根没被 step（等于恒定）。这些都不会报错，只会让结果变差，
所以必须用可验证的曲线来钉住。

测法：
1. `lr_factor()` 是纯函数 → 直接断言形状（warmup 爬升、cosine 单调降、restart 回升）。
2. `_make_scheduler()` 造出来的真实 torch 调度器 → 断言末端落到 `lr × min_ratio`。
3. ReduceLROnPlateau 走的是"喂 loss"分支 → 断言它要求 loss 且真的会减半。
4. `_make_optimizer()` 三种取值 + 未知值回退。
5. 对话框下拉框必须与 `LR_SCHEDULERS` / `OPTIMIZERS` 同源（防止以后加选项忘了界面）。
6. 源码级：训练循环里必须有 `accum` / `lr_sched.step(raw_loss)` / `_clear_adapters`。

跑法: venv\\Scripts\\python.exe tests\\test_lora_train_schedule.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: E402

_bootstrap.isolate_config_globally()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = []


def check(name):
    def deco(fn):
        try:
            info = fn()
            RESULT.append((name, True, info))
            print(f"[PASS] {name}" + (f" | {info}" if info else ""))
        except Exception as exc:  # noqa: BLE001 - 测试脚本自己兜底
            RESULT.append((name, False, str(exc)))
            print(f"[FAIL] {name} | {exc}")
            traceback.print_exc()
        return fn
    return deco


def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------- 纯函数形状
@check("warmup 期间学习率线性爬升")
def test_warmup_ramps():
    from core.lora_train import lr_factor

    total, warmup = 100, 5
    vals = [lr_factor(s, total, warmup, "cosine", 0.05) for s in range(warmup)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), vals
    assert _close(vals[-1], 1.0, 1e-9), vals
    assert _close(lr_factor(warmup, total, warmup, "cosine", 0.05), 1.0, 1e-9)
    return f"{[round(v, 3) for v in vals]}"


@check("五种调度器的形状都符合预期")
def test_shapes():
    from core.lora_train import lr_factor

    total, warmup, lo = 100, 5, 0.05
    info = {}
    for name in ("constant", "linear", "cosine", "poly"):
        # 注意：调度器是按"步号"取的，最后一步的步号是 total（此时 p=1，正好落到下限）；
        # 若只看 range(total) 的最后一个，p≈0.99，末端值会略高于下限。
        seq = [lr_factor(s, total, warmup, name, lo) for s in range(warmup, total + 1)]
        assert _close(seq[0], 1.0, 1e-9), (name, seq[0])
        # 非递增
        for a, b in zip(seq, seq[1:]):
            assert b <= a + 1e-9, (name, a, b)
        if name == "constant":
            assert _close(seq[-1], 1.0, 1e-9), seq[-1]
        else:
            assert _close(seq[-1], lo, 1e-6), (name, seq[-1])
        info[name] = round(seq[-1], 4)
    # restart 必须出现回升
    rs = [lr_factor(s, total, warmup, "cosine_restarts", lo) for s in range(warmup, total)]
    rises = [(i, rs[i], rs[i + 1]) for i in range(len(rs) - 1) if rs[i + 1] > rs[i] + 1e-6]
    assert rises, "cosine_restarts 没有出现重启"
    info["cosine_restarts"] = f"回升 {len(rises)} 次，首次 {rises[0][1]:.3f}→{rises[0][2]:.3f}"
    return str(info)


@check("min_ratio=0 时末端真的降到 0")
def test_zero_floor():
    from core.lora_train import lr_factor

    assert _close(lr_factor(1000, 1000, 10, "cosine", 0.0), 0.0, 1e-9)
    assert lr_factor(1000, 1000, 10, "cosine", 0.0) < 1e-6
    return "ok"


@check("真实调度器末端落在 lr × min_ratio")
def test_real_scheduler_floor():
    import torch
    from core.lora_train import _make_scheduler

    for name, expect in (("cosine", 0.05), ("linear", 0.2), ("poly", 0.1)):
        p = torch.nn.Parameter(torch.zeros(3))
        opt = torch.optim.AdamW([{"params": [p], "lr": 1e-3}])
        sched, needs_loss, desc = _make_scheduler(opt, 40, 4, name, expect)
        assert not needs_loss, name
        for _ in range(40):
            sched.step()
        got = opt.param_groups[0]["lr"]
        assert _close(got, 1e-3 * expect, 1e-6), (name, got)
        assert desc, name
    return "cosine/linear/poly 末端都等于下限"


@check("plateau 调度要求喂 loss，且不降时会减半")
def test_plateau():
    import torch
    from core.lora_train import _make_scheduler

    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([{"params": [p], "lr": 1e-3}])
    sched, needs_loss, desc = _make_scheduler(opt, 100, 0, "plateau", 0.05)
    assert needs_loss is True, needs_loss
    assert "loss" in desc or "自动" in desc, desc
    lr0 = opt.param_groups[0]["lr"]
    for _ in range(60):          # patience = max(5, 100//20) = 5
        sched.step(1.0)          # loss 一直不降
    lr1 = opt.param_groups[0]["lr"]
    assert lr1 < lr0, (lr0, lr1)
    assert _close(lr1, lr0 * 0.5, 1e-9) or lr1 <= lr0 * 0.5 + 1e-12, (lr0, lr1)
    return f"{lr0:.2e} → {lr1:.2e}（{desc}）"


@check("未知调度名按余弦处理，不报错")
def test_unknown_name_falls_back():
    from core.lora_train import lr_factor

    a = lr_factor(50, 100, 5, "cosine", 0.05)
    b = lr_factor(50, 100, 5, "我不认识这个", 0.05)
    assert _close(a, b, 1e-12), (a, b)
    return "ok"


# ---------------------------------------------------------------- 优化器
@check("优化器工厂：adamw / adafactor / 未知值回退")
def test_optimizer_factory():
    import torch
    from core.lora_train import _make_optimizer, TrainConfig

    notes = []

    def _mk(name):
        cfg = TrainConfig(optimizer=name)
        p = torch.nn.Parameter(torch.zeros(4))
        return _make_optimizer([{"params": [p], "lr": 1e-4}], cfg, notes.append)

    opt = _mk("adamw")
    assert isinstance(opt, torch.optim.AdamW), type(opt)
    opt2 = _mk("adafactor")
    assert type(opt2).__name__ == "Adafactor", type(opt2)
    opt3 = _mk("随便写点什么")
    assert isinstance(opt3, torch.optim.AdamW), type(opt3)
    opt4 = _mk("adamw8bit")
    assert opt4 is not None
    return " | ".join(notes)[:160] or "ok"


@check("TrainConfig 新字段能 asdict 往返（进 .train.json）")
def test_config_roundtrip():
    import dataclasses
    from core.lora_train import TrainConfig

    cfg = TrainConfig(lr_scheduler="cosine_restarts", min_lr_ratio=0.15,
                      optimizer="adafactor", weight_decay=0.05,
                      grad_accumulation=3)
    data = dataclasses.asdict(cfg)
    for key in ("lr_scheduler", "min_lr_ratio", "optimizer", "weight_decay",
                "grad_accumulation"):
        assert key in data, key
    back = TrainConfig(**data)
    assert back.lr_scheduler == "cosine_restarts"
    assert _close(back.min_lr_ratio, 0.15)
    assert back.optimizer == "adafactor"
    assert _close(back.weight_decay, 0.05)
    assert back.grad_accumulation == 3
    return "5 个字段都在"


# ---------------------------------------------------------------- 界面同源
@check("训练对话框的下拉框与 core 同源")
def test_dialog_matches_core():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QWidget
    from core import lora_train as lt

    app = QApplication.instance() or QApplication([])
    host = QWidget()
    from ui.lora_train_dialog import LoraTrainDialog

    dlg = LoraTrainDialog(host)
    scheds = [dlg.combo_lr_sched.itemData(i) for i in range(dlg.combo_lr_sched.count())]
    opts = [dlg.combo_optim.itemData(i) for i in range(dlg.combo_optim.count())]
    assert scheds == list(lt.LR_SCHEDULERS), (scheds, list(lt.LR_SCHEDULERS))
    assert opts == list(lt.OPTIMIZERS), (opts, list(lt.OPTIMIZERS))

    dlg.combo_lr_sched.setCurrentIndex(dlg.combo_lr_sched.findData("linear"))
    dlg.spin_min_lr.setValue(20.0)
    dlg.combo_optim.setCurrentIndex(dlg.combo_optim.findData("adafactor"))
    dlg.spin_accum.setValue(4)
    dlg.spin_wd.setValue(0.05)
    cfg = dlg._current_cfg()
    assert cfg.lr_scheduler == "linear", cfg.lr_scheduler
    assert _close(cfg.min_lr_ratio, 0.2), cfg.min_lr_ratio
    assert cfg.optimizer == "adafactor", cfg.optimizer
    assert cfg.grad_accumulation == 4, cfg.grad_accumulation
    assert _close(cfg.weight_decay, 0.05), cfg.weight_decay
    dlg.deleteLater()
    host.deleteLater()
    app.processEvents()
    return f"{len(scheds)} 种调度 / {len(opts)} 种优化器"


@check("训练循环真的用上了调度/累积/清适配器")
def test_loop_wiring():
    import io

    src = io.open(os.path.join(ROOT, "core", "lora_train.py"), encoding="utf-8").read()
    need = [
        "stepped = (step_done + 1) % accum == 0",
        "lr_sched.step(raw_loss)",
        "lr_sched.step()",
        "_clear_adapters(unet)",
        "cur_lr = optimizer.param_groups[0][\"lr\"]",
        "sched_needs_loss",
    ]
    missing = [n for n in need if n not in src]
    assert not missing, missing
    return f"{len(need)} 处接线都在"


@check("调度说明文字会对用户说清末端下限")
def test_describe_text():
    from core.lora_train import describe_schedule

    txt = describe_schedule("cosine", 0.05)
    assert "5%" in txt, txt
    txt2 = describe_schedule("plateau", 0.05)
    assert "loss" in txt2 or "自动" in txt2, txt2
    return txt


def main():
    print("=" * 60)
    totals = len(RESULT)
    passed = sum(1 for _, ok, _ in RESULT if ok)
    print(f"总计 {totals} 项：PASS {passed} / FAIL {totals - passed}")
    print("=" * 60)
    return 0 if passed == totals else 1


if __name__ == "__main__":
    sys.exit(main())
