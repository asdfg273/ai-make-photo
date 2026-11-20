# -*- coding: utf-8 -*-
"""
ADetailer 模型路径统一（市场 / 下载器 / 检测器只有一处真相）

背景
----
市场条目、`utils/model_downloader.py` 写的是 `models/adetailer/`，
但旧版本把文件实际下到 `models_cache/adetailer/`，于是：
  · 文件明明在盘上，市场仍显示"未安装"；
  · 用户点"下载"会重复拉一遍同一份权重；
  · 检测器有时按另一个目录找不到，日志提示"未找到本地模型"。

现在：`utils.paths.adetailer_path()` 是唯一解析入口，
`migrate_adetailer_models()` 把旧位置的东西搬过来（幂等）。
"""
import io
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: E402

_bootstrap.isolate_config_globally()

from utils import paths  # noqa: E402

OK = 0
FAIL = 0


def check(name):
    def deco(fn):
        global OK, FAIL
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
            FAIL += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            OK += 1
            print(f"ok   {name}")
        return fn
    return deco


def _sandbox_roots():
    """项目内沙箱目录（不用 tempfile.mkdtemp：本环境 ACL 会锁住临时目录）"""
    base = os.path.join(ROOT, "_tmp_asset_sandbox")
    new = os.path.join(base, "models_adetailer")
    old = os.path.join(base, "cache_adetailer")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(new, exist_ok=True)
    return base, new, old


def _touch(path, size=16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)


@check("adetailer_path_normalizes_every_spelling")
def t_normalize():
    name = "face_yolov8n.pt"
    want = os.path.join(paths.ADETAILER_DIR, name)
    for raw in (name,
                f"models/adetailer/{name}",
                f"models\\adetailer\\{name}",
                f"models_cache/adetailer/{name}"):
        got = paths.adetailer_path(raw)
        assert got == want, f"{raw!r} -> {got!r} != {want!r}"
    # 绝对路径原样保留（用户可能在设置里填了别处）
    absy = os.path.join(ROOT, "somewhere", name)
    assert paths.adetailer_path(absy) == os.path.normpath(absy)
    # 空值 → 目录本身
    assert paths.adetailer_path("") == paths.ADETAILER_DIR


@check("migrate_moves_legacy_files")
def t_migrate():
    base, new, old = _sandbox_roots()
    _touch(os.path.join(old, "face_yolov8n.pt"), 32)
    _touch(os.path.join(old, "hand_yolov8n.pt"), 32)

    saved = (paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR)
    paths.ADETAILER_DIR = new
    paths.LEGACY_ADETAILER_DIR = old
    try:
        moved = paths.migrate_adetailer_models()
        assert sorted(moved) == ["face_yolov8n.pt", "hand_yolov8n.pt"], moved
        assert os.path.isfile(os.path.join(new, "face_yolov8n.pt"))
        assert not os.path.isdir(old), "旧目录空了应该被删掉"
        # 幂等
        assert paths.migrate_adetailer_models() == []
    finally:
        paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR = saved
        shutil.rmtree(base, ignore_errors=True)


@check("migrate_keeps_legacy_dir_when_it_still_has_other_files")
def t_migrate_keeps_notes():
    base, new, old = _sandbox_roots()
    _touch(os.path.join(old, "face_yolov8n.pt"), 32)
    _touch(os.path.join(old, "notes.txt"), 4)          # 非模型后缀不该被搬

    saved = (paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR)
    paths.ADETAILER_DIR = new
    paths.LEGACY_ADETAILER_DIR = old
    try:
        assert paths.migrate_adetailer_models() == ["face_yolov8n.pt"]
        assert os.path.isfile(os.path.join(old, "notes.txt"))
        assert os.path.isdir(old), "还有别的文件时不该删目录"
    finally:
        paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR = saved
        shutil.rmtree(base, ignore_errors=True)


@check("migrate_never_overwrites_newer_file")
def t_migrate_no_overwrite():
    base, new, old = _sandbox_roots()
    _touch(os.path.join(old, "face_yolov8n.pt"), 8)
    _touch(os.path.join(new, "face_yolov8n.pt"), 999)

    saved = (paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR)
    paths.ADETAILER_DIR = new
    paths.LEGACY_ADETAILER_DIR = old
    try:
        assert paths.migrate_adetailer_models() == []
        assert os.path.getsize(os.path.join(new, "face_yolov8n.pt")) == 999
    finally:
        paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR = saved
        shutil.rmtree(base, ignore_errors=True)


@check("migrate_is_noop_without_legacy_dir")
def t_migrate_noop():
    base, new, old = _sandbox_roots()
    saved = (paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR)
    paths.ADETAILER_DIR = new
    paths.LEGACY_ADETAILER_DIR = old
    try:
        assert paths.migrate_adetailer_models() == []
    finally:
        paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR = saved
        shutil.rmtree(base, ignore_errors=True)


@check("image_processor_uses_the_same_resolver")
def t_image_processor():
    import utils.image_processor as ip

    p = ip._get_adetailer_model_path({"local_path": "models/adetailer/hand_yolov8n.pt"})
    assert str(p) == os.path.join(paths.ADETAILER_DIR, "hand_yolov8n.pt"), p
    p2 = ip._get_adetailer_model_path({"name": "face_yolov8n.pt"})
    assert str(p2) == os.path.join(paths.ADETAILER_DIR, "face_yolov8n.pt"), p2
    # 缺配置要报错，而不是悄悄用某个默认路径
    try:
        ip._get_adetailer_model_path({})
    except ValueError:
        pass
    else:
        raise AssertionError("空配置应当抛 ValueError")


@check("resolver_falls_back_to_legacy_location")
def t_legacy_fallback():
    import utils.image_processor as ip

    base, new, old = _sandbox_roots()
    _touch(os.path.join(old, "face_yolov8n.pt"), 32)

    saved = (paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR)
    paths.ADETAILER_DIR = new
    paths.LEGACY_ADETAILER_DIR = old
    try:
        # migrate 会搬过去，所以这里先禁用它来验证"兜底"分支
        orig = paths.migrate_adetailer_models
        paths.migrate_adetailer_models = lambda: []
        try:
            got = ip._resolve_local_adetailer_model({"name": "face_yolov8n.pt"})
        finally:
            paths.migrate_adetailer_models = orig
        assert got and got.endswith("face_yolov8n.pt"), got
        # 搬过之后也能找到（正常路径）
        paths.migrate_adetailer_models()
        got2 = ip._resolve_local_adetailer_model({"name": "face_yolov8n.pt"})
        assert got2 == os.path.join(new, "face_yolov8n.pt"), got2
    finally:
        paths.ADETAILER_DIR, paths.LEGACY_ADETAILER_DIR = saved
        shutil.rmtree(base, ignore_errors=True)


@check("market_agrees_with_the_single_location")
def t_market_agrees():
    """市场"已安装"必须等价于"唯一位置上有文件"，不允许两套路径。"""
    from utils import extension_manager as ext

    for ext_id, fname in (("adetailer_face", "face_yolov8n.pt"),
                          ("adetailer_hand", "hand_yolov8n.pt")):
        want = os.path.isfile(paths.adetailer_path(fname))
        got = ext.is_installed(ext_id)
        assert bool(got) == want, f"{ext_id}: 市场={got} 文件存在={want}"
        # 条目里的 check 必须就是 models/adetailer/<fname>
        entry = ext.EXTENSIONS[ext_id]
        assert entry["check"] == [f"models/adetailer/{fname}"], entry["check"]


@check("no_code_writes_adetailer_to_cache_dir")
def t_no_legacy_writes():
    """代码里除迁移用的常量外，不允许再把 ADetailer 写到 models_cache。

    只看**字符串字面量**（注释/文档里提到旧路径是合法的，那是在解释历史）。
    """
    offenders = []
    for rel in ("utils/extension_manager.py", "utils/model_downloader.py",
                "utils/image_processor.py", "core/regional.py"):
        p = os.path.join(ROOT, rel)
        src = io.open(p, encoding="utf-8").read()
        for lit in ('"models_cache/adetailer', "'models_cache/adetailer",
                    '"models_cache\\\\adetailer', "'models_cache\\\\adetailer"):
            if lit in src:
                offenders.append(f"{rel}: {lit}")
    assert not offenders, offenders
    # main.py 必须启动时调用迁移
    main_src = io.open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "migrate_adetailer_models" in main_src


@check("regional_hint_uses_the_same_path")
def t_regional():
    import core.regional as reg
    from utils.image_processor import ADETAILER_MODELS

    src = io.open(os.path.join(ROOT, "core", "regional.py"), encoding="utf-8").read()
    assert "_resolve_local_adetailer_model" in src
    # ADETAILER_MODELS 的键是中文 target（现实脸部 / 二次元手部 …）
    for target in ADETAILER_MODELS:
        got = str(reg._adetailer_hint_path({"target": target}))
        assert got.endswith(".pt"), f"{target} -> {got!r}"
        assert got.startswith(paths.ADETAILER_DIR), f"{target} -> {got!r}"


if __name__ == "__main__":
    print(f"\n{OK} passed, {FAIL} failed")
    if FAIL:
        sys.exit(1)
    print("ASSET_PATHS_OK")
