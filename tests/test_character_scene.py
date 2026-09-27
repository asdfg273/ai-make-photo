"""角色库 / 场景库 / 批量 无头测试 —— 可独立运行，也可被 pytest 收集。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import tempfile
from pathlib import Path

# 让本项目 import 生效（独立运行时 sys.path[0]=tests 目录，需补项目根）
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import QApplication, QLabel, QWidget
_APP = QApplication.instance() or QApplication([])

from utils.scene_library import Scene, SceneLibrary
from utils.character_library import Character, CharacterLibrary
import ui.scene_batch_dialog as sbd
from ui.pages import characters_page as cp


def test_scene_fields_exist_on_dataclass():
    keys = {k for k, *_ in sbd._SCENE_FIELDS}
    missing = keys - set(Scene.__dataclass_fields__)
    assert not missing, f"UI 字段映射到了 Scene 不存在的属性: {missing}"
    assert "pose_prompt" in keys
    assert "negative_extra" in keys
    enable = ("name", "pose_prompt", "negative_extra", "count", "seed", "enabled")
    assert set(enable) <= set(Scene.__dataclass_fields__), "启用名单含假字段"


def test_scene_merge():
    s = Scene(character_uuid="c", name="n", pose_prompt="standing",
              negative_extra="blurry")
    assert s.merged_positive("base") == "base, standing"
    assert s.merged_negative("base") == "base, blurry"


def test_scene_library_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "scenes.json"
        lib = SceneLibrary(p)
        s = Scene(character_uuid="c1", name="夜景", pose_prompt="night, city",
                  count=3, enabled=True)
        assert lib.add(s), "带 character_uuid 的场景应能入库"
        assert not lib.add(Scene(name="无主")), "缺 character_uuid 应被拒绝"

        lib2 = SceneLibrary(p)
        loaded = lib2.list_for_character("c1")
        assert len(loaded) == 1
        assert loaded[0].pose_prompt == "night, city"
        assert loaded[0].count == 3


def test_character_roundtrip_with_tags():
    with tempfile.TemporaryDirectory() as d:
        lib = CharacterLibrary(d)
        c = Character(name="甲", tags=["日常", "泳装"])
        assert lib.save_character(c)
        got = lib.get_character(c.uuid)
        assert got is not None and got.tags == ["日常", "泳装"]
        assert c.uuid in {x.uuid for x in lib.list_characters(tag="日常")}


def _stub_shell():
    s = QWidget()
    s.nav = None
    s.lbl_status = QLabel()
    s.combo_loras = []
    s.scale_loras = []
    s.txt_prompt = s.txt_neg = None
    s.combo_res = s.spin_steps = s.spin_seed = s.scale_cfg = None
    return s


def test_characters_page_build_and_tag_filter():
    page = cp.CharactersPage()
    stub = _stub_shell()
    page.build(stub)
    assert page.workspace() is not None
    assert page.params_widget() is not None

    with tempfile.TemporaryDirectory() as d:
        page.library = CharacterLibrary(d)
        lib = page.library
        c = Character(name="甲", tags=["泳装"])
        lib.save_character(c)
        page.refresh()
        assert page._cards, "没生成任何卡片"
        assert page.cbo_tag.findData("泳装") >= 0

        page.cbo_tag.setCurrentIndex(page.cbo_tag.findData("泳装"))
        page.refresh()
        assert page._cards, "按标签筛选后应为非空"

        page.cbo_tag.setCurrentIndex(0)   # 切回“全部标签”
        page.refresh()
    page.dispose()


if __name__ == "__main__":
    _fns = [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]
    for n, f in _fns:
        f()
        print(f"PASS {n}")
    print(f"\n共 {len(_fns)} 项通过")
