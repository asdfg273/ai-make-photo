"""「存为角色」按钮无头测试 —— 可独立运行，也可被 pytest 收集。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import tempfile
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QTextEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QSlider, QCheckBox,
)
from PyQt6.QtCore import Qt
_APP = QApplication.instance() or QApplication([])

from utils.character_library import Character, CharacterLibrary
from ui.pages import characters_page as cp


def _stub_shell(**attrs):
    s = QWidget()
    s.nav = None
    s.lbl_status = QLabel()
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _shell_with_params(tmp_png: str = ""):
    """模拟主窗口当前参数状态的 shell。"""
    s = _stub_shell()
    s.txt_prompt = QTextEdit()
    s.txt_prompt.setPlainText("1girl, brown_hair, fox_ears, white_dress")
    s.txt_neg = QTextEdit()
    s.txt_neg.setPlainText("blurry, lowres")
    s.combo_loras = []
    s.scale_loras = []
    s.chk_use_ipa = QCheckBox()
    s.spin_ipa_scale = QDoubleSpinBox()
    s.spin_ipa_scale.setValue(0.55)
    s.chk_use_pose = QCheckBox()
    s.combo_cn_type = QComboBox()
    s.combo_cn_type.addItems(["", "openpose", "canny"])
    s.combo_cn_type.setCurrentText("openpose")
    s.scale_cn_strength = QSlider(Qt.Orientation.Horizontal)
    s.scale_cn_strength.setRange(0, 200)
    s.scale_cn_strength.setValue(80)
    s.combo_model = QComboBox()
    s.combo_model.addItem("meina.safetensors")
    s.spin_steps = QSpinBox()
    s.spin_steps.setValue(28)
    s.scale_cfg = QDoubleSpinBox()
    s.scale_cfg.setValue(7.5)
    s.combo_sampler = QComboBox()
    s.combo_sampler.addItems(["Euler a", "DPM++ 2M Karras"])
    s.combo_res = QComboBox()
    s.combo_res.addItems(["512x768", "768x512"])
    s.combo_res.setCurrentText("768x512")
    s.spin_seed = QSpinBox()
    s.spin_seed.setRange(0, 999999)
    s.spin_seed.setValue(12345)
    s.scale_strength = QSlider(Qt.Orientation.Horizontal)
    s.scale_strength.setRange(0, 200)
    s.scale_strength.setValue(60)
    if tmp_png:
        s.ipa_image_path = tmp_png
    return s


def test_gen_params_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        lib = CharacterLibrary(d)
        c = Character(name="甲")
        c.gen_params = {"steps": 28, "cfg": 7.5, "sampler": "Euler a",
                        "resolution": "512x768", "seed": 12345}
        assert lib.save_character(c)
        got = lib.get_character(c.uuid)
        assert got is not None
        assert got.gen_params["steps"] == 28
        assert got.gen_params["sampler"] == "Euler a"
    # 旧 JSON（无 gen_params 字段）加载后应为默认空 dict
    c2 = Character.from_dict({"name": "乙"})
    assert c2.gen_params == {}


def test_capture_shell_params():
    shell = _shell_with_params()
    cap = cp.capture_shell_params(shell)
    assert cap["prompt"] == "1girl, brown_hair, fox_ears, white_dress"
    assert cap["negative"] == "blurry, lowres"
    # IPA 未勾选 → 不抓强度
    assert cap["ip_adapter_scale"] is None
    # CN 未勾选 → 类型留空
    assert cap["controlnet_type"] == ""
    # gen_params 记录采样参数
    assert cap["gen_params"]["steps"] == 28
    assert cap["gen_params"]["cfg"] == 7.5
    assert cap["gen_params"]["sampler"] == "Euler a"
    assert cap["gen_params"]["resolution"] == "768x512"
    assert cap["gen_params"]["seed"] == 12345
    assert cap["gen_params"]["strength"] == 0.6
    assert "meina.safetensors" in cap["gen_params"]["model"]


def test_capture_toggles_on():
    shell = _shell_with_params()
    shell.chk_use_ipa.setChecked(True)
    shell.chk_use_pose.setChecked(True)
    cap = cp.capture_shell_params(shell)
    assert cap["ip_adapter_scale"] == 0.55
    assert cap["controlnet_type"] == "openpose"
    assert cap["controlnet_scale"] == 0.8


def test_create_character_from_shell():
    class _Dlg:
        @staticmethod
        def getText(*a, **kw):
            return "测试角色", True

    old_dlg = cp.QInputDialog
    cp.QInputDialog = _Dlg
    try:
        page = cp.CharactersPage()
        shell = _stub_shell()
        page.build(shell)
        with tempfile.TemporaryDirectory() as d:
            page.library = CharacterLibrary(d)
            gen_shell = _shell_with_params()
            char = page.create_character_from_shell(gen_shell, parent=shell)
            assert char is not None
            assert char.name == "测试角色"
            assert char.base_prompt.startswith("1girl")
            assert char.negative_prompt == "blurry, lowres"
            assert char.gen_params["steps"] == 28
            # 入库后可重新加载
            got = page.library.get_character(char.uuid)
            assert got is not None and got.name == "测试角色"
            # 页面已选中新卡
            assert page._selected_uuid == char.uuid
        page.dispose()
    finally:
        cp.QInputDialog = old_dlg


def test_create_character_copies_ref_image():
    class _Dlg:
        @staticmethod
        def getText(*a, **kw):
            return "带图角色", True

    from PIL import Image
    old_dlg = cp.QInputDialog
    cp.QInputDialog = _Dlg
    try:
        with tempfile.TemporaryDirectory() as img_d, tempfile.TemporaryDirectory() as lib_d:
            src = Path(img_d) / "ref.png"
            Image.new("RGB", (64, 64), (200, 100, 50)).save(src)

            page = cp.CharactersPage()
            shell = _stub_shell()
            page.build(shell)
            page.library = CharacterLibrary(lib_d)

            gen_shell = _shell_with_params(tmp_png=str(src))
            char = page.create_character_from_shell(gen_shell, parent=shell)
            assert char is not None
            # 参考图应拷进库目录，而非引用源路径
            assert char.ref_image_path != str(src)
            assert Path(char.ref_image_path).exists()
            assert Path(char.ref_image_path).parent == Path(lib_d)
            assert char.reference_images and char.reference_images[0]["pose"] == "front"
            assert char.thumbnail_path, "入库时应生成缩略图"
            page.dispose()
    finally:
        cp.QInputDialog = old_dlg


def test_create_character_empty_prompt_rejected():
    class _MB:
        @staticmethod
        def warning(*a, **kw):
            pass

        critical = warning

    old_mb = cp.QMessageBox
    cp.QMessageBox = _MB
    try:
        page = cp.CharactersPage()
        shell = _stub_shell()
        page.build(shell)
        empty = _shell_with_params()
        empty.txt_prompt.setPlainText("   ")
        assert page.create_character_from_shell(empty, parent=shell) is None
        page.dispose()
    finally:
        cp.QMessageBox = old_mb


def test_apply_character_restores_gen_params():
    class _MB:
        @staticmethod
        def information(*a, **kw):
            pass

        warning = information
        critical = information
        question = staticmethod(
            lambda *a, **kw: None)

    old_mb = cp.QMessageBox
    cp.QMessageBox = _MB
    try:
        shell = _shell_with_params()
        shell._pages = {"img2img": object()}

        char = Character(name="还原测试")
        char.base_prompt = "1girl, silver_hair"
        char.negative_prompt = ""
        char.gen_params = {"steps": 40, "cfg": 5.5, "sampler": "DPM++ 2M Karras",
                         "resolution": "512x768", "seed": 777,
                         "strength": 0.75}
        cp._apply_character_to_shell(shell, char, shell)

        assert shell.spin_steps.value() == 40
        assert abs(shell.scale_cfg.value() - 5.5) < 1e-6
        assert shell.combo_sampler.currentText() == "DPM++ 2M Karras"
        assert shell.combo_res.currentText() == "512x768"
        assert shell.spin_seed.value() == 777
        # QSlider(0-200) 强度 0.75 → 75
        assert shell.scale_strength.value() == 75
    finally:
        cp.QMessageBox = old_mb


if __name__ == "__main__":
    _fns = [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]
    for n, f in _fns:
        f()
        print(f"PASS {n}")
    print(f"\n共 {len(_fns)} 项通过")
