# tests/test_inference_productivity.py — 推理性能与素材管理增强（不加载真实模型）
import os
import sys
import _bootstrap  # noqa: F401,E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_QT_APP = None


def _qapp():
    """整个测试进程共享一个 QApplication，避免全局 Qt 对象随局部引用销毁。"""
    global _QT_APP
    from PyQt6.QtWidgets import QApplication
    if _QT_APP is None:
        _QT_APP = QApplication.instance() or QApplication([])
    return _QT_APP


def test_quantization_mapping_and_signature():
    from unittest.mock import patch
    from core import dit_optimization as opt

    from diffusers import BitsAndBytesConfig
    fp8_config = BitsAndBytesConfig(load_in_8bit=True)
    nvfp4_config = BitsAndBytesConfig(load_in_4bit=True)
    assert opt.quantization_config("off") is None
    with patch.object(opt, "_fp8_config", return_value=fp8_config):
        fp8 = opt.quantization_config("fp8")
    with patch.object(opt, "_nvfp4_config", return_value=nvfp4_config):
        nvfp4 = opt.quantization_config("nvfp4")
    assert fp8.quant_mapping == {"transformer": fp8_config}
    assert nvfp4.quant_mapping == {"transformer": nvfp4_config}
    assert opt.optimization_signature({"dit_quantization": "fp8", "dit_offload_blocks": 2}) == (
        "fp8", False, 2)
    assert opt.optimization_signature({"dit_quantization": "nvfp4"})[0] == "nvfp4"
    print("PASS test_quantization_mapping_and_signature")


def test_group_offload_marks_pipe():
    from types import SimpleNamespace
    from unittest.mock import patch
    from core.dit_optimization import apply_group_offload

    pipe = SimpleNamespace(transformer=object(), _dit_group_offloaded=False)
    calls = []
    def fake_apply(module, **kwargs):
        calls.append((module, kwargs))
    with patch("diffusers.hooks.apply_group_offloading", fake_apply):
        assert apply_group_offload(pipe, True, 2, "cuda") is True
    assert pipe._dit_group_offloaded is True
    assert calls[0][0] is pipe.transformer
    assert calls[0][1]["offload_type"] == "block_level"
    assert calls[0][1]["num_blocks_per_group"] == 2
    assert apply_group_offload(pipe, True, 2, "cuda") is True
    assert len(calls) == 1, "已分组卸载的管线不得重复安装 hook"
    print("PASS test_group_offload_marks_pipe")


def test_quantization_backend_error_is_explicit():
    from unittest.mock import patch
    from core import dit_optimization as opt

    with patch.object(opt, "_fp8_config", side_effect=RuntimeError("请安装 torchao")):
        try:
            opt.quantization_config("fp8")
        except RuntimeError as exc:
            assert "torchao" in str(exc)
        else:
            raise AssertionError("缺少 FP8 后端时必须明确失败")
    print("PASS test_quantization_backend_error_is_explicit")


def test_step_cache_reuses_and_resets():
    from types import SimpleNamespace
    import torch
    from utils.step_cache import configure_step_cache, reset_step_cache, step_cache_stats

    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, sample):
            self.calls += 1
            return SimpleNamespace(sample=sample * 2)

    pipe = SimpleNamespace(unet=TinyUNet())
    original = pipe.unet.forward
    configure_step_cache(pipe, "sdxl", True, threshold=0.5, max_skips=1)
    pipe.unet(sample=torch.ones(1, 4))
    pipe.unet(sample=torch.ones(1, 4) * 1.001)
    assert pipe.unet.calls == 1, "相似第二步应复用缓存"
    pipe.unet(sample=torch.ones(1, 4) * 1.002)
    assert pipe.unet.calls == 2, "达到连续复用上限必须真实计算"
    assert step_cache_stats()["reused"] == 1
    reset_step_cache()
    pipe.unet(sample=torch.ones(1, 4))
    assert pipe.unet.calls == 3, "每张图复位后首步必须真实计算"
    configure_step_cache(pipe, "sdxl", False)
    assert pipe.unet.forward == original
    print("PASS test_step_cache_reuses_and_resets")


def test_generation_queue_fifo_edit_and_pause():
    import threading
    import time
    from utils.generation_queue import GenerationQueue

    queue = GenerationQueue()
    first = queue.enqueue({"values": {"prompt": "first", "steps": 20}})
    second = queue.enqueue({"values": {"prompt": "second", "steps": 25}})
    queue.update(second, {"prompt": "edited", "steps": 31})
    queue.promote(second)
    job = queue.start_next()
    assert job.id == second and job.snapshot["values"]["prompt"] == "edited"
    try:
        queue.update(second, {"prompt": "too late"})
    except ValueError:
        pass
    else:
        raise AssertionError("运行中的参数不得编辑")
    queue.finish_current("done")
    assert queue.start_next().id == first

    queue.set_paused(True)
    entered = threading.Event()
    finished = threading.Event()
    def waiter():
        entered.set()
        queue.wait_if_paused()
        finished.set()
    thread = threading.Thread(target=waiter)
    thread.start()
    entered.wait(1)
    time.sleep(0.05)
    assert not finished.is_set(), "暂停必须等待"
    queue.set_paused(False)
    thread.join(1)
    assert finished.is_set(), "恢复必须唤醒等待线程"
    print("PASS test_generation_queue_fifo_edit_and_pause")


def test_queue_controller_dispatches_and_batch_bypasses():
    from PyQt6.QtWidgets import QWidget
    from utils.generation_queue_controller import GenerationQueueMixin

    app = _qapp()

    class Host(QWidget, GenerationQueueMixin):
        def __init__(self):
            super().__init__()
            self.is_generating = False
            self.cancel_flag = False
            self.calls = 0

        def _start_generation_immediate(self):
            self.calls += 1
            self.is_generating = True

    host = Host()
    host.start_generation()
    host.start_generation()
    assert host.calls == 1 and host._generation_queue.pending_count() == 1
    host.is_generating = False
    host._generation_queue_tick()
    assert host.calls == 2
    host.is_generating = False
    host._generation_queue_tick()
    host._batch_mode = True
    host.start_generation()
    assert host.calls == 3 and host._generation_queue.pending_count() == 0
    host._generation_queue_timer.stop()
    host.close()
    print("PASS test_queue_controller_dispatches_and_batch_bypasses")


def test_model_preview_naming_and_priority():
    from pathlib import Path
    import tempfile
    from utils.model_previews import find_model_preview

    root = Path("tests/artifacts/tmp")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        model = Path(tmp) / "hero.safetensors"
        model.touch()
        plain = Path(tmp) / "hero.png"
        preview = Path(tmp) / "hero.preview.png"
        other = Path(tmp) / "other.png"
        plain.touch()
        preview.touch()
        other.touch()
        assert find_model_preview(str(model)) == str(preview)
        preview.unlink()
        assert find_model_preview(str(model)) == str(plain)
        assert find_model_preview(str(Path(tmp) / "missing.safetensors")) is None
    print("PASS test_model_preview_naming_and_priority")


def test_gallery_text_and_cosine_search():
    from utils.gallery_search import cosine_rank, match_all_terms

    assert match_all_terms("Red 少女", "hero.png", "少女, red dress, cinematic")
    assert not match_all_terms("red blue", "hero.png", "red dress")
    assert match_all_terms("", "hero.png", "")
    scores = cosine_rank([1.0, 0.0], {"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert list(scores) == ["a", "b"] and scores["a"] > scores["b"]
    print("PASS test_gallery_text_and_cosine_search")


def test_clip_index_reuses_and_invalidates():
    from pathlib import Path
    import tempfile
    from utils.gallery_search import ClipSearchEngine

    class FakeEncoder:
        def __init__(self):
            self.image_calls = 0
        def encode_text(self, text):
            return [1.0, 0.0]
        def encode_image(self, path):
            self.image_calls += 1
            return [1.0, 0.0]

    root = Path("tests/artifacts/tmp")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        image = Path(tmp) / "one.png"
        image.write_bytes(b"one")
        encoder = FakeEncoder()
        engine = ClipSearchEngine(index_path=str(Path(tmp) / "index.json"), encoder=encoder)
        engine.search_text("one", [str(image)])
        engine.search_text("one", [str(image)])
        assert encoder.image_calls == 1, "未变化图片应复用向量"
        image.write_bytes(b"changed")
        engine.search_text("one", [str(image)])
        assert encoder.image_calls == 2, "文件变化必须重新编码"
    print("PASS test_clip_index_reuses_and_invalidates")


def test_clip_missing_local_model_is_explicit():
    from utils.gallery_search import ClipSearchEngine
    try:
        ClipSearchEngine(model_path="models/clip/definitely-missing").encode_text("test")
    except RuntimeError as exc:
        assert "CLIP" in str(exc) and "本地" in str(exc)
    else:
        raise AssertionError("显式 CLIP 目录不存在时必须给出错误")
    print("PASS test_clip_missing_local_model_is_explicit")


def test_queue_refresh_preserves_unsaved_edits():
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QWidget
    from ui.generation_queue_dialog import GenerationQueueDialog
    from utils.generation_queue import GenerationQueue

    app = _qapp()

    class Host(QWidget):
        def __init__(self):
            super().__init__()
            self.queue = GenerationQueue()
            self._generation_queue_timer = QTimer(self)

        def _generation_queue_state(self):
            return self.queue

        def toggle_generation_queue_pause(self):
            self.queue.set_paused(not self.queue.paused)

    host = Host()
    host.queue.enqueue({"values": {"txt_prompt": "original"}})
    dialog = GenerationQueueDialog(host)
    dialog.txt_prompt.setPlainText("unsaved edit")
    dialog.refresh()
    assert dialog.txt_prompt.toPlainText() == "unsaved edit", \
        "定时刷新不得覆盖尚未保存的参数编辑"
    dialog._timer.stop()
    dialog.close()
    host.close()
    print("PASS test_queue_refresh_preserves_unsaved_edits")


def test_gallery_semantic_search_uses_window_clip_config():
    from types import SimpleNamespace
    from unittest.mock import patch
    from PyQt6.QtWidgets import QWidget
    from ui.gallery_panel import GalleryPanel

    app = _qapp()
    host = QWidget()
    host.config = SimpleNamespace(clip_model_path="models/clip/custom")
    container = QWidget(host)
    panel = GalleryPanel(container)
    panel._all_items = [("one.png", "", False)]
    panel.search_box.setText("portrait")
    captured = {}

    class FakeEngine:
        def __init__(self, model_path=None):
            captured["model_path"] = model_path

        def search_text(self, query, paths):
            return {"one.png": 1.0}

    class ImmediateThread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

    with patch("utils.gallery_search.ClipSearchEngine", FakeEngine), \
            patch("threading.Thread", ImmediateThread):
        panel.start_semantic_search()
    assert captured["model_path"] == "models/clip/custom"
    panel.close()
    host.close()
    print("PASS test_gallery_semantic_search_uses_window_clip_config")


def test_gallery_context_menu_can_find_similar_images():
    from unittest.mock import patch
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QListWidgetItem, QMenu
    from ui.gallery_panel import GalleryPanel

    app = _qapp()
    panel = GalleryPanel()
    item = QListWidgetItem("one")
    item.setData(Qt.ItemDataRole.UserRole, "one.png")
    panel.list_widget.addItem(item)
    panel.list_widget.setCurrentItem(item)
    calls = []
    panel.start_semantic_search = lambda image_path="": calls.append(image_path)

    def choose_similar(menu, *_):
        return next((action for action in menu.actions()
                     if action.text() == "查找相似图"), None)

    with patch.object(QMenu, "exec", choose_similar):
        panel._show_menu(panel.list_widget.rect().center())
    assert calls == ["one.png"], "图片右键菜单必须提供相似图检索"
    panel.close()
    print("PASS test_gallery_context_menu_can_find_similar_images")


def test_gallery_query_change_rearms_semantic_button():
    from ui.gallery_panel import GalleryPanel

    app = _qapp()
    panel = GalleryPanel()
    panel.btn_semantic.setEnabled(False)
    panel.btn_semantic.setText("索引中…")
    panel._search_text_changed()
    assert panel.btn_semantic.isEnabled(), "修改查询后必须允许重新检索"
    assert panel.btn_semantic.text() == "语义"
    panel.close()
    print("PASS test_gallery_query_change_rearms_semantic_button")


def test_settings_saves_local_clip_directory():
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QWidget
    from ui.settings_dialog import SettingsDialog

    app = _qapp()
    host = QWidget()
    host.config = SimpleNamespace(ui_language="zh", save=lambda: None)
    dialog = SettingsDialog(host)
    dialog.txt_clip_model_path.setText("models/clip/local")
    dialog.accept()
    assert host.config.clip_model_path == "models/clip/local"
    host.close()
    print("PASS test_settings_saves_local_clip_directory")


if __name__ == "__main__":
    tests = [
        test_quantization_mapping_and_signature,
        test_group_offload_marks_pipe,
        test_quantization_backend_error_is_explicit,
        test_step_cache_reuses_and_resets,
        test_generation_queue_fifo_edit_and_pause,
        test_queue_controller_dispatches_and_batch_bypasses,
        test_model_preview_naming_and_priority,
        test_gallery_text_and_cosine_search,
        test_clip_index_reuses_and_invalidates,
        test_clip_missing_local_model_is_explicit,
        test_queue_refresh_preserves_unsaved_edits,
        test_gallery_semantic_search_uses_window_clip_config,
        test_gallery_context_menu_can_find_similar_images,
        test_gallery_query_change_rearms_semantic_button,
        test_settings_saves_local_clip_directory,
    ]
    for fn in tests:
        fn()
    print(f"\nINFERENCE_PRODUCTIVITY_OK: {len(tests)} tests")
