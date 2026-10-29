"""Qt 主线程的生成队列调度；实际等待逻辑在 generation_queue.py。"""
from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

_SNAPSHOT_CONTROLS = (
    "txt_prompt", "txt_neg", "spin_steps", "scale_cfg", "combo_res",
    "spin_count", "spin_seed", "combo_sampler", "scale_str", "scale_strength",
    "combo_trans_mode", "combo_enhance_mode", "chk_use_ipa", "spin_ipa_scale",
    "combo_ipa_variant", "chk_pose_transfer", "slider_pt_cn", "chk_auto_features",
    "chk_reference_only", "scale_ref_fidelity", "chk_use_pose", "combo_cn_type",
    "scale_cn_strength", "chk_use_adetailer", "combo_ad_target",
    "combo_adetailer_model", "scale_adetailer_strength", "chk_use_ad_hand",
    "combo_ad_hand", "scale_ad_hand_strength", "scale_ad_hand_blend", "chk_freeu",
    "chk_hires", "combo_hires_scale", "scale_hires_denoise",
    "combo_hires_upscaler", "combo_hires_mode", "combo_hires_order",
    "chk_make_comic", "chk_use_preview", "spin_preview_interval",
)
_SNAPSHOT_PATHS = (
    "ref_image_path", "mask_image_path", "pose_image_path", "ipa_image_path",
    "_current_page_id",
)


class GenerationQueueMixin:
    """宿主还需继承 GenerationMixin，并把本 Mixin 放在它前面。"""

    def _generation_queue_state(self):
        from PyQt6.QtCore import QTimer
        from utils.generation_queue import GenerationQueue
        queue = getattr(self, "_generation_queue", None)
        if queue is None:
            queue = self._generation_queue = GenerationQueue()
            self._generation_queue_dialog = None
            self._generation_queue_timer = QTimer(self)
            self._generation_queue_timer.setInterval(250)
            self._generation_queue_timer.timeout.connect(self._generation_queue_tick)
            self._generation_queue_dispatching = False
        return queue

    def start_generation(self):
        # 角色批量已有专用串行调度器，不能再次进入普通 FIFO。
        if getattr(self, "_batch_mode", False):
            return self._start_generation_immediate()
        queue = self._generation_queue_state()
        snapshot = self._capture_generation_snapshot()
        queue.enqueue(snapshot)
        self._generation_queue_timer.start()
        if hasattr(self, "_bridge"):
            self._bridge.status_signal.emit(
                f"已加入生成队列（等待 {queue.pending_count()} 项）", "#89dceb")
        self._refresh_generation_queue_dialog()
        # 空闲时立即派发，不必等第一个 250ms 周期。
        if not getattr(self, "is_generating", False) and not queue.paused:
            self._generation_queue_tick()

    def _capture_generation_snapshot(self) -> dict:
        values = {}
        for name in _SNAPSHOT_CONTROLS:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if hasattr(widget, "toPlainText"):
                value = widget.toPlainText()
            elif hasattr(widget, "isChecked"):
                value = widget.isChecked()
            elif hasattr(widget, "currentText") and hasattr(widget, "count"):
                value = widget.currentText()
            elif hasattr(widget, "float_value"):
                value = widget.float_value()
            elif hasattr(widget, "value"):
                value = widget.value()
            else:
                continue
            values[name] = copy.deepcopy(value)
        model_data = getattr(self, "combo_model", None)
        model_type = getattr(self, "combo_model_type", None)
        loras = [combo.currentText() for combo in (getattr(self, "combo_loras", None) or [])]
        paths = {name: getattr(self, name, None) for name in _SNAPSHOT_PATHS}
        return {
            "values": values,
            "model_data": copy.deepcopy(model_data.currentData()) if model_data else None,
            "model_type": copy.deepcopy(model_type.currentData()) if model_type else None,
            "loras": loras,
            "paths": copy.deepcopy(paths),
        }

    def _restore_generation_snapshot(self, snapshot: dict):
        from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                                     QSlider, QSpinBox, QTextEdit)
        values = snapshot.get("values", {})
        for name, value in values.items():
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if isinstance(widget, QTextEdit):
                widget.setPlainText(str(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
                widget.setValue(value)
            elif hasattr(widget, "set_value"):
                widget.set_value(float(value))
        combo_type = getattr(self, "combo_model_type", None)
        if combo_type is not None and snapshot.get("model_type") is not None:
            index = combo_type.findData(snapshot["model_type"])
            if index >= 0:
                combo_type.setCurrentIndex(index)
        combo_model = getattr(self, "combo_model", None)
        if combo_model is not None and snapshot.get("model_data") is not None:
            index = combo_model.findData(snapshot["model_data"])
            if index >= 0:
                combo_model.setCurrentIndex(index)
        for combo, text in zip(getattr(self, "combo_loras", None) or [],
                               snapshot.get("loras", [])):
            index = combo.findText(text)
            if index >= 0:
                combo.setCurrentIndex(index)
        for name, value in snapshot.get("paths", {}).items():
            setattr(self, name, value)

    def _generation_queue_tick(self):
        queue = self._generation_queue_state()
        if getattr(self, "_generation_queue_dispatching", False):
            return
        current = queue.current
        if current is not None:
            thread = getattr(self, "_gen_thread", None)
            if getattr(self, "is_generating", False) or (thread and thread.is_alive()):
                return
            queue.finish_current("cancelled" if getattr(self, "cancel_flag", False) else "done")
        if getattr(self, "is_generating", False):
            return
        job = queue.start_next()
        if job is None:
            self._generation_queue_timer.stop()
            self._refresh_generation_queue_dialog()
            return
        self._generation_queue_dispatching = True
        try:
            self._restore_generation_snapshot(job.snapshot)
            self._start_generation_immediate()
            if not getattr(self, "is_generating", False):
                queue.finish_current("failed", "参数校验未通过")
        except Exception as exc:
            logger.exception("队列任务派发失败")
            queue.finish_current("failed", str(exc))
        finally:
            self._generation_queue_dispatching = False
        self._refresh_generation_queue_dialog()

    def _generation_queue_wait(self) -> bool:
        queue = getattr(self, "_generation_queue", None)
        if queue is None or queue.current is None:
            return True
        return queue.wait_if_paused(lambda: bool(getattr(self, "cancel_flag", False)))

    def toggle_generation_queue_pause(self):
        queue = self._generation_queue_state()
        queue.set_paused(not queue.paused)
        if hasattr(self, "_bridge"):
            text = "队列将在当前采样步后暂停" if queue.paused else "队列已恢复"
            self._bridge.status_signal.emit(text, "#f9e2af")
        self._generation_queue_timer.start()
        self._refresh_generation_queue_dialog()

    def show_generation_queue(self):
        from ui.generation_queue_dialog import GenerationQueueDialog
        self._generation_queue_state()
        dialog = getattr(self, "_generation_queue_dialog", None)
        if dialog is None:
            dialog = self._generation_queue_dialog = GenerationQueueDialog(self)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.refresh()

    def _refresh_generation_queue_dialog(self):
        dialog = getattr(self, "_generation_queue_dialog", None)
        if dialog is not None:
            dialog.refresh()
