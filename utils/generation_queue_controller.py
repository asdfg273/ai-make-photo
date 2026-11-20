"""Qt 主线程的生成队列调度；实际等待逻辑在 generation_queue.py。"""
from __future__ import annotations

import copy
import logging

logger = logging.getLogger(__name__)

_SNAPSHOT_CONTROLS = (
    "txt_prompt", "txt_neg", "spin_steps", "scale_cfg", "combo_res",
    "spin_count", "spin_seed", "combo_sampler", "combo_scheduler",
    "scale_str", "scale_strength",
    "combo_trans_mode", "combo_enhance_mode", "chk_use_ipa", "spin_ipa_scale",
    "chk_translate_prompt",
    "combo_ipa_variant", "chk_pose_transfer", "slider_pt_cn", "chk_auto_features",
    "chk_reference_only", "scale_ref_fidelity", "chk_use_pose", "combo_cn_type",
    "scale_cn_strength", "chk_use_adetailer", "combo_ad_target",
    "combo_adetailer_model", "scale_adetailer_strength", "chk_use_ad_hand",
    "combo_ad_hand", "scale_ad_hand", "scale_ad_hand_blend", "chk_freeu",
    "chk_hires", "combo_hires_scale", "scale_hires_denoise",
    "scale_hires_tile_denoise",
    "combo_hires_upscaler", "combo_hires_mode", "combo_hires_order",
    "chk_make_comic", "chk_use_preview", "spin_preview_interval",
)
_SNAPSHOT_PATHS = (
    "ref_image_path", "mask_image_path", "pose_image_path", "ipa_image_path",
    "_current_page_id",
)
# 区域提示词与多参考图**存在 config 里**（`regions()` / `ipa_refs()` 都读
# `self.config`），不是宿主属性，所以必须在快照里单独带一份。
# 不带的话：排两张图、中途改了区域或多参考图，第二张派发时用的是
# **第一张之后改的那一份**，而第一张再也不会回到它自己的设置。
_SNAPSHOT_CONFIG = ("use_regions", "regions", "use_ipa_refs", "ipa_refs")


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
            # 排队前用户自己在界面上的一套设置；队列跑完要还回去
            self._generation_queue_user_snapshot = None
            self._generation_queue_last_snapshot = None
            self._generation_queue_dispatched = 0
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
        config_values = {}
        config = getattr(self, "config", None)
        if config is not None:
            for name in _SNAPSHOT_CONFIG:
                if hasattr(config, name):
                    config_values[name] = copy.deepcopy(getattr(config, name))
        return {
            "values": values,
            "model_data": copy.deepcopy(model_data.currentData()) if model_data else None,
            "model_type": copy.deepcopy(model_type.currentData()) if model_type else None,
            "loras": loras,
            "paths": copy.deepcopy(paths),
            "config": config_values,
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
            # ⚠️ FloatSlider 是 QWidget 子类，**不是 QSlider**：它只有
            # `float_value()` / `setValue(float)`，没有 `set_value()`。
            # 以前这里只写了 `set_value` 分支，于是 CFG、重绘强度、CN 强度、
            # ADetailer 强度、手部融合、高清重绘幅度……所有这些滑杆
            # **从来不会被队列快照还原**：排队的第二张及以后全部沿用
            # "当前界面上那一组滑杆值"，而且不报错。
            elif hasattr(widget, "float_value") and hasattr(widget, "setValue"):
                widget.setValue(float(value))
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
        config_values = snapshot.get("config") or {}
        config = getattr(self, "config", None)
        if config is not None and config_values:
            for name, value in config_values.items():
                try:
                    setattr(config, name, copy.deepcopy(value))
                except Exception as exc:
                    logger.warning("队列快照还原 config.%s 失败: %s", name, exc)
            # 两个摘要标签是 config 的投影，改完 config 要刷新，
            # 否则界面上显示的还是上一张图的区域/参考图。
            for refresh in ("_refresh_regions_summary", "_refresh_ipa_refs_summary"):
                callback = getattr(self, refresh, None)
                if callback is None:
                    continue
                try:
                    callback()
                except Exception as exc:
                    logger.debug("刷新摘要标签失败 %s: %s", refresh, exc)

    def _generation_queue_tick(self):
        queue = self._generation_queue_state()
        if getattr(self, "_generation_queue_dispatching", False):
            return
        current = queue.current
        if current is not None:
            thread = getattr(self, "_gen_thread", None)
            if getattr(self, "is_generating", False) or (thread and thread.is_alive()):
                return
            # 按生成线程自己记的结论归档：异常退出以前会被标成 done，
            # 弹窗显示成功但实际没出图。
            outcome = getattr(self, "_gen_outcome", None)
            if outcome == "failed":
                queue.finish_current("failed", getattr(self, "_gen_outcome_error", "") or "生成失败")
            elif outcome == "cancelled" or (
                    outcome is None and getattr(self, "cancel_flag", False)):
                queue.finish_current("cancelled")
            else:
                queue.finish_current("done")
        if getattr(self, "is_generating", False):
            return
        job = queue.start_next()
        if job is None:
            self._generation_queue_timer.stop()
            self._refresh_generation_queue_dialog()
            self._restore_queue_resting_state()
            return
        self._generation_queue_dispatching = True
        try:
            # 第一次派发前先把"用户当前界面"存一份，跑完还回去。
            if getattr(self, "_generation_queue_user_snapshot", None) is None:
                self._generation_queue_user_snapshot = \
                    self._capture_generation_snapshot()
            self._generation_queue_last_snapshot = job.snapshot
            self._generation_queue_dispatched = \
                int(getattr(self, "_generation_queue_dispatched", 0) or 0) + 1
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

    def _restore_queue_resting_state(self) -> bool:
        """队列跑完，把界面还原成"排队前用户自己那套"。

        为什么需要：`_generation_queue_tick` 每派发一张都要把该任务的快照
        写进控件，否则第二张会沿用第一张之后的界面值。副作用是队列结束时
        界面停在**最后一张**的参数上，用户排队前自己调好的设置就没了，
        而且没有任何提示（用户实测反馈）。

        为什么带"用户改过就别动"的守卫：队列运行期间用户完全可能在界面上
        调参数（准备下一轮）。这时还原排队前的快照反而会覆盖他的新设置，
        所以只在"控件值 === 最后一张任务的值"（说明没人碰过）时才还原。
        """
        user = getattr(self, "_generation_queue_user_snapshot", None)
        last = getattr(self, "_generation_queue_last_snapshot", None)
        dispatched = int(getattr(self, "_generation_queue_dispatched", 0) or 0)
        self._generation_queue_user_snapshot = None
        self._generation_queue_last_snapshot = None
        self._generation_queue_dispatched = 0
        if not user:
            return False
        if last is not None:
            try:
                current = self._capture_generation_snapshot()
            except Exception as exc:
                logger.debug("队列结束时读取界面状态失败: %s", exc)
                current = None
            if current is not None and current != last:
                logger.info("队列结束：界面在排队期间被改动过，保留用户当前设置")
                return False
        try:
            self._restore_generation_snapshot(user)
        except Exception as exc:
            logger.warning("队列结束后还原界面失败: %s", exc)
            return False
        # 只排了一张就没必要打扰用户（这时本来就等于没动过界面）
        if dispatched >= 2:
            status = getattr(self, "_bridge", None)
            if status is not None:
                status.status_signal.emit(
                    "队列已完成，界面已恢复到排队前的设置", "#89dceb")
        return True

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
