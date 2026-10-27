# ui/preset_manager.py
"""
🎯 预设管理 Mixin
- 加载/保存用户预设
- 右键菜单（保存/删除/导入/导出）
- 收集当前 UI 状态为预设
"""
import json
import logging
from PyQt6.QtWidgets import (
    QInputDialog, QMessageBox, QFileDialog, QMenu
)
from ui.tooltips import tip
from ui.i18n import ui_i18n
logger = logging.getLogger(__name__)


class PresetManagerMixin:
    """预设管理功能 Mixin。需要宿主类提供:
    - self.combo_preset / self.txt_prompt / self.txt_neg / self.spin_steps ...
    - self.apply_preset / self._set_status
    """

    # ───────────────────── 工具方法 ─────────────────────

    def _refresh_preset_combo(self, select_name: str = None):
        """重新加载 combo_preset 的选项"""
        if not hasattr(self, "combo_preset"):
            logger.warning("⚠️ combo_preset 尚未创建，跳过 _refresh_preset_combo")
            return
        from core.presets import get_all_presets, is_builtin_preset

        current = select_name or self._get_current_preset_name()

        # 断开信号避免触发 apply_preset
        try:
            self.combo_preset.currentIndexChanged.disconnect(self.apply_preset)
        except TypeError:
            pass

        self.combo_preset.clear()
        self.combo_preset.addItem("（无）", userData="（无）")

        all_presets = get_all_presets()
        builtin_names = [n for n in all_presets if is_builtin_preset(n)]
        user_names    = [n for n in all_presets if not is_builtin_preset(n)]

        for n in builtin_names:
            self.combo_preset.addItem(f"🎯 {n}", userData=n)

        if user_names:
            self.combo_preset.insertSeparator(self.combo_preset.count())
            for n in user_names:
                self.combo_preset.addItem(f"⭐ {n}", userData=n)

        # 恢复选择
        idx = self.combo_preset.findData(current)
        if idx >= 0:
            self.combo_preset.setCurrentIndex(idx)

        # 重连信号
        self.combo_preset.currentIndexChanged.connect(self.apply_preset)

    def _get_current_preset_name(self) -> str:
        """从 combo_preset 取真实预设名（去掉🎯/⭐前缀）"""
        if not hasattr(self, "combo_preset"):
            return ""
        data = self.combo_preset.currentData()
        if data:
            return data
        txt = self.combo_preset.currentText()
        for prefix in ("🎯 ", "⭐ "):
            if txt.startswith(prefix):
                return txt[len(prefix):]
        return txt

    # ───────────────────── 收集当前状态 ─────────────────────

    def _get_float(self, widget):
        """兼容 FloatSlider / QDoubleSpinBox"""
        for attr in ("float_value", "value"):
            if hasattr(widget, attr):
                v = getattr(widget, attr)
                return v() if callable(v) else v
        return None

    def _collect_current_state_as_preset(self) -> dict:
        """从当前 UI 状态构造一个预设字典"""
        preset = {
            "p": self.txt_prompt.toPlainText().strip(),
            "n": self.txt_neg.toPlainText().strip(),
            "params": {
                "steps":      self.spin_steps.value(),
                "cfg":        self._get_float(self.scale_cfg),
                "sampler":    self.combo_sampler.currentText(),
                "resolution": self.combo_res.currentText(),
            },
        }

        if hasattr(self, "scale_strength"):
            try:
                preset["strength"] = self._get_float(self.scale_strength)
            except Exception:
                pass

        # ADetailer 修脸（区分动漫/真人）
        if getattr(self, "chk_use_adetailer", None) and self.chk_use_adetailer.isChecked():
            preset["adetailer_face"] = {
                "enabled":  True,
                "model":    self.combo_adetailer_model.currentText(),
                "target":   self.combo_ad_target.currentText(),
                "strength": self._get_float(self.scale_adetailer_strength),
            }

        # ADetailer 修手
        if getattr(self, "chk_use_ad_hand", None) and self.chk_use_ad_hand.isChecked():
            preset["adetailer_hand"] = {
                "enabled":  True,
                "target":   self.combo_ad_hand.currentText(),
                "strength": self._get_float(self.scale_ad_hand),
                "blend":    self._get_float(self.scale_ad_hand_blend),
            }

        # FreeU
        if getattr(self, "chk_freeu", None) and self.chk_freeu.isChecked():
            preset["freeu"] = {"enabled": True}

        # Hires.fix
        if getattr(self, "chk_hires", None) and self.chk_hires.isChecked():
            preset["hires"] = {
                "enabled":  True,
                "scale":    self.combo_hires_scale.currentText(),
                "denoise":  self._get_float(self.scale_hires_denoise),
                "upscaler": self.combo_hires_upscaler.currentText(),
            }

        # ControlNet
        if getattr(self, "chk_use_pose", None) and self.chk_use_pose.isChecked():
            preset["controlnet"] = {
                "enabled":  True,
                "type":     self.combo_cn_type.currentText(),
                "strength": self._get_float(self.scale_cn_strength),
            }

        return preset

    # ───────────────────── 业务方法 ─────────────────────

    def save_current_as_preset(self):
        """💾 把当前控件状态保存为用户预设"""
        from core.presets import load_user_presets, save_user_presets, is_builtin_preset, PROMPT_PRESETS

        name, ok = QInputDialog.getText(
            self, ui_i18n.tr("保存预设"),
            ui_i18n.tr("请输入预设名称（不能与内置同名）:"), text=""
        )
        name = name.strip()
        if not ok or not name:
            return

        if is_builtin_preset(name):
            QMessageBox.warning(
                self, ui_i18n.tr("名称冲突"),
                ui_i18n.trf("「{name}」是内置预设，不能覆盖。\n建议加前缀，如「我的-{name}」。",
                            name=name))
            return

        user = load_user_presets()
        if name in user:
            ret = QMessageBox.question(
                self, ui_i18n.tr("已存在"),
                ui_i18n.trf("用户预设「{name}」已存在，是否覆盖？", name=name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        preset = self._collect_current_state_as_preset()
        user[name] = preset

        if save_user_presets(user):
            PROMPT_PRESETS[name] = preset
            self._refresh_preset_combo(select_name=name)
            self._set_status(ui_i18n.trf("已保存预设「{name}」", name=name), "#a6e3a1")
            logger.info(f"💾 保存用户预设: {name}")
        else:
            QMessageBox.critical(
                self, ui_i18n.tr("保存失败"), ui_i18n.tr("写入 user_presets.json 失败。"))

    def delete_current_preset(self):
        """🗑️ 删除当前选中的用户预设"""
        from core.presets import load_user_presets, save_user_presets, is_builtin_preset, PROMPT_PRESETS

        name = self._get_current_preset_name()
        if not name or name == "（无）":
            return
        if is_builtin_preset(name):
            QMessageBox.warning(
                self, ui_i18n.tr("无法删除"),
                ui_i18n.trf("「{name}」是内置预设，不能删除。", name=name))
            return

        ret = QMessageBox.question(
            self, ui_i18n.tr("删除预设"),
            ui_i18n.trf("确定要删除用户预设「{name}」吗？\n此操作不可撤销。", name=name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        user = load_user_presets()
        user.pop(name, None)
        PROMPT_PRESETS.pop(name, None)

        if save_user_presets(user):
            self._refresh_preset_combo(select_name="（无）")
            self._set_status(ui_i18n.trf("已删除预设「{name}」", name=name), "#fab387")
            logger.info(f"🗑️ 删除用户预设: {name}")

    def export_user_presets(self):
        """📤 导出用户预设到外部 JSON"""
        from core.presets import load_user_presets

        user = load_user_presets()
        if not user:
            QMessageBox.information(
                self, ui_i18n.tr("导出"), ui_i18n.tr("当前没有用户预设可导出。"))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, ui_i18n.tr("导出用户预设"), "my_presets.json", "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(user, f, ensure_ascii=False, indent=2)
            self._set_status(ui_i18n.trf("已导出 {count} 个预设", count=len(user)), "#a6e3a1")
            logger.info(f"📤 导出预设到 {path}")
        except Exception as e:
            QMessageBox.critical(self, ui_i18n.tr("导出失败"), str(e))

    def import_user_presets(self):
        """📥 从外部 JSON 导入用户预设"""
        from core.presets import load_user_presets, save_user_presets, is_builtin_preset, PROMPT_PRESETS

        path, _ = QFileDialog.getOpenFileName(
            self, ui_i18n.tr("导入用户预设"), "", "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                new_data = json.load(f)
            if not isinstance(new_data, dict):
                raise ValueError("根节点必须是字典")
        except Exception as e:
            QMessageBox.critical(
                self, ui_i18n.tr("导入失败"), ui_i18n.trf("读取文件失败:\n{error}", error=e))
            return

        skipped = [n for n in new_data if is_builtin_preset(n)]
        valid   = {n: v for n, v in new_data.items() if not is_builtin_preset(n)}

        if not valid:
            QMessageBox.warning(
                self, ui_i18n.tr("无可导入项"), ui_i18n.tr("所有预设都与内置同名，已跳过。"))
            return

        user = load_user_presets()
        overlap = set(user) & set(valid)
        if overlap:
            preview = ", ".join(list(overlap)[:5]) + ("..." if len(overlap) > 5 else "")
            ret = QMessageBox.question(
                self, ui_i18n.tr("名称冲突"),
                ui_i18n.trf("有 {count} 个预设同名：\n{preview}\n\n是否覆盖？\n（选'否'跳过冲突项）",
                            count=len(overlap), preview=preview),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            if ret == QMessageBox.StandardButton.Cancel:
                return
            if ret == QMessageBox.StandardButton.No:
                valid = {n: v for n, v in valid.items() if n not in overlap}

        user.update(valid)
        PROMPT_PRESETS.update(valid)

        if save_user_presets(user):
            self._refresh_preset_combo()
            msg = ui_i18n.trf("已导入 {count} 个预设", count=len(valid))
            if skipped:
                msg += ui_i18n.trf("（跳过 {count} 个内置同名）", count=len(skipped))
            self._set_status(msg, "#a6e3a1")
            logger.info(msg)

    def show_preset_menu(self, pos):
        """combo_preset 右键菜单"""
        from core.presets import is_builtin_preset

        menu = QMenu(self)

        act_save = menu.addAction("💾 保存当前参数为新预设")
        act_save.triggered.connect(self.save_current_as_preset)

        name = self._get_current_preset_name()
        if name and name != "（无）" and not is_builtin_preset(name):
            act_del = menu.addAction(f"🗑️ 删除「{name}」")
            act_del.triggered.connect(self.delete_current_preset)

        menu.addSeparator()
        menu.addAction("📤 导出用户预设...").triggered.connect(self.export_user_presets)
        menu.addAction("📥 导入用户预设...").triggered.connect(self.import_user_presets)
        menu.addSeparator()
        menu.addAction("🔄 重新加载预设").triggered.connect(lambda: self._refresh_preset_combo())

        menu.exec(self.combo_preset.mapToGlobal(pos))

class TooltipMixin:
    def apply_tooltips(self):
        """统一给所有控件加中文气泡，调用一次即可"""
        mapping = {
            # 控件名: tooltips key
            "spin_steps":               "steps",
            "scale_cfg":                "cfg",
            "combo_sampler":            "sampler",
            "combo_res":                "resolution",
            "spin_count":               "count",
            "spin_seed":                "seed",
            "txt_prompt":               "prompt_positive",
            "txt_neg":                  "prompt_negative",
            "combo_device":             "device",
            # img2img
            "scale_strength":           "strength",
            # Hires
            "combo_hires_scale":        "hires_scale",
            "scale_hires_denoise":      "hires_denoise",
            "combo_hires_upscaler":     "hires_upscaler",
            # ADetailer
            "scale_adetailer_strength": "adetailer_strength",
            "combo_adetailer_model":    "adetailer_model",
            # ControlNet
            "scale_cn_strength":        "cn_strength",
            "combo_cn_type":            "cn_type",
        }
        applied = 0
        for attr, key in mapping.items():
            w = getattr(self, attr, None)
            if w is not None:
                w.setToolTip(tip(key))
                applied += 1
        logger.info(f"[Tooltip] 已应用 {applied}/{len(mapping)} 条参数提示")

