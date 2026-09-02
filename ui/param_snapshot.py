# ui/param_snapshot.py
# ============================================================
#  预设快照/还原辅助 Mixin — 从 ui_builder.py 原样迁出
#  被 main.py / preset_manager.py 通过 self.xxx 调用
# ============================================================
import logging

logger = logging.getLogger(__name__)


class ParamSnapshotMixin:
    """预设参数快照、还原、diff、闪烁提示等 UI 辅助方法。"""

    _CONTROL_LABELS = {
        "spin_steps":                ("步数 Steps",        "🎨 基础"),
        "scale_cfg":                 ("CFG Scale",         "🎨 基础"),
        "combo_res":                 ("分辨率",            "🎨 基础"),
        "combo_sampler":             ("采样器",            "🎨 基础"),
        "spin_count":                ("生成数量",          "🎨 基础"),
        "spin_seed":                 ("种子",              "🎨 基础"),
        "scale_strength":            ("重绘强度",          "🖼 图生图"),
        "chk_use_ipa":               ("IP-Adapter",       "🖼 图生图"),
        "spin_ipa_scale":            ("IPA 影响力",        "🖼 图生图"),
        "combo_ipa_variant":         ("IPA 版本",          "🖼 图生图"),
        "chk_pose_transfer":         ("Pose Transfer",    "🖼 图生图"),
        "slider_pt_cn":              ("姿势约束",          "🖼 图生图"),
        "chk_auto_features":         ("自动提取特征",      "🖼 图生图"),
        "chk_reference_only":        ("Reference-Only",   "🖼 图生图"),
        "scale_ref_fidelity":        ("参考强度",          "🖼 图生图"),
        "chk_use_pose":              ("ControlNet",        "🕹 ControlNet"),
        "combo_cn_type":             ("CN 类型",           "🕹 ControlNet"),
        "scale_cn_strength":         ("CN 条件强度",       "🕹 ControlNet"),
        "chk_use_adetailer":         ("修脸",              "⚙️ 高级"),
        "combo_ad_target":           ("脸部检测目标",      "⚙️ 高级"),
        "combo_adetailer_model":     ("脸部模型",          "⚙️ 高级"),
        "scale_adetailer_strength":  ("脸部修复强度",      "⚙️ 高级"),
        "chk_use_ad_hand":           ("修手",              "⚙️ 高级"),
        "combo_ad_hand":             ("手部检测目标",      "⚙️ 高级"),
        "scale_ad_hand":             ("手部重绘强度",      "⚙️ 高级"),
        "scale_ad_hand_blend":       ("手部融合度",        "⚙️ 高级"),
        "chk_hires":                 ("Hires.fix",         "⚙️ 高级"),
        "combo_hires_scale":         ("放大倍率",          "⚙️ 高级"),
        "scale_hires_denoise":       ("Hires 降噪",        "⚙️ 高级"),
        "combo_hires_upscaler":      ("Upscaler",          "⚙️ 高级"),
        "txt_prompt":                ("正向提示词",        "🎨 基础"),
        "txt_neg":                   ("负向提示词",        "🎨 基础"),
    }

    # --- 安全写控件 ---
    def _safe_set_check(self, name, val):
        w = getattr(self, name, None)
        if w is None or val is None:
            return
        try:
            w.setChecked(bool(val))
        except Exception as e:
            logger.warning(f"[preset] setChecked {name} 失败: {e}")

    def _safe_set_combo(self, name, text):
        w = getattr(self, name, None)
        if w is None or text is None:
            return
        try:
            idx = w.findText(str(text))
            if idx >= 0:
                w.setCurrentIndex(idx)
            else:
                # 模糊匹配（比如 "plus" 命中 "plus (推荐)"）
                for i in range(w.count()):
                    if str(text).lower() in w.itemText(i).lower():
                        w.setCurrentIndex(i)
                        return
        except Exception as e:
            logger.warning(f"[preset] setCombo {name} 失败: {e}")

    def _safe_set_float(self, name, val):
        """适配 FloatSlider / QDoubleSpinBox / QSlider(整数*100)"""
        w = getattr(self, name, None)
        if w is None or val is None:
            return
        try:
            if hasattr(w, 'set_value'):
                w.set_value(float(val))
            elif hasattr(w, 'setValue'):
                from PyQt6.QtWidgets import QSlider
                if isinstance(w, QSlider):
                    w.setValue(int(round(float(val) * 100)))
                else:
                    w.setValue(float(val))
        except Exception as e:
            logger.warning(f"[preset] setFloat {name} 失败: {e}")

    def _safe_set_int(self, name, val):
        w = getattr(self, name, None)
        if w is None or val is None:
            return
        try:
            w.setValue(int(val))
        except Exception as e:
            logger.warning(f"[preset] setInt {name} 失败: {e}")

    def _read_float(self, w):
        """读取 FloatSlider / QDoubleSpinBox / QSlider 的当前值"""
        try:
            for m in ('value', 'get_value'):
                if hasattr(w, m):
                    v = getattr(w, m)()
                    return float(v)
        except Exception:
            pass
        return None

    # --- 读取控件当前值（统一接口） ---
    def _get_widget_value(self, name):
        from PyQt6.QtWidgets import (
            QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QSlider, QTextEdit
        )
        w = getattr(self, name, None)
        if w is None:
            return None
        try:
            if isinstance(w, QCheckBox):     return w.isChecked()
            if isinstance(w, QComboBox):     return w.currentText()
            if isinstance(w, (QSpinBox, QDoubleSpinBox)): return w.value()
            if isinstance(w, QSlider):       return w.value()
            if isinstance(w, QTextEdit):     return w.toPlainText()
            if hasattr(w, 'value'):          return w.value()  # FloatSlider
            if hasattr(w, 'get_value'):      return w.get_value()
        except Exception:
            pass
        return None

    # --- 快照：应用预设前备份当前参数，方便"还原" ---
    def _snapshot_current_params(self):
        """把当前所有可被预设修改的参数存到 self._preset_backup"""
        try:
            self._preset_backup = {
                "prompt": self.txt_prompt.toPlainText(),
                "neg":    self.txt_neg.toPlainText(),
                "steps":  self.spin_steps.value(),
                "cfg":    self._read_float(self.scale_cfg),
                "res":    self.combo_res.currentText(),
                "sampler":self.combo_sampler.currentText(),
                "strength": self._read_float(self.scale_strength),
                "ad_face_on":  self.chk_use_adetailer.isChecked(),
                "ad_face_target": self.combo_ad_target.currentText(),
                "ad_face_model":  self.combo_adetailer_model.currentText(),
                "ad_face_str": self._read_float(self.scale_adetailer_strength),
                "ad_hand_on":  self.chk_use_ad_hand.isChecked(),
                "ad_hand_target": self.combo_ad_hand.currentText(),
                "ad_hand_str": self._read_float(self.scale_ad_hand),
                "ad_hand_blend": self._read_float(self.scale_ad_hand_blend),
                "hires_on":   self.chk_hires.isChecked(),
                "hires_scale":self.combo_hires_scale.currentText(),
                "hires_denoise": self._read_float(self.scale_hires_denoise),
                "hires_upscaler": self.combo_hires_upscaler.currentText(),
                "cn_on":     self.chk_use_pose.isChecked(),
                "cn_type":   self.combo_cn_type.currentText(),
                "cn_strength": self._read_float(self.scale_cn_strength),
                "ipa_on":    self.chk_use_ipa.isChecked(),
                "ipa_scale": self.spin_ipa_scale.value(),
                "ipa_variant": self.combo_ipa_variant.currentText(),
                "pt_on":     self.chk_pose_transfer.isChecked(),
                "pt_cn":     self.slider_pt_cn.value(),
                "auto_features": self.chk_auto_features.isChecked(),
                "ref_only":  self.chk_reference_only.isChecked(),
                "ref_fidelity": self.scale_ref_fidelity.value(),
            }
        except Exception as e:
            logger.warning(f"[preset] 快照失败: {e}")
            self._preset_backup = None

    def _update_preset_badge(self, n: int, lines: list):
        """更新还原按钮旁边的徽章 + tooltip"""
        if hasattr(self, "lbl_preset_badge"):
            if n > 0:
                self.lbl_preset_badge.setText(f"● {n} 项已改")
            else:
                self.lbl_preset_badge.setText("")

        if hasattr(self, "btn_restore_preset"):
            if n > 0:
                import re
                plain_lines = []
                for ln in lines:
                    txt = re.sub(r'<[^>]+>', '', ln).strip()
                    plain_lines.append(txt)
                tip = ("<b>↩️ 点击还原以下改动：</b><br>"
                       + "<br>".join(plain_lines[:30]))
                if len(plain_lines) > 30:
                    tip += f"<br>...还有 {len(plain_lines)-30} 项"
                self.btn_restore_preset.setToolTip(tip)
            else:
                self.btn_restore_preset.setToolTip("还原到套用预设前的所有参数")

    # --- 还原预设前参数 ---
    def restore_preset_backup(self):
        bk = getattr(self, "_preset_backup", None)
        if not bk:
            self._set_status("⚠️ 没有可还原的快照", "#ff7a17")
            return
        try:
            self.txt_prompt.setPlainText(bk["prompt"])
            self.txt_neg.setPlainText(bk["neg"])
            self._safe_set_int("spin_steps", bk["steps"])
            self._safe_set_float("scale_cfg", bk["cfg"])
            self._safe_set_combo("combo_res", bk["res"])
            self._safe_set_combo("combo_sampler", bk["sampler"])
            self._safe_set_float("scale_strength", bk["strength"])
            self._safe_set_check("chk_use_adetailer", bk["ad_face_on"])
            self._safe_set_combo("combo_ad_target", bk["ad_face_target"])
            self._safe_set_combo("combo_adetailer_model", bk["ad_face_model"])
            self._safe_set_float("scale_adetailer_strength", bk["ad_face_str"])
            self._safe_set_check("chk_use_ad_hand", bk["ad_hand_on"])
            self._safe_set_combo("combo_ad_hand", bk["ad_hand_target"])
            self._safe_set_float("scale_ad_hand", bk["ad_hand_str"])
            self._safe_set_float("scale_ad_hand_blend", bk["ad_hand_blend"])
            self._safe_set_check("chk_hires", bk["hires_on"])
            self._safe_set_combo("combo_hires_scale", bk["hires_scale"])
            self._safe_set_float("scale_hires_denoise", bk["hires_denoise"])
            self._safe_set_combo("combo_hires_upscaler", bk["hires_upscaler"])
            self._safe_set_check("chk_use_pose", bk["cn_on"])
            self._safe_set_combo("combo_cn_type", bk["cn_type"])
            self._safe_set_float("scale_cn_strength", bk["cn_strength"])
            self._safe_set_check("chk_use_ipa", bk["ipa_on"])
            self.spin_ipa_scale.setValue(float(bk["ipa_scale"]))
            self._safe_set_combo("combo_ipa_variant", bk["ipa_variant"])
            self._safe_set_check("chk_pose_transfer", bk["pt_on"])
            self.slider_pt_cn.setValue(int(bk["pt_cn"]))
            self._safe_set_check("chk_auto_features", bk["auto_features"])
            self._safe_set_check("chk_reference_only", bk["ref_only"])
            self.scale_ref_fidelity.setValue(int(bk["ref_fidelity"]))
            self._toggle_adetailer(); self._toggle_ad_hand()
            self._toggle_hires(); self._toggle_cn()
        except Exception as e:
            self._set_status(f"⚠️ 还原失败: {e}", "#ff7a17")
            if hasattr(self, "lbl_preset_badge"):
                self.lbl_preset_badge.setText("")
            if hasattr(self, "btn_restore_preset"):
                self.btn_restore_preset.setToolTip("还原到套用预设前的所有参数")

    def _flash_widget(self, name, color="#dadbdf"):
        from PyQt6.QtWidgets import QGraphicsColorizeEffect
        from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
        from PyQt6.QtGui import QColor

        w = getattr(self, name, None)
        if w is None:
            return
        try:
            if w.graphicsEffect() is not None:
                return
            effect = QGraphicsColorizeEffect(w)
            effect.setColor(QColor(color))
            effect.setStrength(0.0)
            w.setGraphicsEffect(effect)

            anim = QPropertyAnimation(effect, b"strength", self)
            anim.setDuration(2500)
            anim.setKeyValueAt(0.0, 0.0)
            anim.setKeyValueAt(0.15, 0.85)
            anim.setKeyValueAt(0.50, 0.85)
            anim.setKeyValueAt(1.0, 0.0)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

            def _cleanup():
                try: w.setGraphicsEffect(None)
                except: pass

            anim.finished.connect(_cleanup)
            anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

            if not hasattr(self, '_flash_anims'):
                self._flash_anims = []
            self._flash_anims.append(anim)
            self._flash_anims = self._flash_anims[-50:]
        except Exception as e:
            logger.info(f"[flash] {name}: {e}")

    def _build_diff_report(self, before: dict, after: dict):
        """返回 (改动列表, 受影响分组集合)"""
        lines = []
        tabs_hit = set()
        for key, (cn_name, tab_name) in self._CONTROL_LABELS.items():
            b = before.get(key)
            a = after.get(key)
            if b is None and a is None:
                continue
            try:
                if isinstance(b, float) and isinstance(a, float):
                    if abs(b - a) < 1e-4:
                        continue
            except: pass
            if b == a:
                continue
            def _fmt(v):
                if v is None: return "—"
                s = str(v)
                return s if len(s) < 40 else s[:37] + "..."
            lines.append(f"  • {cn_name}: <span style='color:#7d8187'>{_fmt(b)}</span> "
                         f"→ <span style='color:#dadbdf'>{_fmt(a)}</span>")
            tabs_hit.add(tab_name)
        return lines, tabs_hit
