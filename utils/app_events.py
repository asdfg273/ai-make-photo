import os
import io
import datetime
import threading
import traceback
from .prompt_enhancer import get_enhancer
from PIL import Image

from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt,pyqtSlot,QMetaObject, Q_ARG, QTimer
# 输出目录统一从 utils.paths 取（不要再硬编码 OUTPUT_DIR —— 那会让
# 主出图与 tiled 放大/临时文件写到两个不同的目录）
from utils.paths import host_output_dir

import logging
logger = logging.getLogger(__name__)
try:
    from photo_turn.pro_editor_qt import ProImageEditor
except ImportError:
    ProImageEditor = None

def _pil_to_pixmap(pil_img: Image.Image) -> QPixmap:
    """将 PIL Image 无损转换为 QPixmap（支持 GPU 画布显示）"""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    qimg = QImage()
    qimg.loadFromData(buf.read())
    return QPixmap.fromImage(qimg)


def _set_label_style(label, text: str, color: str = "#cdd6f4"):
    """统一更新 QLabel 文字与颜色"""
    label.setText(text)
    label.setStyleSheet(f"color: {color};")

class EventMixin:
    """专门负责处理所有按钮点击、滑块拖动、下拉框刷新的逻辑"""

    def _show_model_diagnostics(self):
        """模型诊断弹窗（新/旧 UI 共用，走 core.model_diagnostics 共享实现）。"""
        from core.model_diagnostics import show_runtime_diagnostics
        show_runtime_diagnostics(self, getattr(self, 'ai', None))

    def apply_config_to_ui(self):
        """把 config 中保存的值还原到 UI 控件"""

        cfg = self.config
        logger.info(f"[配置还原] default_batch={getattr(cfg, 'default_batch', 'MISSING')}, "
                    f"spin_batch存在={hasattr(self, 'spin_batch')}")

        # ── 提示词 ──────────────────────────────────────────
        if getattr(cfg, 'last_prompt', ''):
            self.txt_prompt.setPlainText(cfg.last_prompt)
        if getattr(cfg, 'last_neg', ''):
            self.txt_neg.setPlainText(cfg.last_neg)

        # ── 基础参数 ─────────────────────────────────────────
        if hasattr(self, 'scale_strength'):
            self.scale_strength.setValue(
                getattr(cfg, 'default_strength', 0.6))

        if hasattr(self, 'scale_cfg'):
            self.scale_cfg.setValue(
                getattr(cfg, 'default_cfg', 7.0))

        if hasattr(self, 'spin_steps'):
            self.spin_steps.setValue(
                getattr(cfg, 'default_steps', 30))

        if hasattr(self, 'spin_width'):
            self.spin_width.setValue(
                getattr(cfg, 'default_width', 512))
        if hasattr(self, 'spin_height'):
            self.spin_height.setValue(
                getattr(cfg, 'default_height', 768))
        # 分辨率下拉框同步还原（生成实际读 combo_res）
        if hasattr(self, 'combo_res'):
            _res = f"{getattr(cfg, 'default_width', 512)}x{getattr(cfg, 'default_height', 768)}"
            _idx = self.combo_res.findText(_res)
            if _idx >= 0:
                self.combo_res.setCurrentIndex(_idx)

        if hasattr(self, 'spin_batch'):
            self.spin_batch.setValue(
                getattr(cfg, 'default_batch', 1))


        # ── 采样器 ───────────────────────────────────────────
        if hasattr(self, 'combo_sampler'):
            saved = getattr(cfg, 'default_sampler', '')
            idx = self.combo_sampler.findText(saved)
            if idx >= 0:
                self.combo_sampler.setCurrentIndex(idx)
        if hasattr(self, 'combo_scheduler'):
            saved_style = getattr(cfg, 'sampler_scheduler', '默认') or '默认'
            idx_style = self.combo_scheduler.findText(saved_style)
            if idx_style >= 0:
                self.combo_scheduler.setCurrentIndex(idx_style)

        # ── ADetailer ────────────────────────────────────────
        if hasattr(self, 'chk_use_adetailer'):
            self.chk_use_adetailer.setChecked(
                getattr(cfg, 'use_adetailer', False))
        if hasattr(self, 'scale_adetailer_strength'):
            self.scale_adetailer_strength.setValue(
                getattr(cfg, 'adetailer_strength', 0.35))

        if hasattr(self, 'chk_use_ad_hand'):
            self.chk_use_ad_hand.setChecked(
                getattr(cfg, 'use_ad_hand', False))
        if hasattr(self, 'scale_ad_hand'):
            self.scale_ad_hand.setValue(
                getattr(cfg, 'ad_hand_strength', 0.25))
        if hasattr(self, 'scale_ad_hand_blend'):
            self.scale_ad_hand_blend.setValue(
                getattr(cfg, 'ad_hand_blend', 0.65))

        # ── FreeU ───────────────────────────────────────────
        if hasattr(self, 'chk_freeu'):
            self.chk_freeu.setChecked(
                getattr(cfg, 'freeu_enabled', False))

        # ── CLIP Skip ────────────────────────────────────────
        if hasattr(self, 'combo_clip_skip'):
            _cs = int(getattr(cfg, 'clip_skip', 0) or 0)
            self.combo_clip_skip.blockSignals(True)
            self.combo_clip_skip.setCurrentIndex(
                _cs if 0 <= _cs < self.combo_clip_skip.count() else 0)
            self.combo_clip_skip.blockSignals(False)

        # ── Qwen 改写模式 ─────────────────────────────────────
        if hasattr(self, 'combo_enhance_mode'):
            _em = getattr(cfg, 'enhance_mode', 'strict') or 'strict'
            _idx = self.combo_enhance_mode.findData(_em)
            self.combo_enhance_mode.blockSignals(True)
            self.combo_enhance_mode.setCurrentIndex(_idx if _idx >= 0 else 0)
            self.combo_enhance_mode.blockSignals(False)

        # ── Hires.fix ────────────────────────────────────────
        if hasattr(self, 'chk_hires'):
            self.chk_hires.setChecked(
                getattr(cfg, 'use_hires', False))
        if hasattr(self, 'scale_hires_denoise'):
            self.scale_hires_denoise.setValue(
                getattr(cfg, 'hires_denoise', 0.45))
        if hasattr(self, 'scale_hires_tile_denoise'):
            self.scale_hires_tile_denoise.setValue(
                getattr(cfg, 'hires_tile_denoise', 0.25))
        # 高清方式 / 顺序（8GB 卡默认走分块）
        if hasattr(self, 'combo_hires_mode'):
            _mode = {"auto": "自动", "latent": "整图 (快)",
                     "tiled": "分块 (省显存)"}.get(
                         getattr(cfg, 'hires_mode', 'auto'), "自动")
            _i = self.combo_hires_mode.findText(_mode)
            if _i >= 0:
                self.combo_hires_mode.setCurrentIndex(_i)
        if hasattr(self, 'combo_hires_order'):
            _order = {"hires_first": "先高清后修脸",
                      "adetailer_first": "先修脸后高清"}.get(
                          getattr(cfg, 'hires_order', 'hires_first'),
                          "先高清后修脸")
            _j = self.combo_hires_order.findText(_order)
            if _j >= 0:
                self.combo_hires_order.setCurrentIndex(_j)

        # ── 区域提示词 ───────────────────────────────────────
        if hasattr(self, 'chk_use_regions'):
            self.chk_use_regions.blockSignals(True)
            self.chk_use_regions.setChecked(
                bool(getattr(cfg, 'use_regions', False)))
            self.chk_use_regions.blockSignals(False)
        if hasattr(self, '_refresh_regions_summary'):
            self._refresh_regions_summary()

        # ── 多参考图 (IP-Adapter) ─────────────────────────────
        # 列表本身由编辑器直接写进 config.ipa_refs，这里只回填开关 + 摘要
        if hasattr(self, 'chk_use_ipa_refs'):
            self.chk_use_ipa_refs.blockSignals(True)
            self.chk_use_ipa_refs.setChecked(
                bool(getattr(cfg, 'use_ipa_refs', False)))
            self.chk_use_ipa_refs.blockSignals(False)
        if hasattr(self, '_refresh_ipa_refs_summary'):
            self._refresh_ipa_refs_summary()

        # ── 输出格式 ─────────────────────────────────────────
        if hasattr(self, 'combo_img_format'):
            fmt = getattr(cfg, 'output_format', 'PNG')
            idx = self.combo_img_format.findText(fmt)
            if idx >= 0:
                self.combo_img_format.setCurrentIndex(idx)

        # ── AI 改写模型档位 ───────────────────────────────────
        if hasattr(self, 'combo_ai_model'):
            idx = self.combo_ai_model.findData(getattr(cfg, 'qwen_model_key', 'qwen2vl_2b'))
            if idx >= 0:
                self.combo_ai_model.setCurrentIndex(idx)
                _key = getattr(cfg, 'qwen_model_key', 'qwen2vl_2b')
                from utils.prompt_enhancer import PromptEnhancer
                PromptEnhancer().set_model_key(_key)
                logger.info(f"启动档位同步 → {_key}")

        # ── 翻译模式默认档位 ──────────────────────────────────
        if hasattr(self, 'combo_trans_mode'):
            self.combo_trans_mode.setCurrentIndex(
                max(0, min(2, int(getattr(cfg, 'default_trans_mode', 2)))))

        # ── 是否翻译成英文（关掉 → 直接把中文原文投递给模型）──
        if hasattr(self, 'chk_translate_prompt'):
            on = bool(getattr(cfg, 'translate_prompt', True))
            # 用 blockSignals 避免启动时触发 _on_translate_toggled 去写配置
            try:
                self.chk_translate_prompt.blockSignals(True)
            except Exception:
                pass
            self.chk_translate_prompt.setChecked(on)
            try:
                self.chk_translate_prompt.blockSignals(False)
            except Exception:
                pass
            if hasattr(self, 'combo_trans_mode'):
                self.combo_trans_mode.setEnabled(on)
            self._refresh_prompt_lang_hint()

        # ── 输出目录 ─────────────────────────────────────────
        # 显示**解析后**的真实目录：配置里可能是历史占位值 "outputs/"
        # （实际会被 resolve_output_dir 当成未设置），直接显示它会让
        # 用户以为输出去了 outputs/，而实际写进 photo/。
        if hasattr(self, 'combo_output_dir'):
            try:
                from utils.paths import resolve_output_dir
                self.combo_output_dir.setCurrentText(
                    str(resolve_output_dir(cfg)))
            except Exception:
                self.combo_output_dir.setCurrentText(
                    getattr(cfg, 'output_dir', 'outputs/'))

        # ── 水印 ─────────────────────────────────────────────
        try:
            if hasattr(self, 'chk_watermark'):
                self.chk_watermark.setChecked(
                    bool(getattr(cfg, 'watermark_enabled', False)))
            if hasattr(self, 'edit_watermark_text'):
                self.edit_watermark_text.setText(
                    getattr(cfg, 'watermark_text', '') or '')
            if hasattr(self, 'combo_watermark_pos'):
                pos = getattr(cfg, 'watermark_position', '右下') or '右下'
                idx = self.combo_watermark_pos.findText(pos)
                self.combo_watermark_pos.setCurrentIndex(idx if idx >= 0 else 0)
            if hasattr(self, 'scale_watermark_opacity'):
                self.scale_watermark_opacity.setValue(
                    float(getattr(cfg, 'watermark_opacity', 0.6)))
            if hasattr(self, 'scale_watermark_size'):
                self.scale_watermark_size.setValue(
                    float(getattr(cfg, 'watermark_scale', 1.0)))
            self._refresh_watermark_label(
                getattr(cfg, 'watermark_image', '') or '')
        except Exception as e:
            logger.warning(f"水印设置回填失败: {e}")

    @pyqtSlot()
    def refresh_models(self):
        """刷新主模型列表与 LoRA 列表"""
        if not self.ai:
            return
        self._on_model_type_changed()
        self.refresh_lora_by_model()

    def _current_model_data(self):
        """取当前模型的完整 dict，失败返回空 dict"""
        data = self.combo_model.currentData()
        return data if isinstance(data, dict) else {}


    def refresh_lora_by_model(self):
        """切换主模型时同步刷新 LoRA 下拉框（按架构检测结果过滤）"""
        if self.ai is None:
            return

        data = self._current_model_data()
        if not data or not data.get("path"):
            return

        from core.arch import detect, get_arch

        try:
            result = detect(data["path"])
            info = get_arch(result.arch_id)
            sub_dir = info.lora_subdir
            arch_str = info.display_name
        except Exception as e:
            logger.warning(f"检测底模架构失败，跳过 LoRA 刷新: {e}")
            return

        if not sub_dir:
            loras = []
            arch_str = f"{arch_str}（无对应 LoRA 目录）"
        else:
            loras = self.ai.get_available_loras(sub_dir)

        for combo in self.combo_loras:
            current = combo.currentText()
            combo.clear()
            combo.addItems(loras)
            if current in loras:
                combo.setCurrentText(current)
            else:
                combo.setCurrentIndex(0)

        self.text_lora_info.setReadOnly(False)
        self.text_lora_info.setPlainText(f"已切换至 {arch_str} 的 LoRA 列表")
        self.text_lora_info.setStyleSheet(
            "color: #585b70; background: #181825; border: 1px solid #313244;"
        )
        self.text_lora_info.setReadOnly(True)

    def load_model_info(self, index=None):
        """显示模型备注（已由 scan_models 读取）"""
        data = self._current_model_data()
        note = data.get("note", "").strip()

        if note:
            preview = note[:80] + ("..."if len(note) > 80 else "")
            _set_label_style(self.lbl_model_info, f"备忘: {preview}", "#89dceb")
        else:
            txt_path = os.path.splitext(data.get("path", ""))[0] + ".txt"
            _set_label_style(
                self.lbl_model_info,
                f"提示: 可在 {txt_path} 记录备注",
                "#585b70"
            )

    def load_lora_info(self, index=None):
        """任意 LoRA 槽位变化时，刷新备忘录文本框"""
        from utils import paths

        info_texts = []
        has_lora   = False
        has_any_txt = False

        for i, combo in enumerate(self.combo_loras):
            lora_name = combo.currentText()
            if not lora_name or lora_name == "无":
                continue

            has_lora = True
            base_name = os.path.splitext(lora_name)[0]
            txt_name  = base_name + ".txt"

            search_paths = [
                os.path.join(paths.LORA_DIR, "sdxl",  txt_name),
                os.path.join(paths.LORA_DIR, "sd1.5", txt_name),
                os.path.join(paths.LORA_DIR,          txt_name),
            ]
            txt_found = next((p for p in search_paths if os.path.exists(p)), None)

            if txt_found:
                try:
                    with open(txt_found, "r", encoding="utf-8") as f:
                        memo = f.read().strip()
                    if len(memo) > 150:
                        memo = memo[:150] + "..."
                    info_texts.append(f"[槽{i + 1}] {memo}")
                    has_any_txt = True
                except Exception as e:
                    info_texts.append(f"[槽{i + 1}]  读取失败: {e}")
            else:
                info_texts.append(f"[槽{i + 1}] (无备忘录)")

        # 更新 UI
        self.text_lora_info.setReadOnly(False)
        self.text_lora_info.clear()

        if not has_lora:
            self.text_lora_info.setPlainText("未使用 LoRA 插件")
            self.text_lora_info.setStyleSheet(
                "color: #585b70; background: #181825; border: 1px solid #313244;"
            )
        else:
            if has_any_txt:
                self.text_lora_info.setPlainText(
                    "备忘录:\n"+ "\n".join(info_texts)
                )
                self.text_lora_info.setStyleSheet(
                    "color: #E066FF; background: #181825; border: 1px solid #313244;"
                )
            else:
                self.text_lora_info.setPlainText(
                    "提示: 可在 loras 目录下新建同名 txt 记录触发词\n"
                    + "\n".join(info_texts)
                )
                self.text_lora_info.setStyleSheet(
                    "color: #585b70; background: #181825; border: 1px solid #313244;"
                )

        self.text_lora_info.setReadOnly(True)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择底图/遮罩", "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.set_reference_image(path)

    def set_reference_image(self, path: str, mask_path: str = None):
        self.ref_image_path  = path
        self.mask_image_path = mask_path
        status = "已挂载局部重绘与遮罩"if mask_path else (
            "已加载底图: "+ os.path.basename(path)
        )
        _set_label_style(self.lbl_img_path, status, "#89dceb")
        self._update_ref_thumb()

    def _update_ref_thumb(self):
        """参考图缩略图:显示已载入图片,有遮罩时叠加红色半透明预览"""
        lbl = getattr(self, "lbl_ref_thumb", None)
        if lbl is None:
            return
        path = getattr(self, "ref_image_path", None)
        if not path or not os.path.exists(path):
            lbl.clear()
            lbl.setText("无参考图")
            return
        try:
            img = Image.open(path).convert("RGBA")
            mask_path = getattr(self, "mask_image_path", None)
            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L").resize(img.size)
                alpha = mask.point(lambda v: int(v * 0.45))
                red = Image.new("RGBA", img.size, (243, 139, 168, 0))
                red.putalpha(alpha)
                img.alpha_composite(red)
            pm = _pil_to_pixmap(img)
            w = lbl.width() if lbl.width() > 40 else 220
            h = lbl.height() if lbl.height() > 40 else 120
            pm = pm.scaled(w, h,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(pm)
        except Exception as e:
            logger.warning(f"参考图缩略图加载失败: {e}")
            lbl.clear()
            lbl.setText(os.path.basename(path))

    def clear_reference(self):
        self.ref_image_path  = None
        self.mask_image_path = None
        _set_label_style(self.lbl_img_path, "未选择参考图", "#585b70")
        self._update_ref_thumb()

    def cleanup_temp_files(self, verbose=True):
        """清理 <输出目录>/temp 下的 inpaint 中转文件,并清空内存引用"""
        try:
            temp_dir = os.path.join(host_output_dir(self), "temp")
            if not os.path.isdir(temp_dir):
                return

            # 只清理已知中转文件,避免误删其他内容
            targets = ["inpaint_ref.png", "inpaint_mask.png"]
            removed = []
            for name in targets:
                p = os.path.join(temp_dir, name)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        removed.append(name)
                    except Exception as e:
                        if verbose:
                            logger.warning(f"删除 {name} 失败: {e}")

            # 清空内存引用,避免下次生图误用旧路径
            if getattr(self, "ref_image_path", None) and \
                    os.path.basename(self.ref_image_path) in targets:
                self.ref_image_path = None
            if getattr(self, "mask_image_path", None) and \
                    os.path.basename(self.mask_image_path) in targets:
                self.mask_image_path = None

            # UI 同步(如果界面上仍挂着那张临时图)
            if hasattr(self, "lbl_img_path") and not self.ref_image_path:
                _set_label_style(self.lbl_img_path, "未选择参考图", "#585b70")
                self._update_ref_thumb()

            if verbose and removed:
                logger.info(f"已清理 temp 文件: {', '.join(removed)}")
        except Exception as e:
            if verbose:
                logger.warning(f"cleanup_temp_files 异常: {e}")

    # ------------------------------------------------------------
    #  水印
    # ------------------------------------------------------------
    def _refresh_watermark_label(self, path: str = ""):
        lbl = getattr(self, "lbl_watermark_image", None)
        if lbl is None:
            return
        if path and os.path.exists(path):
            lbl.setText(os.path.basename(path))
            lbl.setToolTip(path)
            lbl.setStyleSheet("color:#a6e3a1;")
        else:
            lbl.setText("未选择")
            lbl.setToolTip("")
            lbl.setStyleSheet("color:#7f849c;")

    def _pick_watermark_image(self):
        """选择自定义图片水印（PNG/JPG，建议带透明通道）。"""
        from PyQt6.QtWidgets import QFileDialog
        start = ""
        try:
            start = os.path.dirname(getattr(self.config, "watermark_image", "") or "")
        except Exception:
            start = ""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择水印图片", start,
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return
        try:
            # 立刻校验能不能打开，别等到出图时才发现是坏文件
            from PIL import Image
            with Image.open(path) as im:
                im.verify()
        except Exception as e:
            QMessageBox.warning(self, "水印图片无法读取",
                                f"这个文件打不开：\n{path}\n\n{e}")
            return
        try:
            self.config.watermark_image = path
            self.config.save()
        except Exception as e:
            logger.warning(f"保存水印设置失败: {e}")
        self._refresh_watermark_label(path)
        if hasattr(self, "chk_watermark"):
            self.chk_watermark.setChecked(True)
        self._set_status(f"已设置图片水印: {os.path.basename(path)}", "#a6e3a1")

    def _clear_watermark_image(self):
        try:
            self.config.watermark_image = ""
            self.config.save()
        except Exception as e:
            logger.warning(f"清除水印设置失败: {e}")
        self._refresh_watermark_label("")
        self._set_status("已清除图片水印", "#89dceb")

    # ==========================================================
    #  区域提示词 / 局部重绘（主界面入口）
    # ==========================================================
    def open_region_editor(self):
        """打开区域编辑器：在图上拖框、每个框写自己的提示词。"""
        try:
            from ui.region_editor import RegionEditorDialog
            w, h = self._region_canvas_size()
            dlg = RegionEditorDialog(self.regions(), width=w, height=h,
                                     parent=self)
            if dlg.exec() == dlg.DialogCode.Accepted:
                regs = dlg.regions()
                self.config.regions = regs
                self.config.use_regions = bool(
                    regs and any(r.get("enabled", True) for r in regs))
                try:
                    self.config.save()
                except Exception as e:
                    logger.warning(f"区域设置保存失败: {e}")
                # 勾选框跟着实际状态走，避免"点了确定但没启用"
                chk = getattr(self, "chk_use_regions", None)
                if chk is not None:
                    chk.blockSignals(True)
                    chk.setChecked(bool(self.config.use_regions))
                    chk.blockSignals(False)
                self._refresh_regions_summary()
                n = len([r for r in regs if r.get("enabled", True)])
                self._set_status(
                    f"已设置 {n} 个区域"if n else "已清空区域",
                    "#a6e3a1"if n else "#7f849c")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "打开区域编辑器失败", str(e))

    def _region_canvas_size(self):
        """当前输出分辨率（区域坐标是归一化的，但画布要按实际比例显示）。"""
        w, h = 512, 768
        try:
            txt = self.combo_res.currentText()
            if "x"in txt.lower():
                a, b = txt.lower().split("x")
                w, h = int(a.strip()), int(b.strip())
        except Exception:
            pass
        return w, h

    def clear_regions(self):
        try:
            self.config.regions = []
            self.config.use_regions = False
            self.config.save()
        except Exception as e:
            logger.warning(f"清空区域失败: {e}")
        chk = getattr(self, "chk_use_regions", None)
        if chk is not None:
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._refresh_regions_summary()
        self._set_status("已清空区域", "#7f849c")

    def regions(self) -> list:
        """当前区域列表（配置为准）。"""
        try:
            return [dict(r) for r in (self.config.regions or [])
                    if isinstance(r, dict)]
        except Exception:
            return []

    def _refresh_regions_summary(self):
        lbl = getattr(self, "lbl_regions_summary", None)
        if lbl is None:
            return
        regs = [r for r in self.regions() if r.get("enabled", True)]
        if not regs:
            lbl.setText("（未设置区域）")
            return
        parts = []
        for i, r in enumerate(regs[:4]):
            nm = r.get("name") or f"区域{i + 1}"
            try:
                wt = float(r.get("weight", 0.5))
            except Exception:
                wt = 0.5
            parts.append(f"{nm}({wt:.2f})")
        more = f"…共 {len(regs)} 个"if len(regs) > 4 else ""
        lbl.setText(" ".join(parts) + more)

    def _apply_regions_if_enabled(self, image, ctx, seed=None):
        """生成后、保存前：按区域逐个局部重绘。任何失败都只警告，不中断出图。"""
        try:
            if not getattr(self.config, "use_regions", False):
                return image
            regs = [r for r in self.regions() if r.get("enabled", True)]
            if not regs:
                return image
            from core.regional import apply_regions
            from workflow import adapters as _ad
            arch = getattr(self.ai, "current_arch_id", "") or "sd15"
            caps = _ad.capabilities_for(arch)
            if not caps.supports_inpaint:
                self.append_log(
                    f"{arch} 架构不支持局部重绘，已跳过区域提示词",
                    "#fab387")
                return image
            model = getattr(self, "_last_model_ref", None)
            if model is None:
                # 主界面这条路上没有 ModelRef，临时包一个给引擎用
                from workflow.values import ModelRef
                model = ModelRef(name=getattr(self.ai, "current_model_name", ""),
                                 path=getattr(self.ai, "current_model_path", ""),
                                 arch_id=arch, pipe=getattr(self.ai, "txt2img_pipe", None),
                                 img2img_pipe=getattr(self.ai, "img2img_pipe", None),
                                 inpaint_pipe=getattr(self.ai, "inpaint_pipe", None),
                                 adapter=_ad.get_adapter(arch),
                                 vae=getattr(getattr(self.ai, "txt2img_pipe", None), "vae", None),
                                 loaded=True)
            steps = int(ctx.get("steps", 24) or 24)
            cfg = float(ctx.get("cfg", 7.0) or 7.0)
            neg = ctx.get("enq_neg") or ctx.get("raw_neg") or ""
            out, meta = apply_regions(
                self.ai, model, image, regs,
                global_negative=neg, steps=steps, cfg=cfg,
                seed=seed if seed is not None else -1,
                progress=lambda a, b: self._region_progress(a, b),
                cancel=lambda: bool(getattr(self, "_cancel_requested", False)),
                status=lambda m: self._set_status(m, "#cba6f7"),
                log=lambda m: self.append_log(m, "#cba6f7"))
            applied = int(meta.get("regions_applied", 0) or 0)
            planned = int(meta.get("regions_planned", len(regs)) or len(regs))
            self.append_log(
                f"区域重绘完成：{applied}/{planned} 个区域生效", "#a6e3a1")
            for wmsg in (meta.get("region_warnings") or [])[:3]:
                self.append_log(f"{wmsg}", "#fab387")
            for emsg in (meta.get("region_errors") or [])[:3]:
                self.append_log(f"{emsg}", "#f38ba8")
            return out
        except Exception as e:
            logger.warning(f"区域重绘失败（沿用原图）: {e}")
            try:
                self.append_log(f"区域重绘失败，已沿用原图：{e}", "#fab387")
            except Exception:
                pass
            return image

    def _region_progress(self, i, n):
        try:
            self.progress_bar.setRange(0, max(1, int(n)))
            self.progress_bar.setValue(int(i))
        except Exception:
            pass

    # ==========================================================
    #  多参考图 (IP-Adapter) —— 主界面入口
    # ==========================================================
    # 与「区域提示词」同一套结构：编辑器把条目写进 config.ipa_refs，
    # 主界面只负责 开关 / 摘要 / 清空；生成时由 _apply_ipa_refs_if_enabled
    # 把条目变成"一次安装 N 个适配器 + 每适配器一个 scale"。
    #
    #  为什么不能只写 config：config 里只是**意图**。真正装上权重、
    #    决定每张图的档位（缺文件时降级）必须走
    #    ModelManager.prepare_ip_adapter_multi()，否则就是"界面设了、
    #    实际没生效"的静默失败。
    def open_ipa_ref_editor(self):
        """打开多参考图编辑器（表格：图片 / 角色 / 权重 / 区域蒙版）。"""
        try:
            from ui.ipa_editor import IPARefEditorDialog
            dlg = IPARefEditorDialog(self.ipa_refs(), parent=self)
            if dlg.exec() == dlg.DialogCode.Accepted:
                refs = dlg.refs()
                self.config.ipa_refs = refs
                self.config.use_ipa_refs = bool(
                    refs and any(r.get("enabled", True) for r in refs))
                try:
                    self.config.save()
                except Exception as e:
                    logger.warning(f"多参考图设置保存失败: {e}")
                # 勾选框跟着实际状态走，避免"点了确定但没启用"
                chk = getattr(self, "chk_use_ipa_refs", None)
                if chk is not None:
                    chk.blockSignals(True)
                    chk.setChecked(bool(self.config.use_ipa_refs))
                    chk.blockSignals(False)
                self._refresh_ipa_refs_summary()
                n = len([r for r in refs if r.get("enabled", True)])
                self._set_status(
                    f"已设置 {n} 张参考图"if n else "已清空参考图",
                    "#a6e3a1"if n else "#7f849c")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "打开多参考图编辑器失败", str(e))

    def clear_ipa_refs(self):
        try:
            self.config.ipa_refs = []
            self.config.use_ipa_refs = False
            self.config.save()
        except Exception as e:
            logger.warning(f"清空参考图失败: {e}")
        chk = getattr(self, "chk_use_ipa_refs", None)
        if chk is not None:
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._refresh_ipa_refs_summary()
        self._set_status("已清空参考图", "#7f849c")

    def ipa_refs(self) -> list:
        """当前参考图列表（配置为准，**含停用条目**）。

        与 `regions()` 的分工一致：这里给编辑器用（停用的条目要能继续
        看到、改回来）；要拿去生成的是 `_enabled_ipa_refs()`。
        """
        try:
            return [dict(r) for r in (self.config.ipa_refs or [])
                    if isinstance(r, dict)]
        except Exception:
            return []

    def _enabled_ipa_refs(self) -> list:
        """启用中的参考图条目（生成时真正参与的那些）。

        过滤规则统一在 `core/ipa_multi.enabled_refs`：以前这四行在各处
        重写了一遍（dict 用 `.get("enabled")`、对象用 `getattr(...)`），
        多写一处就多一个"停用了却还生效"的机会。
        """
        from core import ipa_multi as ipa_mod
        return ipa_mod.enabled_refs(self.ipa_refs())

    def _toggle_ipa_refs(self, checked=False):
        """「启用多参考图」勾选框：同步配置 + 摘要 + 状态栏提示。"""
        try:
            self.config.use_ipa_refs = bool(checked)
        except Exception as e:
            logger.warning(f"记录多参考图开关失败: {e}")
        self._refresh_ipa_refs_summary()
        try:
            if not checked:
                self._set_status("多参考图已关闭", "#7f849c")
                return
            n = len(self._enabled_ipa_refs())
            if n:
                self._set_status(f"多参考图已开启：{n} 张", "#a6e3a1")
            else:
                self._set_status(
                    "已开启多参考图，但还没设置参考图（点「编辑参考图…」）",
                    "#fab387")
        except Exception:
            pass

    def _refresh_ipa_refs_summary(self):
        lbl = getattr(self, "lbl_ipa_refs_summary", None)
        if lbl is None:
            return
        from ui.ipa_editor import summary_text
        refs = self._enabled_ipa_refs()
        if not refs:
            lbl.setText("（未设置参考图）")
            return
        txt = summary_text(refs)
        if not getattr(self.config, "use_ipa_refs", False):
            txt += "（未启用，生成时不生效）"
        lbl.setText(txt)

    def _apply_ipa_refs_if_enabled(self, ctx):
        """主生成路径：把主界面的多参考图**真的**装进管线。

        返回信息 dict（plan / images / scales / masks / summary），
        没启用或装不上时返回 None —— 调用方此时应回落到「单张
        IP-Adapter」或完全不用 IP-Adapter（绝不能静默当作已生效）。

        为什么落点在这里（而不是每个参考图各装一次）：
        diffusers 0.40 的 `load_ip_adapter()` 收**列表**，一次装 N 个适配器、
        每个一个 scale；装第二次是**覆盖**语义。所以必须先把 N 张图
        收齐了再装 —— 这与 `nodes/sampling_nodes.apply_ip_adapter_kwargs`
        的做法完全一致（只是那条路走工作流，这条走主界面）。
        """
        try:
            if not getattr(self.config, "use_ipa_refs", False):
                return None
            entries = self._enabled_ipa_refs()
            if not entries:
                self.append_log(
                    "已开启多参考图，但一条都没启用 → 本次不做多参考图",
                    "#fab387")
                return None
            ai = getattr(self, "ai", None)
            if ai is None:
                self.append_log("模型管理器不在位，已跳过多参考图", "#fab387")
                return None

            # ── 能力门禁：看能力，不看模型名 ──
            arch = getattr(ai, "current_arch_id", "") or "sd15"
            try:
                from workflow import adapters as _ad
                _ad.capabilities_for(arch).require("ip_adapter")
            except Exception as e:
                self.append_log(
                    f"当前架构（{arch}）不支持 IP-Adapter，"
                    f"已跳过多参考图：{e}", "#fab387")
                return None

            from core import ipa_multi as ipa_mod

            def _log_ref(msg):
                self.append_log(msg, "#fab387")

            self._set_status(
                f"正在安装 {len(entries)} 个 IP-Adapter 多参考图...", "#fab387")

            # ── 七步流程走唯一实现：core.ipa_multi.build_runtime ──
            # 工作流那条（`nodes/sampling_nodes.apply_ip_adapter_kwargs`）调的是
            # 同一个函数，只是 on_missing="raise"；这里用 "degrade"——主界面的
            # 语义是"一张参考图读不到不该毁掉整次生成"。
            rt = ipa_mod.build_runtime(
                entries, ai,
                width=int(ctx.get("width", 0) or 0),
                height=int(ctx.get("height", 0) or 0),
                pipe=getattr(ai, "txt2img_pipe", None),
                on_missing="degrade", log=_log_ref, device="cpu")
            if not rt.ok:
                self.append_log(
                    f"多参考图装载失败：{rt.error or '未知原因'}", "#f38ba8")
                return None

            self.append_log(f"多参考图就绪：{rt.summary()}", "#a6e3a1")
            for wmsg in rt.warnings:
                # "调用采样时要带 cross_attention_kwargs"这句是给开发者看的，
                # 对着用户说只会让人困惑。
                if "cross_attention_kwargs" in wmsg:
                    continue
                self.append_log(f"{wmsg}", "#fab387")

            return {
                "plan": rt.plan,
                "refs": list(rt.refs),
                "images": list(rt.images),
                "scales": list(rt.scales),
                "masks": rt.masks,
                "summary": rt.summary(),
                "warnings": list(rt.warnings),
            }
        except Exception as e:
            logger.warning(f"多参考图准备失败（改用单张/IP-Adapter 关闭）: {e}")
            try:
                self.append_log(f"多参考图准备失败，已跳过：{e}", "#fab387")
            except Exception:
                pass
            return None

    def load_pose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择骨架/线稿图", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        self.pose_image_path = path

        # 更新路径标签
        try:
            _set_label_style(
                self.lbl_pose_path,
                "已加载动作图: "+ os.path.basename(path),
                "#f9e2af",
            )
        except Exception:
            self.lbl_pose_path.setText("已加载动作图: "+ os.path.basename(path))

        if hasattr(self, 'chk_use_pose'):
            self.chk_use_pose.setChecked(True)

        # 同步显示骨架预览缩略图
        try:
            img = Image.open(path)
            self.show_pose_preview(img)
        except Exception as e:
            logger.warning(f"骨架图预览失败: {e}")

        # 状态栏提示
        try:
            self._set_status(
                f"已加载 ControlNet 参考图: {os.path.basename(path)}",
                "#a6e3a1"
            )
        except Exception:
            pass

    # stop_generation 的唯一实现在 utils/app_generation.py (含按钮状态恢复)

    # ── 「是否翻译成英文」开关 ──────────────────────────────────
    def _on_translate_toggled(self, on: bool):
        """勾选/取消「翻译成英文再生成」。

        取消时直接把中文（或任何原文）投递给模型 —— 这是 MOSS-Image 这类
        文本编码器本身懂中文的模型要的；SD1.5/SDXL 用 CLIP，关掉会出乱码，
        所以状态栏给出明确提醒，而不是默默放行。
        """
        try:
            self.config.translate_prompt = bool(on)
            self.config.save()
        except Exception as e:
            logger.warning(f"保存「翻译成英文」设置失败: {e}")

        # 关掉翻译时，翻译模式下拉框没有意义 → 禁用（保留值）
        if hasattr(self, 'combo_trans_mode'):
            try:
                self.combo_trans_mode.setEnabled(bool(on))
            except Exception as e:
                logger.debug(f"禁用翻译模式下拉框失败: {e}")

        self._refresh_prompt_lang_hint()

        if on:
            self._set_status("提示词会先翻译成英文再生成（CLIP 类模型只认英文）",
                             "#a6e3a1")
        elif self._arch_supports_zh():
            self._set_status("已关闭翻译：直接把中文原文投递给模型"
                             "（当前模型原生支持中文）", "#a6e3a1")
        else:
            self._set_status("⚠️ 已关闭翻译：当前模型的文本编码器是 CLIP，"
                             "中文可能被当成乱码；换用 MOSS-Image 这类模型再关",
                             "#f38ba8")

    def _selected_arch_supports_zh(self) -> bool:
        """按模型下拉框**当前选中项**判断是否原生吃中文（不必先加载模型）。"""
        try:
            data = self.combo_model.currentData()
            arch = str(data.get("type") or "") if isinstance(data, dict) else ""
            if not arch:
                # 扫描出来的条目只有 name/path，没有 type —— 还没加载模型时
                # 就判成"CLIP 只认英文"会让 MOSS-Image 用户看到错误建议。
                # 这里按路径做一次轻量架构识别（只读 model_index.json），并缓存。
                path = str(data.get("path") or "") if isinstance(data, dict) else ""
                if path:
                    cache = getattr(self, "_arch_hint_cache", None)
                    if cache is None:
                        cache = self._arch_hint_cache = {}
                    if path not in cache:
                        try:
                            from core.arch.signatures import detect
                            cache[path] = detect(path).arch_id or ""
                        except Exception as e:
                            logger.debug(f"按路径识别架构失败: {e}")
                            cache[path] = ""
                    arch = cache[path]
            if not arch:
                return self._arch_supports_zh()
            from core.arch import get_arch
            return bool(getattr(getattr(get_arch(arch), "caps", None),
                                "supports_zh_prompt", False))
        except Exception as e:
            logger.debug(f"按选中项判断「中文提示词」能力失败: {e}")
            return self._arch_supports_zh()

    def _refresh_prompt_lang_hint(self):
        """根据当前模型能力，更新提示词区的"中文/英文"提醒文案。"""
        lbl = getattr(self, 'lbl_prompt_lang_hint', None)
        if lbl is None:
            return
        try:
            on = True
            chk = getattr(self, 'chk_translate_prompt', None)
            if chk is not None:
                on = bool(chk.isChecked())
            zh = self._selected_arch_supports_zh()

            if zh and on:
                text = ("💡 当前模型原生支持中文（可渲染画面中的汉字）——"
                        "你可以取消上面的勾选，直接用中文提示词，"
                        "翻译成英文反而会削弱中文文字渲染")
                color = "#f9e2af"
            elif zh:
                text = "✅ 当前模型原生支持中文：现在直接把中文原文投递给模型"
                color = "#a6e3a1"
            elif on:
                text = ("ℹ️ 当前模型用 CLIP 编码器，只认英文 —— "
                        "建议保持「翻译成英文再生成」")
                color = "#7f849c"
            else:
                text = ("⚠️ 当前模型用 CLIP 编码器，不翻译中文通常出不来预期结果；"
                        "换成 MOSS-Image 这类原生支持中文的模型再关闭翻译")
                color = "#f38ba8"

            lbl.setText(text)
            lbl.setStyleSheet(f"color: {color};")
            # MOSS-Image 额外提示速度与内存（架构是 5B DiT，8GB 卡上很吃力）
            self._refresh_moss_speed_hint(lbl, text, color, zh)
        except Exception as e:
            logger.debug(f"更新提示词语言提示失败: {e}")

    def _refresh_moss_speed_hint(self, lbl, text, color, zh):
        """在同一个提示行里追加 MOSS-Image 的分辨率/步数建议。"""
        arch = ""
        try:
            combo = getattr(self, 'combo_model', None)
            data = combo.currentData() if combo is not None else None
            arch = (data or {}).get("type", "") if isinstance(data, dict) else ""
        except Exception as e:
            logger.debug(f"读取当前架构失败: {e}")
        if arch != "mossimage":
            return
        accel = bool(getattr(getattr(self, "config", None), "moss_accel", False))
        tip = ("\n⏱️ MOSS-Image 是 5B DiT：768×768 / 25~30 步约 1.5 分钟，"
               "1024² 要约 5 分钟（8GB 卡）")
        if accel:
            tip += ("；8bit 常驻显存已开（若 bitsandbytes 内核不可用会自动退回 "
                    "bf16，见日志）")
        else:
            tip += "；DiT 有 9.5GB 要占内存，开 IDE 时容易内存不足"
        lbl.setText(text + tip)
        lbl.setStyleSheet(f"color: {color};")

    def show_preview(self, img_path: str):
        """将图片加载到右侧 GPU 画布并激活操作按钮"""
        try:
            img = Image.open(img_path)
            img.thumbnail((900, 1200), Image.Resampling.LANCZOS)
            self.preview_canvas.set_pixmap(_pil_to_pixmap(img))
            self.current_generated_path = img_path

            # 兼容旧版按钮（可能已废弃）
            for btn_name in ('btn_edit', 'btn_upscale'):
                btn = getattr(self, btn_name, None)
                if btn is not None:
                    btn.setEnabled(True)

            # 画廊去重添加
            if hasattr(self, 'gallery') and img_path:
                if not hasattr(self, '_gallery_seen_paths'):
                    self._gallery_seen_paths = set()
                abs_path = os.path.abspath(img_path)
                if abs_path not in self._gallery_seen_paths:
                    self._gallery_seen_paths.add(abs_path)
                    self.gallery.add_image(img_path, prepend=True)
        except Exception as e:
            logger.warning(f"预览加载失败: {e}")

    def update_preview_ui(self, preview_img: Image.Image):
        """生成过程中实时更新预览（来自生成线程）"""
        try:
            self.preview_canvas.set_pixmap(_pil_to_pixmap(preview_img))
        except Exception as e:
            logger.warning(f"实时预览更新失败: {e}")

    def show_pose_preview(self, img: Image.Image):
        """在 ControlNet 缩略图区域显示参考图"""
        try:
            # 缩略
            thumb = img.copy()
            thumb.thumbnail((300, 180), Image.Resampling.LANCZOS)

            # PIL → QPixmap
            from PIL.ImageQt import ImageQt
            from PyQt6.QtGui import QPixmap, QImage

            qimg = ImageQt(thumb.convert("RGBA"))
            pix = QPixmap.fromImage(QImage(qimg))

            if hasattr(self, 'lbl_cn_thumb'):
                self.lbl_cn_thumb.set_pixmap(pix)
        except Exception as e:
            logger.warning(f"show_pose_preview 失败: {e}")

    def _send_editor_to_inpaint(self, edited_pil_img, mask_pil_img):
        """编辑器「发送到局部重绘」回调:挂参考图+遮罩,不归档画廊"""
        _tmp = os.path.join(host_output_dir(self), "temp")
        os.makedirs(_tmp, exist_ok=True)
        ref_path  = os.path.abspath(os.path.join(_tmp, "inpaint_ref.png"))
        mask_path = os.path.abspath(os.path.join(_tmp, "inpaint_mask.png"))
        edited_pil_img.save(ref_path)
        mask_pil_img.save(mask_path)

        try:
            mn, mx = mask_pil_img.getextrema()
            logger.info(f"[EDITOR→INPAINT] 遮罩 extrema=({mn},{mx})")
            if mx == 0:
                self._set_status(
                    "遮罩为全黑(未涂抹),将退化为图生图模式", "#fab387")
                self.set_reference_image(ref_path, None)
            else:
                self.set_reference_image(ref_path, mask_path)
        except Exception as e:
            logger.warning(f"[EDITOR→INPAINT] 遮罩检查失败: {e}")
            self.set_reference_image(ref_path, mask_path)

        # 关掉 ControlNet,避免抢走 inpaint 分支
        try:
            if hasattr(self, 'chk_use_pose') and self.chk_use_pose.isChecked():
                self.chk_use_pose.setChecked(False)
                logger.info("[工作流] 已自动关闭 ControlNet,优先执行局部重绘")
        except Exception as e:
            logger.warning(f"关闭 ControlNet 失败(可忽略): {e}")

        self._set_status(
            "遮罩已挂载！点击生成将自动进入【局部重绘】模式", "#a6e3a1")
        self.show_preview(ref_path)

    def _preview_image_path(self) -> str:
        """预览区当前这张图的可编辑路径（不存在就返回空串）。

        顺序: current_generated_path（预览区正在显示的那张）→ last_generated_path。
        「🖌️ 编辑」按钮要的就是这个: 不再弹文件夹选图，直接编辑正在预览的那张。
        """
        for attr in ("current_generated_path", "last_generated_path"):
            path = getattr(self, attr, None)
            if not path:
                continue
            try:
                if os.path.isfile(str(path)):
                    return os.path.abspath(str(path))
            except OSError as e:
                logger.debug(f"[EDITOR] 检查预览图 {path} 失败: {e}")
        return ""

    def _ask_pick_image_for_edit(self) -> bool:
        """预览区没图时问一句要不要从磁盘选（默认不弹文件对话框）。"""
        answer = QMessageBox.question(
            self, "没有可编辑的图片",
            "预览区还没有图片。\n\n"
            "可以先生成一张，或在下方「历史画廊」里选中一张再点「🖌️ 编辑」。\n\n"
            "现在就从磁盘里选一张图片来编辑吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def open_editor(self):
        """打开专业修图器 —— 编辑预览区当前这张图。

        主界面「🖌️ 编辑」按钮、画廊右键「载入预览/发送到编辑」、「发送到修脸」
        都走这里: 不弹文件夹选图，直接编辑正在预览的那张。
        """
        logger.debug(f"[EDITOR] open_editor 打开预览区当前图, from {__file__}")
        # 防重入
        if getattr(self, "_editor_opening", False):
            return

        # 必须有可编辑的图（预览区的当前图），而不是让用户再去文件管理器里挑
        target_path = self._preview_image_path()
        if not target_path:
            self._set_status(
                "预览区还没有图片：请先生成一张，或在画廊里选一张", "#f38ba8")
            if self._ask_pick_image_for_edit():
                self.open_gallery_to_edit()
            return

        if ProImageEditor is None:
            QMessageBox.critical(
                self, "模块缺失",
                "ProImageEditor 尚未迁移到 PyQt6,请检查 photo_turn/pro_editor_qt.py"
            )
            return

        # ---- 保存回调 ----
        def on_editor_saved(edited_pil_img, mask_pil_img):
            logger.info("[EDITOR] ▶ on_editor_saved 被触发")
            _tmp = os.path.join(host_output_dir(self), "temp")
            os.makedirs(_tmp, exist_ok=True)
            ref_path  = os.path.abspath(os.path.join(_tmp, "inpaint_ref.png"))
            mask_path = os.path.abspath(os.path.join(_tmp, "inpaint_mask.png"))
            edited_pil_img.save(ref_path)
            mask_pil_img.save(mask_path)

            # 归档到当前输出目录（跟随用户设置），与画廊读取路径一致
            try:
                photo_dir = host_output_dir(self)
                os.makedirs(photo_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                archive_path = os.path.join(photo_dir, f"pro_edited_{timestamp}.png")
                n = 1
                while os.path.exists(archive_path):
                    archive_path = os.path.join(
                        photo_dir, f"pro_edited_{timestamp}_{n}.png")
                    n += 1
                edited_pil_img.save(archive_path)
                logger.info(f"[EDITOR] 修图结果已归档到画廊: {archive_path}")
                if hasattr(self, "refresh_gallery"):
                    QTimer.singleShot(300, self.refresh_gallery)
            except Exception as e:
                logger.warning(f"修图结果归档失败(不影响重绘流程): {e}")

            # 诊断
            try:
                mn, mx = mask_pil_img.getextrema()
                logger.info(f"[EDITOR] 遮罩 extrema=({mn},{mx}) size={mask_pil_img.size}")
                if mx == 0:
                    self._set_status(
                        "遮罩为全黑(未涂抹),将退化为图生图模式", "#fab387")
                    self.set_reference_image(ref_path, None)
                else:
                    self.set_reference_image(ref_path, mask_path)
            except Exception as e:
                logger.warning(f"[EDITOR] 遮罩检查失败: {e}")
                self.set_reference_image(ref_path, mask_path)

            # 关掉 ControlNet,避免抢走 inpaint 分支
            try:
                if hasattr(self, 'chk_use_pose') and self.chk_use_pose.isChecked():
                    self.chk_use_pose.setChecked(False)
                    logger.info("[工作流] 已自动关闭 ControlNet,优先执行局部重绘")
            except Exception as e:
                logger.warning(f"关闭 ControlNet 失败(可忽略): {e}")

            self._set_status(
                "遮罩已准备完毕！点击生成将自动进入【局部重绘】模式", "#a6e3a1")
            self.show_preview(ref_path)

        # ---- 打开修图器 ----
        self._editor_opening = True
        try:
            editor = ProImageEditor(
                self,
                target_path,
                callback_on_save=on_editor_saved,   #  关键:一定要传这个
                callback_on_inpaint=self._send_editor_to_inpaint,
            )
            # 「发送到修脸」入口:编辑器显示后自动跑一次 ADetailer
            if getattr(self, "_editor_auto_adetailer", False):
                self._editor_auto_adetailer = False
                QTimer.singleShot(400, editor.run_adetailer)
            editor.exec()
        finally:
            self._editor_opening = False


    def open_gallery_to_edit(self):
        """从磁盘挑一张图来编辑 —— 预览区没有图时的兜底入口。

        这里不再自带一套保存/遮罩回调: 选中后先载入预览区, 再交给 open_editor,
        保证「编辑」只有一个实现、一套回调(以前两份回调各写一遍, 容易改漏)。
        """
        logger.debug(f"[EDITOR] open_gallery_to_edit 选择文件, from {__file__}")

        photo_dir = host_output_dir(self)
        os.makedirs(photo_dir, exist_ok=True)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择要进行高级调色的图片",
            photo_dir,
            "Image files (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        if ProImageEditor is None:
            QMessageBox.critical(
                self, "模块缺失",
                "ProImageEditor 尚未迁移到 PyQt6,请检查 photo_turn/pro_editor_qt.py"
            )
            return

        # 选中的图先变成「预览区当前图」，再走统一的编辑入口
        try:
            self.show_preview(file_path)
        except Exception as e:
            logger.warning(f"[EDITOR] 载入预览失败, 直接写入路径: {e}")
            self.current_generated_path = file_path
            self.last_generated_path = file_path

        self.open_editor()

    def on_model_selected(self, index: int = 0):
        """模型切换统一入口 - 信息+LoRA+备注"""
        self.load_model_info(index)
        self.refresh_lora_by_model()
        self._refresh_model_note()  # 备注独立刷新
        # 选中项变了就知道该用哪个架构 → 顺手更新"要不要翻译"的提示
        self._refresh_prompt_lang_hint()

    def _refresh_model_note(self):
        """从 currentData 读 txt 备注,刷新 lbl_model_info"""
        label = getattr(self, 'lbl_model_info', None)
        if not label:
            return

        data = self.combo_model.currentData()
        if not data or not isinstance(data, dict):
            return

        note = (data.get('note') or '').strip()
        mtype = data.get('type', '')
        fname = data.get('name', '')

        # 组合信息: [类型] 文件名 + 备注
        lines = []
        if mtype and fname:
            lines.append(f"[{mtype}] {fname}")
        if note:
            lines.append(f"{note}")
        else:
            lines.append("可建同名 .txt 文件添加备注")

        label.setText("\n".join(lines))
        label.setStyleSheet(
            f"color:{'#89dceb'if note else '#a6adc8'}; "
            "padding:4px; font-family:Consolas; font-size:11px;"
        )

    def on_save_character(self):
        """「存为角色」：把当前界面参数一键建成角色卡。"""
        pages = getattr(self, "_pages", None) or {}
        page = pages.get("characters")
        if page is None or not hasattr(page, "create_character_from_shell"):
            try:
                self._set_status("角色库页面不可用", "#f9e2af")
            except Exception:
                pass
            return
        # 先切到角色库页，建卡后用户能直接看到新卡片
        try:
            if hasattr(self, "nav"):
                self.nav.select("characters")
        except Exception:
            pass
        page.create_character_from_shell(self, parent=self)

    def on_enhance_prompt(self):
        """智能改写：按所选模式处理提示词。

        模式（UI 下拉框 combo_enhance_mode）：
          strict   —— ①严格翻译：忠实转换，不改变用户原意（默认）
          polish   —— ②适当润色：修语法/统一用词/规范化排序
          keywords —— ③关键词帮写：从少量关键词扩写成完整提示词

        负面提示词始终走 negative 模式（与正向模式无关）。
        """
        raw_pos = self.txt_prompt.toPlainText().strip()
        raw_neg = self.txt_neg.toPlainText().strip()

        if not raw_pos and not raw_neg:
            self._set_status("提示词为空", "#f9e2af")
            return

        # 读取模式
        mode_key = "strict"
        try:
            if hasattr(self, 'combo_enhance_mode'):
                mode_key = self.combo_enhance_mode.currentData() or "strict"
        except Exception:
            pass
        try:
            self.config.enhance_mode = mode_key
        except Exception:
            pass

        self.btn_enhance_prompt.setEnabled(False)
        self.btn_enhance_prompt.setText("改写中...")

        def task():
            try:
                enhancer = get_enhancer()
                enhancer.ensure_loaded()

                logger.info(f"[ENHANCE] 模式={mode_key} 正面输入: {raw_pos!r}")
                logger.info(f"[ENHANCE] 负面输入: {raw_neg!r}")

                result_pos = enhancer.enhance(raw_pos, mode=mode_key) if raw_pos else ""
                result_neg = enhancer.enhance(raw_neg, mode="negative") if raw_neg else ""

                logger.info(f"[ENHANCE] 正面输出: {result_pos!r}")
                logger.info(f"[ENHANCE] 负面输出: {result_neg!r}")

                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self, "_apply_enhance_result",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, result_pos),
                    Q_ARG(str, result_neg),
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                logger.error(f"[enhance_prompt] 异常: {e}")
                from PyQt6.QtCore import QMetaObject, Qt
                QMetaObject.invokeMethod(
                    self, "_restore_enhance_button",
                    Qt.ConnectionType.QueuedConnection,
                )

            finally:
                if not getattr(self, 'is_generating', False):
                    try:
                        get_enhancer().unload()
                        logger.info("Qwen 已卸载 (智能改写完成)")
                    except Exception:
                        pass

        import threading
        threading.Thread(target=task, daemon=True).start()

    @pyqtSlot(str, str)
    def _apply_enhance_result(self, pos: str, neg: str):
        if pos.strip():
            self.txt_prompt.setPlainText(pos.strip())
        if neg.strip():
            self.txt_neg.setPlainText(neg.strip())
        self._restore_enhance_button()
        _m = "严格翻译"if (hasattr(self, 'combo_enhance_mode')
                            and self.combo_enhance_mode.currentData() == "strict") else "改写"
        self._set_status(f"{_m}完成", "#a6e3a1")

    @pyqtSlot()
    def _restore_enhance_button(self):
        self.btn_enhance_prompt.setEnabled(True)
        self.btn_enhance_prompt.setText("智能改写")

    # ============================================================
    #  动画 Tab AI 工具（智能改写 / 识图生成 / 旅行段改写）
    # ============================================================

    def on_enhance_video_prompt(self):
        """动画 Tab 智能改写：润色正面+负面视频提示词"""
        raw_pos = self.txt_video_prompt.toPlainText().strip()
        raw_neg = self.txt_video_neg.toPlainText().strip()

        if not raw_pos and not raw_neg:
            self._set_status("视频提示词为空", "#f9e2af")
            return

        self.btn_enhance_video_prompt.setEnabled(False)
        self.btn_enhance_video_prompt.setText("改写中...")

        def task():
            try:
                enhancer = get_enhancer()
                enhancer.ensure_loaded()
                result_pos = enhancer.enhance(raw_pos, mode="positive") if raw_pos else ""
                result_neg = enhancer.enhance(raw_neg, mode="negative") if raw_neg else ""
                QMetaObject.invokeMethod(
                    self, "_apply_video_enhance",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, result_pos), Q_ARG(str, result_neg))
            except Exception:
                traceback.print_exc()
                QMetaObject.invokeMethod(
                    self, "_restore_video_enhance_button",
                    Qt.ConnectionType.QueuedConnection)

        threading.Thread(target=task, daemon=True).start()

    @pyqtSlot(str, str)
    def _apply_video_enhance(self, pos: str, neg: str):
        if pos.strip():
            self.txt_video_prompt.setPlainText(pos.strip())
        if neg.strip():
            self.txt_video_neg.setPlainText(neg.strip())
        self.btn_enhance_video_prompt.setEnabled(True)
        self.btn_enhance_video_prompt.setText("智能改写")
        self._set_status("视频提示词改写完成", "#a6e3a1")

    @pyqtSlot()
    def _restore_video_enhance_button(self):
        if hasattr(self, 'btn_enhance_video_prompt'):
            self.btn_enhance_video_prompt.setEnabled(True)
            self.btn_enhance_video_prompt.setText("智能改写")



    def on_vision_video_prompt(self):
        """动画 Tab 识图生成：选图 → AI 识别 → 填入视频正向提示词"""
        user_hint = self.txt_video_prompt.toPlainText().strip()
        if user_hint:
            user_hint = user_hint + ""

        img_path, _ = QFileDialog.getOpenFileName(
            self, "选择要识别的图片", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not img_path:
            return

        if hasattr(self, 'btn_vision_video_prompt'):
            self.btn_vision_video_prompt.setEnabled(False)
            self.btn_vision_video_prompt.setText("识图中...")
        self._set_status("正在识别图片，请稍候...", "#f9e2af")

        def task():
            try:
                enhancer = get_enhancer()
                enhancer.ensure_loaded()
                result = enhancer.describe_image(img_path, user_hint)
                self._bridge.video_enhance_done_signal.emit(result)
            except Exception as e:
                traceback.print_exc()
                self._bridge.video_enhance_done_signal.emit(f"[识图失败] {e}")
            finally:
                try:
                    get_enhancer().unload(reason="video vision done")
                except Exception as e:
                    logger.warning(f"Qwen 卸载失败: {e}")

        threading.Thread(target=task, daemon=True).start()

    def _on_video_vision_done(self, result: str):
        """视频 Tab 识图结果回填(独立信号,不动主 Tab 的 enhance_done_signal)"""
        if result.startswith("[识图失败]"):
            self._set_status(result, "#f38ba8")
        else:
            self.txt_video_prompt.setPlainText(result.strip())
            self._set_status("视频识图完成", "#a6e3a1")
        if hasattr(self, 'btn_vision_video_prompt'):
            self.btn_vision_video_prompt.setEnabled(True)
            self.btn_vision_video_prompt.setText("识图生成")

    def on_enhance_travel_prompts(self):
        """改写所有旅行分段的提示词"""
        segments_to_enhance = []
        for seg in self.travel_segments:
            try:
                if isinstance(seg, dict):
                    txt = seg['prompt_edit'].text().strip()
                else:
                    txt = seg[1].text().strip()
            except Exception:
                txt = ""
            if txt:
                segments_to_enhance.append((seg, txt))

        if not segments_to_enhance:
            self._set_status("旅行分段无提示词可改写", "#f9e2af")
            return

        self.btn_enhance_travel.setEnabled(False)
        self.btn_enhance_travel.setText("改写中...")
        self._set_status(f"正在改写 {len(segments_to_enhance)} 个旅行段...", "#f9e2af")

        def task():
            try:
                enhancer = get_enhancer()
                enhancer.ensure_loaded()
                for seg, orig in segments_to_enhance:
                    try:
                        enhanced = enhancer.enhance(orig, mode="positive")
                        QMetaObject.invokeMethod(
                            self, "_apply_travel_seg_enhance",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(object, seg), Q_ARG(str, enhanced))
                    except Exception as e:
                        logger.warning(f"改写旅行段失败: {e}")
                QMetaObject.invokeMethod(
                    self, "_restore_travel_enhance_button",
                    Qt.ConnectionType.QueuedConnection, Q_ARG(int, len(segments_to_enhance)))
            except Exception:
                traceback.print_exc()
                QMetaObject.invokeMethod(
                    self, "_restore_travel_enhance_button",
                    Qt.ConnectionType.QueuedConnection, Q_ARG(int, 0))

        threading.Thread(target=task, daemon=True).start()

    @pyqtSlot(object, str)
    def _apply_travel_seg_enhance(self, seg, text: str):
        if not text.strip():
            return
        try:
            if isinstance(seg, dict):
                seg['prompt_edit'].setText(text.strip())
            else:
                seg[1].setText(text.strip())
        except Exception:
            pass

    @pyqtSlot(int)
    def _restore_travel_enhance_button(self, count: int):
        if hasattr(self, 'btn_enhance_travel'):
            self.btn_enhance_travel.setEnabled(True)
            self.btn_enhance_travel.setText("改写旅行段")
        if count > 0:
            self._set_status(f"已完成 {count} 个旅行段改写", "#a6e3a1")
        else:
            self._set_status("旅行段改写失败", "#f38ba8")


    def load_ipa_image(self):
        """加载 IP-Adapter 角色参考图"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择角色参考图", "",
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self.ipa_image_path = path
            self.lbl_ipa_image.setText(os.path.basename(path))
            self.lbl_ipa_image.setStyleSheet("color:#a6e3a1; padding:4px;")
            self.chk_use_ipa.setChecked(True)
            self._set_status(
                f"已加载角色参考图: {os.path.basename(path)}", "#a6e3a1")

    # on_unload_models 的唯一实现在 ui/ui_builder.py (MRO 生效版本)

    def _on_enhance_done(self, result: str):
        """
        改写 / 识图 完成的统一回调（运行在主线程，可安全操作 UI）
    
        result 内容约定:
            - 正常结果 → 直接是 prompt 字符串
            - 失败时   → "[识图失败] xxx"或 "[改写失败] xxx"
        """
        # ── 恢复按钮状态 ──────────────────────────────────────
        if hasattr(self, 'btn_enhance_prompt'):
            self.btn_enhance_prompt.setEnabled(True)
            self.btn_enhance_prompt.setText("智能改写")

        if hasattr(self, 'btn_vision_prompt'):
            self.btn_vision_prompt.setEnabled(True)
            self.btn_vision_prompt.setText("识图生成")

        # ── 失败 ─────────────────────────────────────────────
        if result.startswith("[识图失败]") or result.startswith("[改写失败]"):
            self._set_status(result, "#f38ba8")
            try:
                QMessageBox.warning(self, "提示词生成失败", result)
            except Exception:
                pass
            return

        # ── 成功：把结果填入正向提示词框 ─────────────────────
        if hasattr(self, 'txt_prompt'):
            self.txt_prompt.setPlainText(result)

        self._set_status("提示词生成完成！", "#a6e3a1")


    def on_vision_prompt(self):
        """识图生成 —— 让 Qwen2-VL 看图后输出 SD 提示词"""
        user_hint = self.txt_prompt.toPlainText().strip() if hasattr(self, 'txt_prompt') else ""
        # 选择参考图
        from PyQt6.QtWidgets import QFileDialog

        img_path, _ = QFileDialog.getOpenFileName(
            self, "选择要识别的图片", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not img_path:
            return

        # 锁定按钮防重入
        if hasattr(self, 'btn_vision_prompt'):
            self.btn_vision_prompt.setEnabled(False)
            self.btn_vision_prompt.setText("识图中...")

        self._set_status("正在识别图片，请稍候（首次需下载约 4GB 模型）...",
                         "#f9e2af")

        def task():
            try:
                from PIL import Image
                enhancer = get_enhancer()

                # 识图前先清 SD 的 embedding 缓存，给 Qwen 腾显存
                try:
                    from core.model_manager import ModelManager
                    ModelManager()._compel_cache.clear()
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                enhancer.ensure_loaded()

                img = Image.open(img_path).convert("RGB")
                result = enhancer.describe_image(img, user_hint=user_hint)
                del img

                self._bridge.enhance_done_signal.emit(result)

            except Exception as e:
                import traceback
                logger.error(f"[vision_prompt] 异常:\n{traceback.format_exc()}")
                self._bridge.enhance_done_signal.emit(f"[识图失败] {str(e)}")

            finally:
                #  关键：无论成功失败都释放 Qwen 显存
                try:
                    get_enhancer().unload(reason="vision done")
                except Exception as e:
                    logger.warning(f"Qwen 卸载失败: {e}")

        import threading
        threading.Thread(target=task, daemon=True).start()


    def run_tiled_diffusion(self):
        from utils.tiled_diffusion import tiled_img2img
        from PIL import Image
        import os
        import glob
        import threading
        import traceback
        import datetime

        # 跟随用户设置的输出目录（以前硬编码 photo/，改目录后这里就找不到图、
        # 也把结果写到别处去了）
        _od = host_output_dir(self)

        # ---- 1. 找最后一张图 ----
        last_path = getattr(self, "last_generated_path", None)
        if not last_path or not os.path.exists(last_path):
            pngs = glob.glob(os.path.join(_od, "*.png"))
            if not pngs:
                self._set_status("没有可用的图，请先生成一张", "#f9e2af")
                return
            last_path = max(pngs, key=os.path.getmtime)
            logger.info(f"自动选择最新图: {last_path}")

        # ---- 2. 检查 pipe ----
        if not getattr(self.ai, "img2img_pipe", None):
            self._set_status("img2img pipeline 未加载，请先跑一次普通生成", "#f38ba8")
            return

        # ---- 3. 收集 prompt (兼容多种控件名) ----
        def _get_text(*names):
            for n in names:
                w = getattr(self, n, None)
                if w is not None:
                    if hasattr(w, "toPlainText"):
                        return w.toPlainText().strip()
                    if hasattr(w, "text"):
                        return w.text().strip()
            return ""

        prompt = _get_text("txt_prompt", "edit_prompt", "input_prompt")
        neg    = _get_text("txt_negative", "txt_neg", "edit_neg", "edit_negative",
                           "input_negative", "negative_prompt")

        # ---- 4. 收集参数 ----
        init_img  = Image.open(last_path).convert("RGB")
        target_w  = self.spin_tiled_w.value()
        target_h  = self.spin_tiled_h.value()
        tile_size = int(self.combo_tile_size.currentText())
        overlap   = self.spin_tile_overlap.value()
        strength  = self.scale_tile_strength.value()

        self._set_status(
            f"大图生成中: {target_w}×{target_h} (tile={tile_size})",
            "#89dceb"
        )

        # ---- 5. 后台线程 ----
        def task():
            try:
                def progress_cb(cur, tot):
                    self._bridge.progress_signal.emit(cur, tot)
                    self._bridge.status_signal.emit(
                        f"大图进度: {cur}/{tot} 块", "#89dceb"
                    )
                def cancel_cb():
                    return getattr(self, "cancel_flag", False)

                result = tiled_img2img(
                    pipe=self.ai.img2img_pipe,
                    init_image=init_img,
                    prompt=prompt,
                    negative_prompt=neg,
                    target_width=target_w,
                    target_height=target_h,
                    tile_size=tile_size,
                    overlap=overlap,
                    strength=strength,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    seed=-1,
                    callback=progress_cb,
                    cancel_check=cancel_cb,
                )

                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(_od, f"{ts}_tiled_{target_w}x{target_h}.png")
                result.save(out_path)

                self._bridge.image_signal.emit(result)
                self._bridge.status_signal.emit(
                    f"大图已保存: {os.path.basename(out_path)}", "#a6e3a1"
                )
                self.last_generated_path = out_path
            except InterruptedError:
                self._bridge.status_signal.emit("大图生成已取消", "#f9e2af")
            except Exception as e:
                traceback.print_exc()
                self._bridge.error_signal.emit(f"大图生成失败: {e}")

        threading.Thread(target=task, daemon=True).start()
