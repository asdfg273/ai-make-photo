import os
import gc
import io
import datetime
import threading
import warnings
from utils.prompt_enhancer import get_enhancer
from PIL import Image

from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt,pyqtSlot,QMetaObject
from utils.app_utils import OUTPUT_DIR
from core.presets         import PROMPT_PRESETS

from utils.prompt_enhancer import get_enhancer, PromptEnhancer
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

        if hasattr(self, 'spin_batch'):
            self.spin_batch.setValue(
                getattr(cfg, 'default_batch', 1))


        # ── 采样器 ───────────────────────────────────────────
        if hasattr(self, 'combo_sampler'):
            saved = getattr(cfg, 'default_sampler', '')
            idx = self.combo_sampler.findText(saved)
            if idx >= 0:
                self.combo_sampler.setCurrentIndex(idx)

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

        # ── Hires.fix ────────────────────────────────────────
        if hasattr(self, 'chk_hires'):
            self.chk_hires.setChecked(
                getattr(cfg, 'use_hires', False))
        if hasattr(self, 'scale_hires_denoise'):
            self.scale_hires_denoise.setValue(
                getattr(cfg, 'hires_denoise', 0.45))

        # ── 输出格式 ─────────────────────────────────────────
        if hasattr(self, 'combo_img_format'):
            fmt = getattr(cfg, 'output_format', 'PNG')
            idx = self.combo_img_format.findText(fmt)
            if idx >= 0:
                self.combo_img_format.setCurrentIndex(idx)

        # ── 输出目录 ─────────────────────────────────────────
        if hasattr(self, 'combo_output_dir'):
            self.combo_output_dir.setCurrentText(
                getattr(cfg, 'output_dir', 'outputs/'))

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
        """切换主模型时同步刷新 LoRA 下拉框（按架构过滤）"""
        if self.ai is None:
            return
        
        data = self._current_model_data()
        if not data:
            return
        
        # SDXL 识别：优先用体积，兜底用关键词
        is_sdxl = data.get("size_gb", 0) > 4.2
        if not is_sdxl:
            name_lower = data["name"].lower()
            is_sdxl = any(k in name_lower for k in ["xl", "sdxl", "pony", "turbo", "lightning"])
        
        loras = self.ai.get_available_loras("sdxl" if is_sdxl else "sd1.5")

        for combo in self.combo_loras:
            current = combo.currentText()
            combo.clear()
            combo.addItems(loras)
            if current in loras:
                combo.setCurrentText(current)
            else:
                combo.setCurrentIndex(0)

        # 更新备忘录区域提示
        arch_str = "SDXL" if is_sdxl else "SD1.5"
        self.text_lora_info.setReadOnly(False)
        self.text_lora_info.setPlainText(f"🔄 已切换至 {arch_str} 的 LoRA 列表")
        self.text_lora_info.setStyleSheet(
            "color: #585b70; background: #181825; border: 1px solid #313244;"
        )
        self.text_lora_info.setReadOnly(True)

    def load_model_info(self, index=None):
        """显示模型备注（已由 scan_models 读取）"""
        data = self._current_model_data()
        note = data.get("note", "").strip()
        
        if note:
            preview = note[:80] + ("..." if len(note) > 80 else "")
            _set_label_style(self.lbl_model_info, f"📌 备忘: {preview}", "#89dceb")
        else:
            txt_path = os.path.splitext(data.get("path", ""))[0] + ".txt"
            _set_label_style(
                self.lbl_model_info, 
                f"💡 提示: 可在 {txt_path} 记录备注", 
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
                    info_texts.append(f"[槽{i + 1}] ❌ 读取失败: {e}")
            else:
                info_texts.append(f"[槽{i + 1}] (无备忘录)")

        # 更新 UI
        self.text_lora_info.setReadOnly(False)
        self.text_lora_info.clear()

        if not has_lora:
            self.text_lora_info.setPlainText("💡 未使用 LoRA 插件")
            self.text_lora_info.setStyleSheet(
                "color: #585b70; background: #181825; border: 1px solid #313244;"
            )
        else:
            if has_any_txt:
                self.text_lora_info.setPlainText(
                    "📌 备忘录:\n" + "\n".join(info_texts)
                )
                self.text_lora_info.setStyleSheet(
                    "color: #E066FF; background: #181825; border: 1px solid #313244;"
                )
            else:
                self.text_lora_info.setPlainText(
                    "💡 提示: 可在 loras 目录下新建同名 txt 记录触发词\n"
                    + "\n".join(info_texts)
                )
                self.text_lora_info.setStyleSheet(
                    "color: #585b70; background: #181825; border: 1px solid #313244;"
                )

        self.text_lora_info.setReadOnly(True)

    def read_png_info(self):
        """从 AI 生成的 PNG 中提取 parameters 元数据"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 AI 生成的 PNG", "", "PNG Images (*.png)"
        )
        if not path:
            return
        try:
            img  = Image.open(path)
            info = img.info.get("parameters", "")
            if not info:
                QMessageBox.information(self, "提示", "这张图片没有包含 AI 生成参数。")
                return

            lines = info.split("\n")
            if lines:
                self.txt_prompt.setPlainText(lines[0])
            if len(lines) >= 2 and lines[1].startswith("Negative prompt:"):
                self.txt_neg.setPlainText(
                    lines[1].replace("Negative prompt: ", "").strip()
                )
            QMessageBox.information(self, "解析成功", "✅ 已成功提取参数！")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

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
        status = "🎨 已挂载局部重绘与遮罩" if mask_path else (
            "已加载底图: " + os.path.basename(path)
        )
        _set_label_style(self.lbl_img_path, status, "#89dceb")

    def clear_reference(self):
        self.ref_image_path  = None
        self.mask_image_path = None
        _set_label_style(self.lbl_img_path, "未选择参考图", "#585b70")

    def cleanup_temp_files(self, verbose=True):
        """清理 outputs/temp 下的 inpaint 中转文件,并清空内存引用"""
        try:
            temp_dir = os.path.join(OUTPUT_DIR, "temp")
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
                            print(f"⚠️ 删除 {name} 失败: {e}")

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

            if verbose and removed:
                print(f"🧹 已清理 temp 文件: {', '.join(removed)}")
        except Exception as e:
            if verbose:
                print(f"⚠️ cleanup_temp_files 异常: {e}")

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
                "已加载动作图: " + os.path.basename(path),
                "#f9e2af",
            )
        except Exception:
            self.lbl_pose_path.setText("已加载动作图: " + os.path.basename(path))

        if hasattr(self, 'chk_use_pose'):
            self.chk_use_pose.setChecked(True)

        # 同步显示骨架预览缩略图
        try:
            img = Image.open(path)
            self.show_pose_preview(img)
        except Exception as e:
            print(f"⚠️ 骨架图预览失败: {e}")

        # 状态栏提示
        try:
            self._set_status(
                f"📂 已加载 ControlNet 参考图: {os.path.basename(path)}",
                "#a6e3a1"
            )
        except Exception:
            pass

    def stop_generation(self):
        if self.is_generating:
            self.cancel_flag = True
            self._set_status("⚠️ 正在强行刹车，请稍候...", "#f38ba8")

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
            print(f"预览加载失败: {e}")

    def update_preview_ui(self, preview_img: Image.Image):
        """生成过程中实时更新预览（来自生成线程）"""
        try:
            self.preview_canvas.set_pixmap(_pil_to_pixmap(preview_img))
        except Exception as e:
            print(f"实时预览更新失败: {e}")

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
            print(f"⚠️ show_pose_preview 失败: {e}")

    def open_editor(self):
        print(f"[DEBUG] open_editor 来自文件: {__file__}")
        print(f"[DEBUG] 这是新版! 应该会传 callback_on_save")
        """打开专业修图器 - PyQt6 版"""
        # 防重入
        if getattr(self, "_editor_opening", False):
            return

        # 必须先有一张已生成的图
        if not getattr(self, "current_generated_path", None) or \
                not os.path.exists(self.current_generated_path):
            self._set_status("⚠️ 请先在预览区选中一张生成的图片！", "#f38ba8")
            return

        if ProImageEditor is None:
            QMessageBox.critical(
                self, "模块缺失",
                "ProImageEditor 尚未迁移到 PyQt6,请检查 photo_turn/pro_editor_qt.py"
            )
            return

        # ---- 保存回调 ----
        def on_editor_saved(edited_pil_img, mask_pil_img):
            print("[EDITOR] ▶ on_editor_saved 被触发")
            os.makedirs(os.path.join(OUTPUT_DIR, "temp"), exist_ok=True)
            ref_path  = os.path.abspath(
                os.path.join(OUTPUT_DIR, "temp", "inpaint_ref.png"))
            mask_path = os.path.abspath(
                os.path.join(OUTPUT_DIR, "temp", "inpaint_mask.png"))
            edited_pil_img.save(ref_path)
            mask_pil_img.save(mask_path)

            # 诊断
            try:
                mn, mx = mask_pil_img.getextrema()
                print(f"[EDITOR] 遮罩 extrema=({mn},{mx}) size={mask_pil_img.size}")
                if mx == 0:
                    self._set_status(
                        "⚠️ 遮罩为全黑(未涂抹),将退化为图生图模式", "#fab387")
                    self.set_reference_image(ref_path, None)
                else:
                    self.set_reference_image(ref_path, mask_path)
            except Exception as e:
                print(f"[EDITOR] 遮罩检查失败: {e}")
                self.set_reference_image(ref_path, mask_path)

            # 关掉 ControlNet,避免抢走 inpaint 分支
            try:
                if hasattr(self, 'chk_use_pose') and self.chk_use_pose.isChecked():
                    self.chk_use_pose.setChecked(False)
                    print("👉 [工作流] 已自动关闭 ControlNet,优先执行局部重绘")
            except Exception as e:
                print(f"⚠ 关闭 ControlNet 失败(可忽略): {e}")

            self._set_status(
                "✅ 遮罩已准备完毕！点击生成将自动进入【局部重绘】模式", "#a6e3a1")
            self.show_preview(ref_path)

        # ---- 打开修图器 ----
        self._editor_opening = True
        try:
            editor = ProImageEditor(
                self,
                self.current_generated_path,
                callback_on_save=on_editor_saved,   # ★ 关键:一定要传这个
            )
            editor.exec()
        finally:
            self._editor_opening = False


    def open_gallery_to_edit(self):
        print(f"[DEBUG] open_gallery_to_edit 被调用, 来自文件: {__file__}")

        photo_dir = os.path.abspath("photo")
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

        # ---- 保存回调 ----
        def on_editor_saved(edited_pil_img: Image.Image, mask_pil_img: Image.Image):
            print("[EDITOR] ▶ on_editor_saved 被触发 (from gallery)")
            try:
                timestamp = datetime.datetime.now().strftime("%H%M%S")

                # 1) 归档到 photo/
                archive_path = os.path.join(photo_dir, f"pro_edited_{timestamp}.png")
                edited_pil_img.save(archive_path)

                # 2) 写入 inpaint 标准路径,和 open_editor 保持一致
                temp_dir = os.path.join(OUTPUT_DIR, "temp")
                os.makedirs(temp_dir, exist_ok=True)
                ref_path  = os.path.abspath(os.path.join(temp_dir, "inpaint_ref.png"))
                mask_path = os.path.abspath(os.path.join(temp_dir, "inpaint_mask.png"))
                edited_pil_img.save(ref_path)

                # 3) 遮罩诊断
                has_mask = False
                if mask_pil_img is not None:
                    mn, mx = mask_pil_img.getextrema()
                    print(f"[EDITOR] 遮罩 extrema=({mn},{mx}) size={mask_pil_img.size}")
                    if mx > 0:
                        mask_pil_img.save(mask_path)
                        has_mask = True
                    else:
                        print("[EDITOR] ⚠ 遮罩全黑,未涂抹,退化为图生图")
                else:
                    print("[EDITOR] ⚠ 未收到遮罩对象,退化为图生图")

                # 4) 设置参考图 / 遮罩
                if has_mask:
                    self.set_reference_image(ref_path, mask_path)
                    self._set_status(
                        "✅ 遮罩已准备完毕！点击生成将自动进入【局部重绘】模式",
                        "#a6e3a1")
                else:
                    self.set_reference_image(ref_path, None)
                    self._set_status(
                        "✅ 修图完成！已加载为参考图(图生图模式)。",
                        "#a6e3a1")

                # 5) 关闭 ControlNet,避免抢走 inpaint 分支
                try:
                    if hasattr(self, 'chk_use_pose') and self.chk_use_pose.isChecked():
                        self.chk_use_pose.setChecked(False)
                        print("👉 [工作流] 已自动关闭 ControlNet,优先执行局部重绘")
                except Exception as e:
                    print(f"⚠ 关闭 ControlNet 失败(可忽略): {e}")

                # 6) 刷新预览
                self.show_preview(ref_path)

                # 7) 更新 current_generated_path,后续再次进入编辑器时可直接复用
                self.current_generated_path = ref_path

            except Exception:
                import traceback
                print("[EDITOR] ❌ 回调内部异常:")
                print(traceback.format_exc())

        # ---- 打开修图器 ----
        self._set_status("正在打开专业级图片编辑器...", "#f9e2af")
        editor = ProImageEditor(
            self, file_path, callback_on_save=on_editor_saved
        )
        editor.exec()

    def on_model_selected(self, index: int = 0):
        """模型切换统一入口 - 信息+LoRA+备注"""
        self.load_model_info(index)
        self.refresh_lora_by_model()
        self._refresh_model_note()  # 备注独立刷新

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
            lines.append(f"📦 [{mtype}] {fname}")
        if note:
            lines.append(f"📌 {note}")
        else:
            lines.append("💡 可建同名 .txt 文件添加备注")
    
        label.setText("\n".join(lines))
        label.setStyleSheet(
            f"color:{'#89dceb' if note else '#a6adc8'}; "
            "padding:4px; font-family:Consolas; font-size:11px;"
        )



    def on_enhance_prompt(self):
        """智能改写：同时润色正面和负面提示词"""
        raw_pos = self.txt_prompt.toPlainText().strip()
        raw_neg = self.txt_neg.toPlainText().strip()

        if not raw_pos and not raw_neg:
            self._set_status("⚠️ 提示词为空", "#f9e2af")
            return

        self.btn_enhance_prompt.setEnabled(False)
        self.btn_enhance_prompt.setText("改写中...")

        def task():
            try:
                enhancer = get_enhancer()
                if enhancer.model is None:
                    enhancer.load(model_id="qwen/Qwen2-VL-2B-Instruct")

                print(f"[ENHANCE] 正面输入: {raw_pos!r}")
                print(f"[ENHANCE] 负面输入: {raw_neg!r}")

                result_pos = enhancer.enhance(raw_pos, mode="positive") if raw_pos else ""
                result_neg = enhancer.enhance(raw_neg, mode="negative") if raw_neg else ""

                print(f"[ENHANCE] 正面输出: {result_pos!r}")
                print(f"[ENHANCE] 负面输出: {result_neg!r}")

                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self, "_apply_enhance_result",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, result_pos),
                    Q_ARG(str, result_neg),
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"❌ [enhance_prompt] 异常: {e}")
                

        import threading
        threading.Thread(target=task, daemon=True).start()

    @pyqtSlot(str, str)
    def _apply_enhance_result(self, pos: str, neg: str):
        if pos.strip():
            self.txt_prompt.setPlainText(pos.strip())
        if neg.strip():
            self.txt_neg.setPlainText(neg.strip())
        self._restore_enhance_button()
        self._set_status("✨ 改写完成", "#a6e3a1")

    @pyqtSlot()
    def _restore_enhance_button(self):
        self.btn_enhance_prompt.setEnabled(True)
        self.btn_enhance_prompt.setText("✨ 智能改写")

    # ============================================================
    #  动画 Tab AI 工具（智能改写 / 识图生成 / 旅行段改写）
    # ============================================================

    def on_enhance_video_prompt(self):
        """动画 Tab 智能改写：润色正面+负面视频提示词"""
        raw_pos = self.txt_video_prompt.toPlainText().strip()
        raw_neg = self.txt_video_neg.toPlainText().strip()

        if not raw_pos and not raw_neg:
            self._set_status("⚠️ 视频提示词为空", "#f9e2af")
            return

        self.btn_enhance_video_prompt.setEnabled(False)
        self.btn_enhance_video_prompt.setText("改写中...")

        def task():
            try:
                enhancer = get_enhancer()
                if enhancer.model is None:
                    enhancer.load()
                result_pos = enhancer.enhance(raw_pos, mode="positive") if raw_pos else ""
                result_neg = enhancer.enhance(raw_neg, mode="negative") if raw_neg else ""
                QMetaObject.invokeMethod(
                    self, "_apply_video_enhance",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, result_pos), Q_ARG(str, result_neg))
            except Exception as e:
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
        self.btn_enhance_video_prompt.setText("✨ 智能改写")
        self._set_status("✨ 视频提示词改写完成", "#a6e3a1")

    @pyqtSlot()
    def _restore_video_enhance_button(self):
        if hasattr(self, 'btn_enhance_video_prompt'):
            self.btn_enhance_video_prompt.setEnabled(True)
            self.btn_enhance_video_prompt.setText("✨ 智能改写")

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
        self._set_status("📷 正在识别图片，请稍候...", "#f9e2af")

        def task():
            try:
                from PIL import Image
                enhancer = get_enhancer()
                if enhancer.model is None:
                    enhancer.load()
                result = enhancer.describe_image(img_path, user_hint)
                self._bridge.enhance_done_signal.emit(result)
            except Exception as e:
                traceback.print_exc()
                self._bridge.enhance_done_signal.emit(f"[识图失败] {e}")

        # 重定向完成信号到视频栏
        def _apply_vision_result(result: str):
            if result.startswith("[识图失败]"):
                self._set_status(result, "#f38ba8")
            else:
                self.txt_video_prompt.setPlainText(result.strip())
                self._set_status("📷 视频识图完成", "#a6e3a1")
            if hasattr(self, 'btn_vision_video_prompt'):
                self.btn_vision_video_prompt.setEnabled(True)
                self.btn_vision_video_prompt.setText("📷 识图生成")

        try:
            self._bridge.enhance_done_signal.disconnect()
        except Exception:
            pass
        self._bridge.enhance_done_signal.connect(_apply_vision_result)

        threading.Thread(target=task, daemon=True).start()

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
            self._set_status("⚠️ 旅行分段无提示词可改写", "#f9e2af")
            return

        self.btn_enhance_travel.setEnabled(False)
        self.btn_enhance_travel.setText("改写中...")
        self._set_status(f"✨ 正在改写 {len(segments_to_enhance)} 个旅行段...", "#f9e2af")

        def task():
            try:
                enhancer = get_enhancer()
                if enhancer.model is None:
                    enhancer.load()
                for seg, orig in segments_to_enhance:
                    try:
                        enhanced = enhancer.enhance(orig, mode="positive")
                        QMetaObject.invokeMethod(
                            self, "_apply_travel_seg_enhance",
                            Qt.ConnectionType.QueuedConnection,
                            Q_ARG(object, seg), Q_ARG(str, enhanced))
                    except Exception as e:
                        print(f"⚠️ 改写旅行段失败: {e}")
                QMetaObject.invokeMethod(
                    self, "_restore_travel_enhance_button",
                    Qt.ConnectionType.QueuedConnection, Q_ARG(int, len(segments_to_enhance)))
            except Exception as e:
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
            self.btn_enhance_travel.setText("✨ 改写旅行段")
        if count > 0:
            self._set_status(f"✨ 已完成 {count} 个旅行段改写", "#a6e3a1")
        else:
            self._set_status("⚠️ 旅行段改写失败", "#f38ba8")


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
                f"✅ 已加载角色参考图: {os.path.basename(path)}", "#a6e3a1")

    def on_unload_models(self):
        """点击'释放内存'按钮"""
        if self.is_generating:
            QMessageBox.warning(self, "提示", "请先停止当前生成任务")
            return

        try:
            self._set_status("🧹 正在释放模型...", "#fab387")
            if hasattr(self, 'ai'):
                self.ai.unload_all()
            self._set_status("✅ 模型已释放,内存空闲", "#a6e3a1")

            # UI 反馈: 给一个内存占用提示
            try:
                import psutil
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"📊 当前内存: {mem:.1f} MB", flush=True)
            except ImportError:
                pass

        except Exception as e:
            QMessageBox.critical(self, "释放失败", str(e))

  

    def _on_enhance_done(self, result: str):
        """
        改写 / 识图 完成的统一回调（运行在主线程，可安全操作 UI）
    
        result 内容约定:
            - 正常结果 → 直接是 prompt 字符串
            - 失败时   → "[识图失败] xxx" 或 "[改写失败] xxx"
        """
        # ── 恢复按钮状态 ──────────────────────────────────────
        if hasattr(self, 'btn_enhance_prompt'):
            self.btn_enhance_prompt.setEnabled(True)
            self.btn_enhance_prompt.setText("✨ 智能改写")

        if hasattr(self, 'btn_vision_prompt'):
            self.btn_vision_prompt.setEnabled(True)
            self.btn_vision_prompt.setText("📷 识图生成")

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
    
        self._set_status("✨ 提示词生成完成！", "#a6e3a1")


    def on_vision_prompt(self):
        """📷 识图生成 —— 让 Qwen2-VL 看图后输出 SD 提示词"""
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

        self._set_status("📷 正在识别图片，请稍候（首次需下载约 4GB 模型）...",
                         "#f9e2af")

        def task():
            try:
                from utils.prompt_enhancer import PromptEnhancer
                from PIL import Image

                enhancer = get_enhancer()
            
                # 强制使用视觉模型
                if enhancer.wd_session is None:
                    enhancer.load_wd_tagger()
                # 如果用户输入了 hint,还需要 LLM
                if user_hint and enhancer.llm_model is None:
                    enhancer.load_llm()

                # 读图（限制大小防 OOM）
                img = Image.open(img_path).convert("RGB")

                # 调用识图接口
                result = enhancer.describe_image(img)
            
                # 通过信号回到主线程更新 UI
                self._bridge.enhance_done_signal.emit(result)

            except Exception as e:
                import traceback
                err = traceback.format_exc()
                print(f"❌ [vision_prompt] 异常:\n{err}")
                self._bridge.enhance_done_signal.emit(f"[识图失败] {str(e)}")

        import threading
        threading.Thread(target=task, daemon=True).start()


    def run_tiled_diffusion(self):
        from utils.tiled_diffusion import tiled_img2img
        from utils.app_utils import OUTPUT_DIR
        from PIL import Image
        import os, glob, threading, traceback, datetime
    
        # ---- 1. 找最后一张图 ----
        last_path = getattr(self, "last_generated_path", None)
        if not last_path or not os.path.exists(last_path):
            pngs = glob.glob(os.path.join(OUTPUT_DIR, "*.png"))
            if not pngs:
                self._set_status("⚠️ 没有可用的图，请先生成一张", "#f9e2af")
                return
            last_path = max(pngs, key=os.path.getmtime)
            print(f"📌 自动选择最新图: {last_path}")
    
        # ---- 2. 检查 pipe ----
        if not getattr(self.ai, "img2img_pipe", None):
            self._set_status("⚠️ img2img pipeline 未加载，请先跑一次普通生成", "#f38ba8")
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
            f"🧩 大图生成中: {target_w}×{target_h} (tile={tile_size})",
            "#89dceb"
        )
    
        # ---- 5. 后台线程 ----
        def task():
            try:
                def progress_cb(cur, tot):
                    self._bridge.progress_signal.emit(cur, tot)
                    self._bridge.status_signal.emit(
                        f"🧩 大图进度: {cur}/{tot} 块", "#89dceb"
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
                out_path = os.path.join(OUTPUT_DIR, f"{ts}_tiled_{target_w}x{target_h}.png")
                result.save(out_path)
            
                self._bridge.image_signal.emit(result)
                self._bridge.status_signal.emit(
                    f"✅ 大图已保存: {os.path.basename(out_path)}", "#a6e3a1"
                )
                self.last_generated_path = out_path
            except InterruptedError:
                self._bridge.status_signal.emit("⏹️ 大图生成已取消", "#f9e2af")
            except Exception as e:
                traceback.print_exc()
                self._bridge.error_signal.emit(f"大图生成失败: {e}")
    
        threading.Thread(target=task, daemon=True).start()
