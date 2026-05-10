import os
import gc
import io
import datetime
import threading
import warnings

from PIL import Image

from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt

from utils.app_utils import OUTPUT_DIR, PROMPT_PRESETS

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

    def refresh_models(self):
        """刷新主模型列表与 LoRA 列表"""
        if not self.ai:
            return

        models = self.ai.get_available_models()
        self.combo_model.clear()
        self.combo_model.addItems(models if models else ["未找到模型"])

        if models:
            self.combo_model.setCurrentIndex(0)
            self.load_model_info()

        arch = "sdxl" if (
            hasattr(self.ai, "is_sdxl") and self.ai.is_sdxl
        ) else "sd1.5"
        loras = self.ai.get_available_loras(arch)

        for combo in self.combo_loras:
            current = combo.currentText()
            combo.clear()
            combo.addItems(loras)
            if current in loras:
                combo.setCurrentText(current)
            else:
                combo.setCurrentIndex(0)  # 默认"无"

    def refresh_lora_by_model(self):
        """切换主模型时同步刷新 LoRA 下拉框（按架构过滤）"""
        current_model = self.combo_model.currentText()
        if not current_model:
            return

        # SDXL 识别
        name_lower = current_model.lower()
        is_sdxl = any(
            k in name_lower for k in ["xl", "sdxl", "pony", "turbo", "lightning"]
        )
        if not is_sdxl:
            model_path = os.path.join("models", current_model)
            if os.path.exists(model_path):
                size_gb = os.path.getsize(model_path) / (1024 ** 3)
                is_sdxl = size_gb > 4.2

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
        """选择模型时读取同名 txt 备忘录，并清空 VRAM"""
        # 清理旧管线
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        model_name = self.combo_model.currentText()
        if not model_name or model_name == "未找到模型":
            return

        txt_path = os.path.join(
            "models",
            model_name.replace(".safetensors", ".txt").replace(".ckpt", ".txt"),
        )
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    memo = f.read().strip()[:80]
                _set_label_style(self.lbl_model_info, f"📌 备忘: {memo}", "#89dceb")
            except Exception:
                pass
        else:
            _set_label_style(
                self.lbl_model_info, "💡 提示: 可新建同名 txt 记录", "#585b70"
            )

    def load_lora_info(self, index=None):
        """任意 LoRA 槽位变化时，刷新备忘录文本框"""
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
                os.path.join("loras", "sdxl",  txt_name),
                os.path.join("loras", "sd1.5", txt_name),
                os.path.join("loras",          txt_name),
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

    def apply_preset(self, index=None):
        """下拉框选择预设时填入提示词"""
        preset_name = self.combo_preset.currentText()
        if preset_name in PROMPT_PRESETS:
            self.txt_prompt.setPlainText(PROMPT_PRESETS[preset_name]["p"])
            self.txt_neg.setPlainText(PROMPT_PRESETS[preset_name]["n"])
            self._set_status(f"✨ 已应用预设: {preset_name}", "#89dceb")

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

    def load_pose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择骨架/线稿图", "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.pose_image_path = path
            _set_label_style(
                self.lbl_pose_path,
                "已加载动作图: " + os.path.basename(path),
                "#f9e2af",
            )
            self.var_use_pose.set(True)
            # 同步显示骨架预览
            try:
                img = Image.open(path)
                self.show_pose_preview(img)
            except Exception as e:
                print(f"骨架图预览失败: {e}")

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
            self.btn_edit.setEnabled(True)
            self.btn_upscale.setEnabled(True)

            # ★ 去重:同一路径只加入画廊一次
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
        """在左侧骨架画布显示 ControlNet 参考图"""
        try:
            img.thumbnail((450, 600), Image.Resampling.LANCZOS)
            self.pose_canvas.set_pixmap(_pil_to_pixmap(img))
        except Exception as e:
            print(f"骨架预览失败: {e}")

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
        self.load_model_info(index)
        self.refresh_lora_by_model()

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