# photo_turn/mixin_ai.py
# ============================================================
#  PyQt6 版 AIToolsMixin
# ============================================================

import cv2
import numpy as np
import threading

from PIL import Image, ImageDraw, ImageFilter
import logging
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore    import QTimer
logger = logging.getLogger(__name__)

class AIToolsMixin:

    def run_adetailer(self):
        if self._adetailer_running:
            QMessageBox.information(self, "提示", "ADetailer 正在运行中，请稍候...") 
            return
        if self.current_img is None:
            QMessageBox.information(self, "提示", "没有可处理的图像。")
            return
        self._adetailer_running = True
        self._adetailer_snapshot = self.current_img.copy() 
        threading.Thread(target=self._adetailer_worker, daemon=True).start()

    def _adetailer_worker(self, snapshot):          # <-- 接收 snapshot
        def _status(text, color="#f9e2af"):
            QTimer.singleShot(0, lambda: (
                self.lbl_status.setText(text),
                self.lbl_status.setStyleSheet(f"color:{color};")
            ))

        try:
            _status("🔍 正在检测人脸...", "#f9e2af")
            cv_img = cv2.cvtColor(np.array(snapshot), cv2.COLOR_RGB2BGR)  # <-- 用 snapshot
            gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            cascade_candidates = [
                "weights/lbpcascade_animeface.xml",
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            ]
            cascade_path = None
            for p in cascade_candidates:
                import os
                if os.path.isfile(p):
                    cascade_path = p
                    break

            if cascade_path is None:
                raise FileNotFoundError(
                    "未找到人脸检测模型（animeface 或 haarcascade）"
                )

            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                QTimer.singleShot(
                    0,
                    lambda: QMessageBox.information(
                        self, "提示", "未检测到明显的人脸。"
                    ),
                )
                return

            result_img  = snapshot.copy()           # <-- 用 snapshot
            result_mask = self.mask_img.copy()
            success_count = 0

            for idx, (x, y, w, h) in enumerate(faces):
                _status(f"🧑‍🎨 ADetailer: 正在修复第 {idx+1}/{len(faces)} 张脸...", "#f9e2af")

                mx, my = int(w * 0.4), int(h * 0.4)
                x1 = max(0, x - mx)
                y1 = max(0, y - int(my * 1.5))
                x2 = min(result_img.width,  x + w + mx)
                y2 = min(result_img.height, y + h + my)
                cw, ch = x2 - x1, y2 - y1

                face_crop     = result_img.crop((x1, y1, x2, y2))
                face_crop_512 = face_crop.resize((512, 512), Image.LANCZOS)

                try:
                    from core.model_manager import ModelManager
                    manager = ModelManager()

                    if manager.img2img_pipe is None:
                        raise RuntimeError("img2img 管线未加载，请先加载底模")

                    enhanced = manager.img2img_pipe(
                        prompt=(
                            "highly detailed face, perfect eyes, "
                            "symmetrical face, beautiful skin, masterpiece, best quality"
                        ),
                        negative_prompt="blurry, low quality, distorted face, ugly",
                        image=face_crop_512,
                        strength=0.35,
                        num_inference_steps=25,
                    ).images[0]

                    fixed = enhanced.resize((cw, ch), Image.LANCZOS)

                    mask        = Image.new("L", (cw, ch), 0)
                    blur_radius = max(5, min(50, int(min(cw, ch) * 0.15)))
                    draw        = ImageDraw.Draw(mask)
                    draw.rectangle(
                        [blur_radius, blur_radius,
                         cw - blur_radius, ch - blur_radius],
                        fill=255,
                    )
                    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

                    result_img.paste(fixed, (x1, y1), mask)

                    draw_mask = ImageDraw.Draw(result_mask)
                    draw_mask.ellipse([x1, y1, x2, y2], fill=255)

                    success_count += 1

                except RuntimeError as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"ADetailer 人脸 {idx+1} 失败: {e}"
                    )
                    QTimer.singleShot(
                        0,
                        lambda err=str(e): QMessageBox.critical(
                            self, "错误", f"修复中止: {err}"
                        ),
                    )
                    return

                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"ADetailer 人脸 {idx+1} 跳过: {e}"
                    )

            if success_count == 0:
                _status("⚠️ ADetailer: 所有人脸均修复失败", "#f38ba8")
                QTimer.singleShot(
                    0,
                    lambda: QMessageBox.warning(
                        self, "提示", "未能成功修复任何人脸，请检查日志。"
                    ),
                )
                return

            _status(f"✅ ADetailer 完成 ({success_count}/{len(faces)} 张)", "#a6e3a1")
            QTimer.singleShot(
                0,
                lambda: self._on_adetailer_complete(
                    result_img, result_mask, success_count, len(faces)
                ),
            )

        except Exception as e:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.critical(self, "错误", f"处理失败: {e}"),
            )
        finally:
            self._adetailer_running = False

    def _on_adetailer_complete(self, result_img, result_mask, success_count, total):
        """ADetailer 完成回调"""
        snapshot = self._adetailer_snapshot  # 从实例变量取
    
        if snapshot is None:
            logger.warning("⚠️ ADetailer 完成但快照丢失，跳过确认")
            self.current_img = result_img
            self.mask_img = result_mask
            self.base_img = result_img.copy()
            self.push_history()
            self.update_canvas(self.current_img, force=True)
            return
    
        # 比较快照和当前画布
        if self.current_img.tobytes() != snapshot.tobytes():
            reply = QMessageBox.question(
                self,
                "确认覆盖",
                f"画布已被修改，是否用 ADetailer 结果覆盖？\n成功修复 {success_count}/{total} 张人脸",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                logger.info("🚫 用户取消 ADetailer 覆盖")
                return
    
        # 应用结果
        self.push_history(self.current_img, self.mask_img)  # 修复前入历史
        self._filter_anchor = None
        self.current_img = result_img
        self.mask_img = result_mask
        self.base_img = result_img.copy()
        self.push_history()  # 修复后入历史
        self.update_canvas(self.current_img, force=True)
    
        self.lbl_status.setText(f"✅ ADetailer 完成 ({success_count}/{total} 张)")
        self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:13px;")



    def _show_model_diagnostics(self):
        """弹出当前已加载模型的诊断信息对话框（共享实现见 core.model_diagnostics）。"""
        from core.model_diagnostics import show_runtime_diagnostics
        mgr = getattr(self, 'ai', None)
        show_runtime_diagnostics(self, mgr)