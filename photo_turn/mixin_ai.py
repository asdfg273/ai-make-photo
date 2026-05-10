# photo_turn/mixin_ai.py
# ============================================================
#  PyQt6 版 AIToolsMixin
# ============================================================

import cv2
import numpy as np
import threading

from PIL import Image, ImageDraw, ImageFilter

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore    import QTimer


class AIToolsMixin:

    def run_adetailer(self):
        if self._adetailer_running:
            QMessageBox.information(self, "提示", "ADetailer 正在运行中，请稍候...")
            return
        self._adetailer_running = True
        threading.Thread(target=self._adetailer_worker, daemon=True).start()

    def _adetailer_worker(self):
        def _status(text, color="#f9e2af"):
            QTimer.singleShot(0, lambda: (
                self.lbl_status.setText(text),
                self.lbl_status.setStyleSheet(f"color:{color};")
            ))

        try:
            _status("🔍 正在检测人脸...", "#f9e2af")
            cv_img = cv2.cvtColor(np.array(self.current_img), cv2.COLOR_RGB2BGR)
            gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade  = cv2.CascadeClassifier(cascade_path)
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

            result_img  = self.current_img.copy()
            result_mask = self.mask_img.copy()

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
                    from model_manager import ModelManager
                    manager = ModelManager()
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

                    # 羽化边缘遮罩
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

                except Exception:
                    continue

            _status("✅ ADetailer 处理完成", "#a6e3a1")
            QTimer.singleShot(
                0, lambda: self._on_adetailer_complete(result_img, result_mask)
            )

        except Exception as e:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.critical(
                    self, "错误", f"处理失败: {e}"
                ),
            )
        finally:
            self._adetailer_running = False

    def _on_adetailer_complete(
        self, result_img: Image.Image, result_mask: Image.Image
    ):
        self.push_history(self.current_img, self.mask_img)
        self.current_img = result_img
        self.mask_img    = result_mask
        self.update_canvas(self.current_img)
        QMessageBox.information(self, "成功", "ADetailer 人脸修复完成！")