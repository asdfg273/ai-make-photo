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
from PyQt6.QtCore    import pyqtSignal
logger = logging.getLogger(__name__)

class AIToolsMixin:
    """AI 修脸（ADetailer）。

    线程模型：worker 跑在裸 threading.Thread 里，**不能**用
    QTimer.singleShot 回 UI —— 普通线程没有 Qt 事件循环，回调永远不触发
    （表现为点按钮毫无反应、错误提示全被吞）。这里统一改成 pyqtSignal：
    跨线程 emit 会由 Qt 自动排队到接收者（GUI）线程执行。
    """

    # 跨线程 → GUI 线程
    sig_adetailer_status = pyqtSignal(str, str)            # text, color
    sig_adetailer_error  = pyqtSignal(str, str)            # text, 级别(info/critical/warning)
    sig_adetailer_done   = pyqtSignal(object, object, int, int)  # result_img, result_mask, ok, total

    def _adetailer_connect(self):
        """惰性连接一次（mixin 自身初始化，不依赖宿主 __init__）。"""
        if getattr(self, "_adetailer_connected", False):
            return
        self._adetailer_connected = True
        self.sig_adetailer_status.connect(self._on_adetailer_status)
        self.sig_adetailer_error.connect(self._on_adetailer_error)
        self.sig_adetailer_done.connect(self._on_adetailer_done)

    def _on_adetailer_status(self, text: str, color: str):
        try:
            self.lbl_status.setText(text)
            self.lbl_status.setStyleSheet(f"color:{color};")
        except Exception:
            logger.debug("[ADetailer] 状态栏更新跳过")

    def _on_adetailer_error(self, text: str, level: str):
        if level == "critical":
            QMessageBox.critical(self, "错误", text)
        elif level == "warning":
            QMessageBox.warning(self, "提示", text)
        else:
            QMessageBox.information(self, "提示", text)

    def _on_adetailer_done(self, result_img, result_mask, success_count, total):
        """GUI 线程：先解锁再套用结果（用户取消时也要解锁）。"""
        self._adetailer_running = False
        self._adetailer_snapshot = None
        try:
            self._on_adetailer_complete(result_img, result_mask,
                                        success_count, total)
        except Exception:
            logger.exception("ADetailer 结果应用失败")
            QMessageBox.warning(self, "提示", "ADetailer 结果应用失败，请查看日志")

    def run_adetailer(self):
        self._adetailer_connect()
        if self._adetailer_running:
            QMessageBox.information(self, "提示", "ADetailer 正在运行中，请稍候...")
            return
        if self.current_img is None:
            QMessageBox.information(self, "提示", "没有可处理的图像。")
            return

        # 与主程序生成互斥：同一个 img2img_pipe 不能被两个线程同时调用
        ai = getattr(self, "ai", None)
        if ai is not None and getattr(ai, "is_generating", False):
            QMessageBox.information(
                self, "提示", "正在生成中，请等本次生成结束再修脸。")
            return

        # 在 GUI 线程构建快照（包含图像数据 + 遮罩），避免线程不安全
        snapshot = self.current_img.copy()
        mask_snapshot = self.mask_img.copy() if self.mask_img else None

        self._adetailer_running = True
        self._adetailer_snapshot = snapshot  # 完成回调会用

        # 传入快照，不是实例属性
        threading.Thread(
            target=self._adetailer_body,
            args=(snapshot, mask_snapshot),
            daemon=True
        ).start()

    # 2. 包装层：只负责把异常报回 GUI，解锁交给完成回调
    def _adetailer_body(self, snapshot, mask_snapshot):
        """只负责跑 worker 并把异常报回 GUI。

        注意：不在这里复位 _adetailer_running —— 结果还没套用到画布，
        提前解锁会让用户能再次点「修脸」而并发调用同一管线。
        解锁统一由 _on_adetailer_done（GUI 线程）负责。
        """
        self._adetailer_connect()
        try:
            self._adetailer_worker(snapshot, mask_snapshot)
        except Exception as e:
            logger.exception("ADetailer 处理失败")
            self.sig_adetailer_error.emit(f"ADetailer 处理失败：{e}", "critical")
            self._adetailer_running = False

    def _adetailer_worker(self, snapshot, mask_snapshot):
        """snapshot: PIL.Image，修复前的图；mask_snapshot: PIL.Image | None，修复前的遮罩。"""
        import os
        import torch                      # 惰性导入：避免拖慢应用启动
        from core.model_manager import ModelManager

        self._adetailer_connect()

        def _status(text, color="#f9e2af"):
            self.sig_adetailer_status.emit(text, color)

        try:
            _status("正在检测人脸…", "#f9e2af")
            cv_img = cv2.cvtColor(np.array(snapshot), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            cascade_candidates = [
                "weights/lbpcascade_animeface.xml",
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            ]
            cascade_path = None
            for p in cascade_candidates:
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
                self.sig_adetailer_error.emit("未检测到明显的人脸。", "info")
                self._adetailer_running = False
                return

            result_img = snapshot.copy()
            result_mask = (
                mask_snapshot.copy() if mask_snapshot
                else Image.new("L", snapshot.size, 0)
            )

            # ✅ 管线检查提到循环外
            manager = ModelManager()
            if manager.img2img_pipe is None:
                raise RuntimeError("img2img 管线未加载，请先加载底模")

            success_count = 0

            for idx, (x, y, w, h) in enumerate(faces):
                _status(
                    f"正在修复第 {idx+1}/{len(faces)} 张人脸…",
                    "#f9e2af",
                )

                mx, my = int(w * 0.4), int(h * 0.4)
                x1 = max(0, x - mx)
                y1 = max(0, y - int(my * 1.5))
                x2 = min(result_img.width, x + w + mx)
                y2 = min(result_img.height, y + h + my)
                cw, ch = x2 - x1, y2 - y1
                if cw < 8 or ch < 8:
                    logger.warning(f"ADetailer 人脸 {idx+1} 区域过小,跳过")
                    continue

                face_crop = result_img.crop((x1, y1, x2, y2))
                face_crop_512 = face_crop.resize((512, 512), Image.LANCZOS)

                try:
                    # 与主程序一致：优先走 encode_prompt（compel + 架构分派），
                    # 拿不到就退回裸 prompt=，保证不同架构都能跑。
                    kwargs = {}
                    try:
                        kwargs = dict(manager.encode_prompt(
                            "highly detailed face, perfect eyes, "
                            "symmetrical face, beautiful skin, "
                            "masterpiece, best quality",
                            "blurry, low quality, distorted face, ugly",
                        ) or {})
                    except Exception as e:
                        logger.debug(f"[ADetailer] encode_prompt 不可用,退回裸 prompt: {e}")

                    if not kwargs.get("prompt_embeds"):
                        kwargs["prompt"] = (
                            "highly detailed face, perfect eyes, "
                            "symmetrical face, beautiful skin, "
                            "masterpiece, best quality"
                        )
                        kwargs["negative_prompt"] = (
                            "blurry, low quality, distorted face, ugly"
                        )

                    with torch.inference_mode():
                        enhanced = manager.img2img_pipe(
                            image=face_crop_512,
                            strength=0.35,
                            num_inference_steps=25,
                            **kwargs,
                        ).images[0]

                    fixed = enhanced.resize((cw, ch), Image.LANCZOS)

                    mask = Image.new("L", (cw, ch), 0)
                    blur_radius = max(5, min(50, int(min(cw, ch) * 0.15)))
                    draw = ImageDraw.Draw(mask)
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
                    logger.error(f"ADetailer 人脸 {idx+1} 失败: {e}")
                    self.sig_adetailer_error.emit(f"修复中止：{e}", "critical")
                    self._adetailer_running = False
                    return

                except Exception as e:
                    logger.warning(f"ADetailer 人脸 {idx+1} 跳过: {e}")

            if success_count == 0:
                _status("人脸修复失败", "#f38ba8")
                self.sig_adetailer_error.emit("未能成功修复任何人脸，请检查日志。", "warning")
                self._adetailer_running = False
                return

            _status(
                f"人脸修复完成（{success_count}/{len(faces)} 张）",
                "#a6e3a1",
            )
            # 交给 GUI 线程套用（结果对象跨线程传递，只读不改）
            self.sig_adetailer_done.emit(
                result_img, result_mask, success_count, len(faces))

        except Exception as e:
            logger.error(f"ADetailer 处理失败: {e}")
            self.sig_adetailer_error.emit(f"处理失败：{e}", "critical")
            self._adetailer_running = False

    def _on_adetailer_complete(self, result_img, result_mask, success_count, total):
        """ADetailer 完成回调（GUI 线程）"""
        snapshot = self._adetailer_snapshot  # 从实例变量取
        self._adetailer_snapshot = None      # 及时释放这张全图

        if snapshot is None:
            logger.warning("ADetailer 完成但快照丢失，跳过确认")
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
                f"画布已被修改，是否用人脸修复结果覆盖？\n成功修复 {success_count}/{total} 张人脸",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                logger.info("用户取消人脸修复结果覆盖")
                return

        # 应用结果
        self.push_history(self.current_img, self.mask_img)  # 修复前入历史
        self._filter_anchor = None
        self.current_img = result_img
        self.mask_img = result_mask
        self.base_img = result_img.copy()
        self.push_history()  # 修复后入历史
        self.update_canvas(self.current_img, force=True)

        self.lbl_status.setText(f"人脸修复完成（{success_count}/{total} 张）")
        self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:12px;")



    def _show_model_diagnostics(self):
        """弹出当前已加载模型的诊断信息对话框（共享实现见 core.model_diagnostics）。"""
        from core.model_diagnostics import show_runtime_diagnostics
        mgr = getattr(self, 'ai', None)
        show_runtime_diagnostics(self, mgr)
