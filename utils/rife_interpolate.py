# utils/rife_interpolate.py
"""
🎞️ RIFE 帧插值 - 用 rife-ncnn-vulkan.exe
把 8fps 视频插值到 24/30/60fps
"""
import os
import subprocess
import tempfile
import shutil
import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RIFEInterpolator:
    def __init__(self, rife_dir=None):
        if rife_dir is None:
            from utils.paths import PROJECT_ROOT
            rife_dir = os.path.join(PROJECT_ROOT, "tools", "rife")
        self.rife_dir = os.path.abspath(rife_dir)
        self.exe = os.path.join(self.rife_dir, "rife-ncnn-vulkan.exe")
        self.model = os.path.join(self.rife_dir, "rife-v4.6")

    def is_available(self) -> bool:
        return os.path.exists(self.exe) and os.path.exists(self.model)

    def interpolate_video(
        self,
        input_video: str,
        output_video: str = None,
        target_fps: int = 24,
        source_fps: int = 8,
    ) -> str:
        """
        input_video : 输入 mp4/gif
        target_fps  : 目标帧率 (24/30/60)
        source_fps  : 原始帧率
        """
        if not self.is_available():
            raise FileNotFoundError(f"❌ RIFE 未安装: {self.exe}")

        # 计算插值倍数 (2/3/4/8)
        multiplier = round(target_fps / source_fps)
        if multiplier < 2:
            multiplier = 2

        # 输出路径
        if output_video is None:
            base = os.path.splitext(input_video)[0]
            output_video = f"{base}_rife{target_fps}fps.mp4"

        # 临时目录
        with tempfile.TemporaryDirectory() as tmp:
            frames_in = os.path.join(tmp, "in")
            frames_out = os.path.join(tmp, "out")
            os.makedirs(frames_in)
            os.makedirs(frames_out)

            # === Step 1: 提取帧 ===
            logger.info(f"🎞️ [RIFE] 提取原始帧...")
            cap = cv2.VideoCapture(input_video)
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(frames_in, f"{idx:06d}.png"), frame)
                idx += 1
            cap.release()
            logger.info(f"   ✅ 提取 {idx} 帧")

            # === Step 2: 调 RIFE 插值 ===
            logger.info(f"🎞️ [RIFE] 插值 x{multiplier}...")
            cmd = [
                self.exe,
                "-i", frames_in,
                "-o", frames_out,
                "-m", self.model,
                "-n", str(idx * multiplier),  # 目标总帧数
                "-f", "%06d.png",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.rife_dir,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"RIFE 插值失败 (code={result.returncode}): "
                    f"{result.stderr[:500]}")

            out_frames = sorted(os.listdir(frames_out))
            if not out_frames:
                raise RuntimeError("RIFE 未产出任何帧")
            logger.info(f"   ✅ 生成 {len(out_frames)} 帧")

            # === Step 3: 合成视频 ===
            logger.info(f"🎞️ [RIFE] 合成 {target_fps}fps 视频...")
            first = cv2.imread(os.path.join(frames_out, out_frames[0]))
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, target_fps, (w, h))
            for fn in out_frames:
                writer.write(cv2.imread(os.path.join(frames_out, fn)))
            writer.release()

        logger.info(f"✅ [RIFE] 完成: {output_video}")
        return output_video