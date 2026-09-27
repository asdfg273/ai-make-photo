# utils/rife_interpolate.py
"""
🎞️ RIFE 帧插值 - 用 rife-ncnn-vulkan.exe
把 8fps 视频插值到 24/30/60fps

⚠️ 注意：OpenCV 的 cv2.imread/imwrite/VideoWriter 在 Windows 上对**非 ASCII 路径
会静默失败**（返回 None / 写出 0 字节）。本项目路径与输出文件名都可能含中文，
所以这里统一走 imdecode/imencode + tofile/fromfile。
"""
import os
import shutil
import subprocess
from contextlib import contextmanager
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# RIFE 是 ncnn-vulkan 实现，首次运行要为每一帧建 pipeline，
# 帧多时耗时线性增长；给一个宽裕但有上限的超时，避免永久挂死。
_RIFE_TIMEOUT_BASE = 300      # 基础 5 分钟
_RIFE_TIMEOUT_PER_FRAME = 3   # 每帧最多再等 3 秒


@contextmanager
def _work_dir():
    """项目内可写的临时工作目录。

    两个刻意的选择：
      · 不用系统 tempfile：受限环境（沙箱、只读 %TEMP%、企业策略）下不可写，
        Windows 上清理失败还会连带抛 PermissionError 把主流程打断；
      · 不用 tempfile.mkdtemp：它创建的目录在部分受限环境下会被 ACL 锁死
        （连在里面新建文件都 Permission denied）。改用 makedirs + 唯一后缀。
    插帧会产生成百上千张 PNG，统一放项目 photo/ 下的临时目录，退出时尽力清理。
    """
    from utils.paths import OUTPUT_DIR
    import uuid
    base = os.path.join(OUTPUT_DIR, ".rife_tmp")
    os.makedirs(base, exist_ok=True)
    tmp = os.path.join(base, "job_" + uuid.uuid4().hex[:12])
    os.makedirs(tmp, exist_ok=True)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _imread(path):
    """读图（兼容非 ASCII 路径）。失败返回 None。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.debug(f"读取失败 {path}: {e}")
        return None


def _imwrite(path, img) -> bool:
    """写图（兼容非 ASCII 路径）。"""
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        logger.warning(f"写帧失败 {path}: {e}")
        return False


def _probe_video_info(path):
    """用 cv2 探测 (fps, frame_count, width, height)，失败项为 0。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return 0.0, 0, 0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, cnt, w, h


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _has_audio_stream(video_path: str) -> bool:
    """用 ffmpeg -i 的 stderr 判断是否有音轨（无 ffprobe 时的稳妥做法）。"""
    ff = _ffmpeg_exe()
    if not ff:
        return False
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-i", video_path],
            capture_output=True, timeout=30,
        )
        text = (r.stderr or b"").decode("utf-8", errors="ignore")
        return "Audio:" in text
    except Exception as e:
        logger.debug(f"探测音轨失败: {e}")
        return False


def _mux_audio(video_in: str, audio_src: str, video_out: str) -> bool:
    """把 audio_src 的音轨合到 video_in 上，输出到 video_out。"""
    ff = _ffmpeg_exe()
    if not ff:
        logger.warning("⚠️ 未找到 ffmpeg，无法保留音轨")
        return False
    try:
        r = subprocess.run(
            [ff, "-hide_banner", "-y",
             "-i", video_in, "-i", audio_src,
             "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "copy",
             "-shortest", video_out],
            capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            err = (r.stderr or b"").decode("utf-8", errors="ignore")[-300:]
            logger.warning(f"⚠️ 音轨合成失败(code={r.returncode}): {err}")
            return False
        return os.path.exists(video_out) and os.path.getsize(video_out) > 0
    except Exception as e:
        logger.warning(f"⚠️ 音轨合成异常: {e}")
        return False


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

    def plan_multiplier(self, target_fps: int, source_fps: float) -> int:
        """计算插值倍数。

        ⚠️ 旧实现 `round(target/source)` 之后强抬到 >=2，有两个后果：
          · source_fps 为 0 → ZeroDivisionError（未捕获，直接崩）
          · source == target（如 24→24）→ 倍数被抬到 2，帧数翻倍但 fps 不变
            = **播放速度变成一半**
        这里改为：源帧率无效时按目标帧率当作"无需插值"处理；只有真的需要
        提升帧率时才求倍数。
        """
        try:
            tf = int(target_fps)
        except (TypeError, ValueError):
            raise ValueError(f"target_fps 非法: {target_fps!r}")
        if tf <= 0:
            raise ValueError(f"target_fps 必须 > 0，得到 {tf}")

        try:
            sf = float(source_fps or 0)
        except (TypeError, ValueError):
            sf = 0.0

        if sf <= 0:
            logger.warning("⚠️ 源帧率探测为 0，按目标帧率处理（不插值）")
            return 1
        if tf <= sf:
            logger.info(f"ℹ️ 目标 {tf}fps <= 源 {sf:g}fps，无需插值（倍数=1）")
            return 1
        return max(2, int(round(tf / sf)))

    def interpolate_video(
        self,
        input_video: str,
        output_video: str = None,
        target_fps: int = 24,
        source_fps: int = None,
        keep_audio: bool = True,
    ) -> str:
        """
        input_video : 输入 mp4/gif
        target_fps  : 目标帧率 (24/30/60)
        source_fps  : 原始帧率；None = 自动从视频探测
        keep_audio  : 是否把原音轨合回输出（默认保留）
        """
        if not self.is_available():
            raise FileNotFoundError(f"❌ RIFE 未安装: {self.exe}")
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"❌ 输入视频不存在: {input_video}")

        # 源帧率：入参优先，否则从视频探测（避免 0 除/半速）
        probed_fps, probed_cnt, _, _ = _probe_video_info(input_video)
        if source_fps is None:
            source_fps = probed_fps
        elif source_fps <= 0:
            source_fps = probed_fps

        multiplier = self.plan_multiplier(target_fps, source_fps)
        if multiplier <= 1:
            logger.info("ℹ️ [RIFE] 无需插值，直接返回原视频路径")
            return input_video

        if output_video is None:
            base = os.path.splitext(input_video)[0]
            output_video = f"{base}_rife{target_fps}fps.mp4"
        output_video = os.path.abspath(output_video)

        with _work_dir() as tmp:
            frames_in = os.path.join(tmp, "in")
            frames_out = os.path.join(tmp, "out")
            os.makedirs(frames_in, exist_ok=True)
            os.makedirs(frames_out, exist_ok=True)

            # === Step 1: 提取帧 ===
            logger.info("🎞️ [RIFE] 提取原始帧...")
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"❌ 无法打开视频: {input_video}")
            idx = 0
            bad = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or frame.size == 0:
                    bad += 1
                    continue
                if not _imwrite(os.path.join(frames_in, f"{idx:06d}.png"), frame):
                    bad += 1
                    continue
                idx += 1
            cap.release()
            if idx == 0:
                raise RuntimeError("❌ 未提取到任何帧（视频可能损坏或编码不支持）")
            logger.info(f"   ✅ 提取 {idx} 帧" + (f"（跳过 {bad} 帧）" if bad else ""))

            # === Step 2: 调 RIFE 插值 ===
            logger.info(f"🎞️ [RIFE] 插值 x{multiplier}（源 {source_fps:g}fps "
                        f"→ 目标 {target_fps}fps）...")
            cmd = [
                self.exe,
                "-i", frames_in,
                "-o", frames_out,
                "-m", self.model,
                "-n", str(idx * multiplier),
                "-f", "%06d.png",
            ]
            timeout = _RIFE_TIMEOUT_BASE + _RIFE_TIMEOUT_PER_FRAME * idx * multiplier
            try:
                result = subprocess.run(
                    cmd, capture_output=True, cwd=self.rife_dir,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"❌ RIFE 插值超时（>{timeout}s，共 {idx * multiplier} 帧）。\n"
                    f"   可减少帧数/降低目标帧率，或检查是否被安全软件拦截。")
            if result.returncode != 0:
                err = (result.stderr or b"").decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"RIFE 插值失败 (code={result.returncode}): {err[:500]}")

            out_frames = sorted(f for f in os.listdir(frames_out)
                                if f.lower().endswith((".png", ".jpg", ".jpeg")))
            if not out_frames:
                raise RuntimeError("RIFE 未产出任何帧")
            logger.info(f"   ✅ 生成 {len(out_frames)} 帧")

            # === Step 3: 合成视频 ===
            logger.info(f"🎞️ [RIFE] 合成 {target_fps}fps 视频...")
            first = _imread(os.path.join(frames_out, out_frames[0]))
            if first is None:
                raise RuntimeError("❌ 无法读取 RIFE 首帧")
            h, w = first.shape[:2]

            # 先写到临时文件，成功合音轨后再落到目标路径，避免半成品覆盖旧文件
            tmp_video = os.path.join(tmp, "noaudio.mp4")
            writer = cv2.VideoWriter(
                tmp_video, cv2.VideoWriter_fourcc(*"mp4v"),
                float(target_fps), (w, h))
            if not writer.isOpened():
                raise RuntimeError("❌ 无法创建视频写入器（缺少 mp4v 编码器？）")
            written = 0
            for fn in out_frames:
                img = _imread(os.path.join(frames_out, fn))
                if img is None:
                    continue
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, (w, h))
                writer.write(img)
                written += 1
            writer.release()
            if written == 0 or not os.path.exists(tmp_video) \
                    or os.path.getsize(tmp_video) == 0:
                raise RuntimeError("❌ 合成视频失败（0 帧或空文件）")

            # === Step 4: 保留音轨 ===
            # 旧实现只写视频流 → 原音轨被静默丢弃（配音白做）。
            dst_dir = os.path.dirname(output_video)
            if dst_dir:
                os.makedirs(dst_dir, exist_ok=True)
            muxed = False
            if keep_audio and _has_audio_stream(input_video):
                muxed = _mux_audio(tmp_video, input_video, output_video)
                if muxed:
                    logger.info("   ✅ 已保留原音轨")
                else:
                    logger.warning("   ⚠️ 音轨保留失败，输出将没有声音")
            if not muxed:
                shutil.move(tmp_video, output_video)

        logger.info(f"✅ [RIFE] 完成: {output_video}（{written} 帧 @ {target_fps}fps）")
        return output_video
