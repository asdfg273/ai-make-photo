# utils/extension_manager.py
"""
🧩 扩展管理器
- 集中管理所有可选扩展 (模型/工具)
- 检测安装状态
- 下载 / 卸载
- 供 GUI 面板调用
"""
import os
import sys
import shutil
import zipfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# hf-mirror 优先,失败回退官方
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from utils.paths import PROJECT_ROOT


def _proj(p: str) -> str:
    """相对路径一律以项目根目录为锚,返回绝对路径字符串"""
    return str(_resolve_path(p))


# ============================================================
#  扩展定义
# ============================================================
EXTENSIONS = {
    # ---------- 图片增强 ----------
    "qwen2vl_2b": {
        "name": "Qwen2-VL-2B 提示词增强",
        "category": "图片增强",
        "size_mb": 4500,
        "required": False,
        "desc": "AI 提示词改写、图像识别、中日翻译",
        "check": [
            "models_cache/modelscope/Qwen/Qwen2-VL-2B-Instruct",
            "models_cache/models--Qwen--Qwen2-VL-2B-Instruct",
        ],
        "check_mode": "any",
    },
    "nllb_600m": {
        "name": "NLLB-200 翻译 (中↔日)",
        "category": "图片增强",
        "size_mb": 2500,
        "required": False,
        "desc": "高质量中日双向翻译,用于日语配音",
        "check": ["models_cache/models--facebook--nllb-200-distilled-600M"],
    },

    # ---------- 控制 ----------
    "controlnet_openpose": {
        "name": "ControlNet 姿势 (OpenPose)",
        "category": "控制",
        "size_mb": 1400,
        "required": False,
        "desc": "通过骨骼图控制人物姿势",
        "check_any": [
            "models_cache/models--lllyasviel--sd-controlnet-openpose",
            "models_cache/huggingface/hub/models--lllyasviel--sd-controlnet-openpose",
            os.path.expanduser("~/.cache/huggingface/hub/models--lllyasviel--sd-controlnet-openpose"),
        ],
        "download": lambda cb=None: _dl_controlnet(cb, "openpose"),
    },
    "controlnet_canny": {
        "name": "ControlNet 边缘 (Canny)",
        "category": "控制",
        "size_mb": 1400,
        "required": False,
        "desc": "通过 Canny 边缘图控制构图",
        "check_any": [
            "models_cache/models--lllyasviel--sd-controlnet-canny",
            "models_cache/huggingface/hub/models--lllyasviel--sd-controlnet-canny",
            os.path.expanduser("~/.cache/huggingface/hub/models--lllyasviel--sd-controlnet-canny"),
        ],
        "download": lambda cb=None: _dl_controlnet(cb, "canny"),
    },
    "controlnet_depth": {
        "name": "ControlNet 深度",
        "category": "控制",
        "size_mb": 1400,
        "required": False,
        "desc": "通过深度图控制场景空间",
        "check_any": [
            "models_cache/models--lllyasviel--sd-controlnet-depth",
            "models_cache/huggingface/hub/models--lllyasviel--sd-controlnet-depth",
            os.path.expanduser("~/.cache/huggingface/hub/models--lllyasviel--sd-controlnet-depth"),
        ],
    },

    # ---------- 修复 ----------
    "adetailer_face": {
        "name": "ADetailer 人脸修复",
        "category": "修复",
        "size_mb": 6,
        "required": False,
        "desc": "自动检测人脸并重绘 (YOLOv8n)",
        "check": ["models/adetailer/face_yolov8n.pt"],
    },
    "adetailer_hand": {
        "name": "ADetailer 手部修复",
        "category": "修复",
        "size_mb": 6,
        "required": False,
        "desc": "自动检测手部并重绘 (YOLOv8n)",
        "check": ["models/adetailer/hand_yolov8n.pt"],
    },

    # ---------- 视频 ----------
    "motion_adapter_v3": {
        "name": "AnimateDiff MotionAdapter v1-5-3",
        "category": "视频",
        "size_mb": 1700,
        "required": True,
        "desc": "视频生成的动态适配器 (必需)",
        "check": ["models/motion_adapter/v1-5-3/diffusion_pytorch_model.safetensors"],
    },
    "motion_lora_pack": {
        "name": "Motion LoRA 运镜包 (8 种)",
        "category": "视频",
        "size_mb": 600,
        "required": False,
        "desc": "推/拉/摇/移/滚 8 种运镜特效",
        "check": [
            "models/motion_lora/zoom-in",
            "models/motion_lora/zoom-out",
            "models/motion_lora/pan-left",
            "models/motion_lora/pan-right",
            "models/motion_lora/tilt-up",
            "models/motion_lora/tilt-down",
            "models/motion_lora/rolling-clockwise",
            "models/motion_lora/rolling-anticlockwise",
        ],
        "check_mode": "any",
    },
    "ip_adapter_sd15": {
        "name": "IP-Adapter SD1.5 (图生视频)",
        "category": "视频",
        "size_mb": 2500,
        "required": False,
        "desc": "参考图 → 视频,含 CLIP-ViT-H 图像编码器",
        "check": [
            "models/ip_adapter/ip-adapter_sd15.safetensors",
            "models/ip_adapter/image_encoder",
        ],
        "check_mode": "all",
    },
    "rife_v46": {
        "name": "RIFE 视频插帧 v4.6",
        "category": "视频",
        "size_mb": 50,
        "required": False,
        "desc": "视频插帧提升流畅度 (8fps → 32fps)",
        "check": [
            "tools/rife/rife-ncnn-vulkan.exe",
            "tools/rife/rife-v4.6/flownet.bin",
        ],
        "check_mode": "all",
    },

    # ---------- 音频 ----------
    "chattts": {
        "name": "ChatTTS 中文语音合成",
        "category": "音频",
        "size_mb": 1100,
        "required": False,
        "desc": "中文 TTS,支持男/女多种音色",
        "check": [
            "~/.cache/huggingface/hub/models--2Noise--ChatTTS",
            "models_cache/models--2Noise--ChatTTS",
        ],
        "check_mode": "any",
    },
    "gpt_sovits": {
        "name": "GPT-SoVITS 日语克隆 TTS",
        "category": "音频",
        "size_mb": 3000,
        "required": False,
        "desc": "日语零样本语音克隆,支持上传参考音频",
        "check": [
            "third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "third_party/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        ],
        "check_mode": "all",
    },
}


# ============================================================
#  路径检测
# ============================================================
def _resolve_path(p: str) -> Path:
    """处理 ~ 展开 + 相对路径（相对路径以项目根目录为锚,不依赖 CWD）"""
    p = os.path.expanduser(str(p))
    path = Path(p)
    if not path.is_absolute():
        path = Path(PROJECT_ROOT) / path
    return path.resolve()


def is_installed(ext_id: str) -> bool:
    """
    检测扩展是否已安装
    - 优先看 check_any: 任一路径存在即视为已装
    - 否则看 check + check_mode (默认 all)
    """
    ext = EXTENSIONS.get(ext_id)
    if not ext:
        return False

    # check_any: 任一路径存在即算装(用于 HF 缓存等多位置)
    if "check_any" in ext:
        return any(_resolve_path(p).exists() for p in ext["check_any"])

    # 标准 check + check_mode
    paths = ext.get("check", [])
    if not paths:
        return False
    mode = ext.get("check_mode", "all")
    results = [_resolve_path(p).exists() for p in paths]
    return any(results) if mode == "any" else all(results)


# 兼容旧调用
def check_installed(ext_id: str) -> bool:
    return is_installed(ext_id)

def get_status_summary() -> dict:
    """返回扫描汇总: {installed: n, total: n, by_category: {...}}"""
    total = len(EXTENSIONS)
    installed = 0
    by_cat = {}
    for ext_id, ext in EXTENSIONS.items():
        ok = is_installed(ext_id)
        if ok:
            installed += 1
        cat = ext.get("category", "其他")
        by_cat.setdefault(cat, {"installed": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if ok:
            by_cat[cat]["installed"] += 1
    return {"installed": installed, "total": total, "by_category": by_cat}


# ============================================================
#  CLI 扫描
# ============================================================
def print_scan():
    """CLI 扫描输出"""
    print("=" * 70)
    print("🧩 扩展安装状态扫描")
    print("=" * 70)
    print()

    by_cat = {}
    for ext_id, info in EXTENSIONS.items():
        by_cat.setdefault(info.get("category", "其他"), []).append((ext_id, info))

    installed_total = 0
    total = len(EXTENSIONS)

    for cat, items in by_cat.items():
        print(f"📂 {cat}")
        for ext_id, info in items:
            ok = is_installed(ext_id)
            if ok:
                installed_total += 1
            icon = "✅" if ok else "⚪"
            size = info.get("size_mb", 0)
            print(f"  {icon} {info['name']} — {size}MB")
            print(f"      id: {ext_id}")
        print()

    print("=" * 70)
    print(f"📊 汇总: {installed_total}/{total} 已安装")
    print("=" * 70)


# ============================================================
#  下载引擎
# ============================================================
ProgressCallback = Optional[Callable[[float, str], None]]


def _emit(cb: ProgressCallback, pct: float, msg: str):
    """统一进度输出"""
    print(f"[{pct:5.1f}%] {msg}")
    if cb:
        try:
            cb(pct, msg)
        except Exception:
            pass


# ------------------------------------------------------------
#  通用下载器
# ------------------------------------------------------------

def _hf_snapshot(repo_id, local_dir, cb, allow_patterns=None, ignore_patterns=None):
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    endpoints = ["https://hf-mirror.com", "https://huggingface.co"]
    last_err = None
    for ep in endpoints:
        os.environ["HF_ENDPOINT"] = ep
        _emit(cb, 5, f"尝试源: {ep}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=4,
            )
            _emit(cb, 100, f"✅ {repo_id} 完成")
            return
        except Exception as e:
            last_err = e
            _emit(cb, 5, f"⚠️ {ep} 失败: {e}")
    raise RuntimeError(f"所有源均失败: {last_err}")


def _hf_single_file(repo_id, filename, target_path, cb):
    from huggingface_hub import hf_hub_download
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    endpoints = ["https://hf-mirror.com", "https://huggingface.co"]
    last_err = None
    for ep in endpoints:
        os.environ["HF_ENDPOINT"] = ep
        _emit(cb, 5, f"尝试源: {ep}")
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(target_path),
                local_dir_use_symlinks=False,
            )
            if os.path.abspath(downloaded) != os.path.abspath(target_path):
                shutil.move(downloaded, target_path)
            _emit(cb, 100, f"✅ {filename} 完成")
            return
        except Exception as e:
            last_err = e
            _emit(cb, 5, f"⚠️ {ep} 失败: {e}")
    raise RuntimeError(f"所有源均失败: {last_err}")


def _ms_snapshot(repo_id, local_dir, cb):
    from modelscope import snapshot_download as ms_download
    os.makedirs(local_dir, exist_ok=True)
    _emit(cb, 5, f"ModelScope: {repo_id}")
    ms_download(repo_id, cache_dir=local_dir)
    _emit(cb, 100, f"✅ {repo_id} 完成")


def _url_download(url, target_path, cb):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    def _hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100 / total_size)
        mb_done = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        _emit(cb, pct, f"下载中 {mb_done:.1f}/{mb_total:.1f} MB")

    urllib.request.urlretrieve(url, target_path, _hook)


# ------------------------------------------------------------
#  各扩展下载实现
# ------------------------------------------------------------

def _dl_qwen2vl_2b(cb):
    _ms_snapshot("qwen/Qwen2-VL-2B-Instruct", _proj("models_cache/modelscope"), cb)


def _dl_nllb_600m(cb):
    _hf_snapshot(
        "facebook/nllb-200-distilled-600M",
        _proj("models_cache/models--facebook--nllb-200-distilled-600M"),
        cb,
        ignore_patterns=["*.msgpack", "*.h5", "flax_*"],
    )


def _dl_controlnet(cb, control_type):
    _hf_snapshot(
        f"lllyasviel/sd-controlnet-{control_type}",
        _proj(f"models_cache/models--lllyasviel--sd-controlnet-{control_type}"),
        cb,
    )


def _dl_adetailer(cb, kind):
    fname = f"{kind}_yolov8n.pt"
    _hf_single_file("Bingsu/adetailer", fname, _proj(f"models/adetailer/{fname}"), cb)


def _dl_motion_adapter_v3(cb):
    _hf_snapshot("guoyww/animatediff-motion-adapter-v1-5-3",
                 _proj("models/motion_adapter/v1-5-3"), cb)


def _dl_motion_lora_pack(cb):
    loras = [
        "zoom-in", "zoom-out",
        "pan-left", "pan-right",
        "tilt-up", "tilt-down",
        "rolling-clockwise", "rolling-anticlockwise",
    ]
    for i, name in enumerate(loras):
        pct = (i / len(loras)) * 100
        _emit(cb, pct, f"下载 Motion LoRA: {name} ({i+1}/{len(loras)})")
        _hf_snapshot(f"guoyww/animatediff-motion-lora-{name}",
                     _proj(f"models/motion_lora/{name}"), None)
    _emit(cb, 100, "✅ 全部 Motion LoRA 完成")


def _dl_ip_adapter_sd15(cb):
    _emit(cb, 5, "下载 IP-Adapter 主权重 ...")
    _hf_single_file(
        "h94/IP-Adapter",
        "models/ip-adapter_sd15.safetensors",
        _proj("models/ip_adapter/ip-adapter_sd15.safetensors"), cb,
    )
    _emit(cb, 50, "下载 image_encoder ...")
    _hf_snapshot(
        "h94/IP-Adapter",
        _proj("models/ip_adapter/_tmp_encoder"),
        None,
        allow_patterns=["models/image_encoder/*"],
    )
    src = _proj("models/ip_adapter/_tmp_encoder/models/image_encoder")
    dst = _proj("models/ip_adapter/image_encoder")
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(src, dst)
    shutil.rmtree(_proj("models/ip_adapter/_tmp_encoder"), ignore_errors=True)
    _emit(cb, 100, "✅ IP-Adapter 完成")


def _dl_rife_v46(cb):
    url = "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip"
    tmp_zip = os.path.join(tempfile.gettempdir(), "rife_v46.zip")
    _emit(cb, 5, "下载 RIFE zip ...")
    _url_download(url, tmp_zip, cb)
    _emit(cb, 90, "解压 ...")
    extract_root = _proj("tools/rife/_extract")
    with zipfile.ZipFile(tmp_zip) as zf:
        # zip-slip 防护: 拒绝逃逸出解压目录的条目
        real_root = os.path.realpath(extract_root)
        for member in zf.namelist():
            target = os.path.realpath(os.path.join(real_root, member))
            if target != real_root and not target.startswith(real_root + os.sep):
                raise RuntimeError(f"zip 包含非法路径条目: {member}")
        zf.extractall(extract_root)
    rife_dir = _proj("tools/rife")
    for root, dirs, files in os.walk(extract_root):
        for f in files:
            if f.endswith(".exe"):
                shutil.copy(os.path.join(root, f), rife_dir)
        for d in dirs:
            if d.startswith("rife-v4.6") or d == "rife-v4.6":
                src = os.path.join(root, d)
                dst = os.path.join(rife_dir, "rife-v4.6")
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
    shutil.rmtree(extract_root, ignore_errors=True)
    try:
        os.remove(tmp_zip)
    except Exception:
        pass
    _emit(cb, 100, "✅ RIFE 完成")


def _dl_chattts(cb):
    default_cache = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(default_cache, exist_ok=True)
    _hf_snapshot("2Noise/ChatTTS",
                 os.path.join(default_cache, "models--2Noise--ChatTTS"), cb)


def _dl_gpt_sovits(cb):
    _emit(cb, 5, "调用 scripts/download_sovits.py ...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/download_sovits.py"],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPT-SoVITS 下载失败:\n{result.stderr}")
    _emit(cb, 100, "✅ GPT-SoVITS 完成")


# ------------------------------------------------------------
#  下载调度中心
# ------------------------------------------------------------

DOWNLOADERS = {
    "qwen2vl_2b":            _dl_qwen2vl_2b,
    "nllb_600m":             _dl_nllb_600m,
    "controlnet_openpose":   lambda cb: _dl_controlnet(cb, "openpose"),
    "controlnet_canny":      lambda cb: _dl_controlnet(cb, "canny"),
    "controlnet_depth":      lambda cb: _dl_controlnet(cb, "depth"),
    "adetailer_face":        lambda cb: _dl_adetailer(cb, "face"),
    "adetailer_hand":        lambda cb: _dl_adetailer(cb, "hand"),
    "motion_adapter_v3":     _dl_motion_adapter_v3,
    "motion_lora_pack":      _dl_motion_lora_pack,
    "ip_adapter_sd15":       _dl_ip_adapter_sd15,
    "rife_v46":              _dl_rife_v46,
    "chattts":               _dl_chattts,
    "gpt_sovits":            _dl_gpt_sovits,
}


def download_extension(ext_id: str, progress_cb: ProgressCallback = None):
    """
    下载指定扩展
    :param ext_id: 扩展 id
    :param progress_cb: 进度回调 fn(pct: float, msg: str)
    """
    if ext_id not in DOWNLOADERS:
        raise ValueError(f"未知扩展: {ext_id}")

    if is_installed(ext_id):
        _emit(progress_cb, 100, f"✔️ 已安装: {ext_id}")
        return True

    _emit(progress_cb, 0, f"开始下载: {ext_id}")
    try:
        DOWNLOADERS[ext_id](progress_cb)
        _emit(progress_cb, 100, f"✅ 完成: {ext_id}")
        return True
    except Exception as e:
        _emit(progress_cb, 0, f"❌ 失败: {e}")
        raise


# ------------------------------------------------------------
#  卸载
# ------------------------------------------------------------

def uninstall_extension(ext_id: str, progress_cb: ProgressCallback = None) -> bool:
    """
    卸载扩展 —— 删除 check 中列出的所有文件/目录
    """
    if ext_id not in EXTENSIONS:
        _emit(progress_cb, 0, f"❌ 未知扩展: {ext_id}")
        return False

    ext = EXTENSIONS[ext_id]
    paths = ext.get("check", [])
    if not paths:
        _emit(progress_cb, 0, f"⚠️ {ext_id} 未定义 check 路径,无法卸载")
        return False

    _emit(progress_cb, 5, f"🗑️  开始卸载: {ext['name']}")
    deleted = []
    dirs_to_check = set()

    for p in paths:
        path = _resolve_path(p)
        if path.exists():
            try:
                if path.is_file():
                    path.unlink()
                    deleted.append(str(path))
                    dirs_to_check.add(path.parent)
                elif path.is_dir():
                    shutil.rmtree(path)
                    deleted.append(str(path))
            except Exception as e:
                _emit(progress_cb, 50, f"⚠️ 删除失败 {path}: {e}")

    # 清理空的父目录 (向上最多 3 层)
    for d in dirs_to_check:
        try:
            for _ in range(3):
                if d.exists() and d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
                    d = d.parent
                else:
                    break
        except Exception:
            pass

    if deleted:
        _emit(progress_cb, 100, f"✅ 已卸载 {len(deleted)} 个文件/目录")
        return True
    else:
        _emit(progress_cb, 100, f"⚠️ 未找到要删除的文件")
        return False


# ------------------------------------------------------------
#  CLI 支持
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="扩展管理器")
    parser.add_argument("action", choices=["scan", "list", "install", "uninstall"])
    parser.add_argument("--id", help="扩展 ID (install/uninstall 时必填)")
    args = parser.parse_args()

    if args.action == "scan":
        print_scan()

    elif args.action == "list":
        print("\n📋 所有扩展:\n")
        by_cat = {}
        for ext_id, ext in EXTENSIONS.items():
            by_cat.setdefault(ext["category"], []).append((ext_id, ext))
        for cat, items in by_cat.items():
            print(f"📂 {cat}")
            for ext_id, ext in items:
                status = "✅" if is_installed(ext_id) else "⚪"
                size = ext.get("size_mb", 0)
                print(f"  {status} {ext_id:22s} — {ext['name']} ({size}MB)")
            print()

    elif args.action == "install":
        if not args.id:
            print("❌ 请用 --id 指定扩展")
            sys.exit(1)
        download_extension(args.id)

    elif args.action == "uninstall":
        if not args.id:
            print("❌ 请用 --id 指定扩展")
            sys.exit(1)
        uninstall_extension(args.id)