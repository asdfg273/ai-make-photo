# utils/model_downloader.py
"""
🌐 智能模型下载器
- 扫描项目缺失的必需/可选模型
- 支持 HuggingFace 镜像 (hf-mirror.com)
- CLI: python -m utils.model_downloader [scan|install-required|install-all|install <key>]
"""
import os
import sys
import shutil
from pathlib import Path

# 使用国内镜像加速
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
#  模型注册表
# ============================================================
# kind:
#   "snapshot" - 整个 HF 仓库 (snapshot_download)
#   "file"     - 单个文件 (hf_hub_download)
MODEL_REGISTRY = {
    # ---------- 必需 (视频功能核心) ----------
    "motion_adapter_v3": {
        "kind": "snapshot",
        "repo": "guoyww/animatediff-motion-adapter-v1-5-3",
        "local": "models/motion_adapter/v1-5-3",
        "check_file": "diffusion_pytorch_model.safetensors",
        "size_mb": 837,
        "required": True,
        "desc": "AnimateDiff MotionAdapter v1-5-3 (视频生成核心)",
    },
    # ---------- 必需 (Motion LoRA × 8) ----------
    **{
        f"motion_lora_{name}": {
            "kind": "snapshot",
            "repo": f"guoyww/animatediff-motion-lora-{name}",
            "local": f"models/motion_lora/{name}",
            "check_file": "diffusion_pytorch_model.safetensors",
            "size_mb": 74,
            "required": True,
            "desc": f"Motion LoRA - {name}",
        }
        for name in [
            "zoom-in", "zoom-out",
            "pan-left", "pan-right",
            "tilt-up", "tilt-down",
            "rolling-clockwise", "rolling-anticlockwise",
        ]
    },

    # ---------- 可选 (图生视频依赖) ----------
    "ip_adapter_sd15": {
        "kind": "file",
        "repo": "h94/IP-Adapter",
        "file": "models/ip-adapter_sd15.safetensors",
        "local": "models/ip_adapter/ip-adapter_sd15.safetensors",
        "check_file": None,  # 直接查 local
        "size_mb": 45,
        "required": False,
        "desc": "IP-Adapter SD1.5 (图生视频)",
    },
    "ip_adapter_image_encoder": {
        "kind": "snapshot",
        "repo": "h94/IP-Adapter",
        "allow_patterns": ["models/image_encoder/*"],
        "local": "models/ip_adapter/image_encoder",
        "repo_subdir": "models/image_encoder",  # HF 内部子路径
        "check_file": "model.safetensors",
        "size_mb": 2500,
        "required": False,
        "desc": "IP-Adapter Image Encoder (CLIP ViT-H)",
    },

    # ---------- 可选 (人脸/手部修复) ----------
    "adetailer_face": {
        "kind": "file",
        "repo": "Bingsu/adetailer",
        "file": "face_yolov8n.pt",
        "local": "models/adetailer/face_yolov8n.pt",
        "check_file": None,
        "size_mb": 6,
        "required": False,
        "desc": "ADetailer 人脸检测 (YOLOv8n)",
    },
    "adetailer_hand": {
        "kind": "file",
        "repo": "Bingsu/adetailer",
        "file": "hand_yolov8n.pt",
        "local": "models/adetailer/hand_yolov8n.pt",
        "check_file": None,
        "size_mb": 6,
        "required": False,
        "desc": "ADetailer 手部检测 (YOLOv8n)",
    },
}


# ============================================================
#  扫描
# ============================================================
def _check_exists(entry: dict) -> bool:
    local = PROJECT_ROOT / entry["local"]
    if entry["kind"] == "file":
        return local.is_file() and local.stat().st_size > 1024
    # snapshot
    check_file = entry.get("check_file")
    if check_file:
        return (local / check_file).is_file()
    return local.is_dir() and any(local.iterdir())


def scan() -> dict:
    """扫描项目,返回 {key: {"ok": bool, "entry": dict}}"""
    result = {}
    for key, entry in MODEL_REGISTRY.items():
        result[key] = {
            "ok": _check_exists(entry),
            "entry": entry,
        }
    return result


def print_scan_report():
    """打印扫描报告(供 main.py 启动调用)"""
    report = scan()
    ok_cnt = sum(1 for v in report.values() if v["ok"])
    miss_required = [k for k, v in report.items()
                     if not v["ok"] and v["entry"]["required"]]
    miss_optional = [k for k, v in report.items()
                     if not v["ok"] and not v["entry"]["required"]]

    print(f"📊 模型扫描: 就绪 {ok_cnt}/{len(report)}")
    if miss_required:
        print(f"   ❌ 缺失必需 ({len(miss_required)}):")
        for k in miss_required:
            e = report[k]["entry"]
            print(f"      • {e['desc']} ({e['size_mb']}MB)")
        print(f"   💡 运行: python -m utils.model_downloader install-required")
    if miss_optional:
        print(f"   ⚠️  缺失可选 ({len(miss_optional)}):")
        for k in miss_optional:
            e = report[k]["entry"]
            print(f"      • {e['desc']} ({e['size_mb']}MB) [key={k}]")
    return report


# ============================================================
#  下载
# ============================================================
def _download_one(key: str, entry: dict) -> bool:
    from huggingface_hub import hf_hub_download, snapshot_download

    local = PROJECT_ROOT / entry["local"]
    print(f"\n⬇️  下载: {entry['desc']} ({entry['size_mb']}MB)")
    print(f"    源:   {entry['repo']}")
    print(f"    目标: {local}")

    try:
        if entry["kind"] == "file":
            local.parent.mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(
                repo_id=entry["repo"],
                filename=entry["file"],
                local_dir=str(local.parent),
            )
            # hf_hub_download 会保留子目录结构,需要移动到目标位置
            downloaded = Path(path)
            if downloaded.resolve() != local.resolve():
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(local))
                # 清理空的中间目录
                try:
                    downloaded.parent.rmdir()
                except OSError:
                    pass
        else:
            # snapshot
            local.mkdir(parents=True, exist_ok=True)
            allow_patterns = entry.get("allow_patterns")

            if allow_patterns:
                # 只下载部分文件,需要把内容"扁平化"到 local
                tmp = PROJECT_ROOT / f".tmp_dl_{key}"
                snapshot_download(
                    repo_id=entry["repo"],
                    local_dir=str(tmp),
                    allow_patterns=allow_patterns,
                )
                # 把 tmp/repo_subdir/* 移到 local/
                subdir = entry.get("repo_subdir", "")
                src = tmp / subdir if subdir else tmp
                for item in src.iterdir():
                    dst = local / item.name
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()
                    shutil.move(str(item), str(dst))
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                snapshot_download(
                    repo_id=entry["repo"],
                    local_dir=str(local),
                )

        if _check_exists(entry):
            print(f"✅ 完成: {key}")
            return True
        else:
            print(f"⚠️  下载完成但校验失败: {key}")
            return False

    except Exception as e:
        print(f"❌ 下载失败: {key} → {e}")
        return False


def install_required():
    report = scan()
    missing = [(k, v["entry"]) for k, v in report.items()
               if not v["ok"] and v["entry"]["required"]]
    if not missing:
        print("✅ 所有必需模型已就绪")
        return
    print(f"📥 待下载必需模型: {len(missing)} 个")
    for k, e in missing:
        _download_one(k, e)


def install_all():
    report = scan()
    missing = [(k, v["entry"]) for k, v in report.items() if not v["ok"]]
    if not missing:
        print("✅ 所有模型已就绪")
        return
    total_mb = sum(e["size_mb"] for _, e in missing)
    print(f"📥 待下载 {len(missing)} 个,共约 {total_mb} MB")
    for k, e in missing:
        _download_one(k, e)


def install(key: str):
    if key not in MODEL_REGISTRY:
        print(f"❌ 未知模型 key: {key}")
        print(f"   可用: {list(MODEL_REGISTRY.keys())}")
        return
    entry = MODEL_REGISTRY[key]
    if _check_exists(entry):
        print(f"✅ 已存在: {key}")
        return
    _download_one(key, entry)


# ============================================================
#  CLI
# ============================================================
def _main():
    if len(sys.argv) < 2:
        print("用法: python -m utils.model_downloader [scan|install-required|install-all|install <key>]")
        return
    cmd = sys.argv[1]
    if cmd == "scan":
        print_scan_report()
    elif cmd == "install-required":
        install_required()
    elif cmd == "install-all":
        install_all()
    elif cmd == "install":
        if len(sys.argv) < 3:
            print("用法: python -m utils.model_downloader install <key>")
            return
        install(sys.argv[2])
    else:
        print(f"❌ 未知命令: {cmd}")


if __name__ == "__main__":
    _main()