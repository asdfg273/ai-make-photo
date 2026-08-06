# scripts/download_sovits.py
"""
🎌 GPT-SoVITS v2 一键下载脚本
下载官方预训练权重到 third_party/GPT-SoVITS/
"""
import os
import sys
import shutil
from pathlib import Path

# 使用 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent.parent
SOVITS_DIR = ROOT / "third_party" / "GPT-SoVITS"
PRETRAINED_DIR = SOVITS_DIR / "GPT_SoVITS" / "pretrained_models"


# ============================================================
#  需要下载的模型清单
# ============================================================
MODELS = [
    {
        "repo": "lj1995/GPT-SoVITS",
        "files": [
            # 主模型 (v2)
            "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "gsv-v2final-pretrained/s2G2333k.pth",
            "gsv-v2final-pretrained/s2D2333k.pth",
            # 中文 HuBERT
            "chinese-hubert-base/config.json",
            "chinese-hubert-base/preprocessor_config.json",
            "chinese-hubert-base/pytorch_model.bin",
            # 中文 RoBERTa
            "chinese-roberta-wwm-ext-large/config.json",
            "chinese-roberta-wwm-ext-large/tokenizer.json",
            "chinese-roberta-wwm-ext-large/pytorch_model.bin",
        ],
        "target": PRETRAINED_DIR,
    },
]


def _download_file(repo_id: str, filename: str, target_dir: Path) -> bool:
    """从 HuggingFace 下载单个文件"""
    from huggingface_hub import hf_hub_download
    
    target_path = target_dir / filename
    if target_path.exists() and target_path.stat().st_size > 1024:
        print(f"  ✅ 已存在: {filename}")
        return True
    
    print(f"  ⬇️  下载: {filename}")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"  ✅ 完成: {filename}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {filename} — {e}")
        return False


def main():
    print("=" * 60)
    print("🎌 GPT-SoVITS v2 下载器")
    print("=" * 60)
    print(f"📁 目标目录: {PRETRAINED_DIR}")
    print(f"🌐 镜像源: {os.environ.get('HF_ENDPOINT')}")
    print()
    
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    
    total = 0
    success = 0
    for group in MODELS:
        repo = group["repo"]
        target = group["target"]
        print(f"\n📦 从 {repo} 下载:")
        for f in group["files"]:
            total += 1
            if _download_file(repo, f, target):
                success += 1
    
    print()
    print("=" * 60)
    print(f"📊 完成: {success}/{total}")
    print("=" * 60)
    
    if success == total:
        print("✅ 全部下载完成!")
        print(f"📂 位置: {PRETRAINED_DIR}")
    else:
        print("⚠️  部分文件下载失败,请重跑此脚本(会自动跳过已下载的)")
        sys.exit(1)


if __name__ == "__main__":
    main()