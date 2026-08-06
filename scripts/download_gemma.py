import os
from huggingface_hub import snapshot_download

# ========== 配置项 ==========
# 模型仓库ID（请确认拼写完全准确）
MODEL_REPO = "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced"
# 本地保存路径，使用原始字符串避免Windows转义问题
SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_cache", "gemma4")
# 切换国内镜像，解决跨境无法访问的问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ============================

def download_model():
    print(f"开始下载模型: {MODEL_REPO}")
    print(f"保存路径: {SAVE_PATH}")
    print(f"下载镜像: {os.environ['HF_ENDPOINT']}")
    
    try:
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=SAVE_PATH,
            local_dir_use_symlinks=False,  # Windows关闭符号链接，避免权限报错
            resume_download=True,         # 开启断点续传，中断后重跑可继续
            max_workers=4                 # 并发下载数，可根据网络情况调整
        )
        print("\n✅ 模型下载完成！")
    except Exception as e:
        print(f"\n❌ 下载出错: {str(e)}")

if __name__ == "__main__":
    download_model()
