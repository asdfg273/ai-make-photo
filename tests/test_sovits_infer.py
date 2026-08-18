# test_sovits_infer.py
"""
GPT-SoVITS 最小合成测试
"""
import torchaudio
import soundfile as sf
import torch
import numpy as np

def _patched_load(filepath, *args, **kwargs):
    """用 soundfile 替代 torchaudio.load,避免 torchcodec 依赖"""
    data, sr = sf.read(str(filepath), dtype='float32', always_2d=True)
    # torchaudio 返回 shape=(channels, samples)
    tensor = torch.from_numpy(data.T).contiguous()
    return tensor, sr

torchaudio.load = _patched_load
print("🔧 已 patch torchaudio.load → soundfile")
import os
import sys
from pathlib import Path

# ========== 路径注入 ==========
ROOT = Path(__file__).parent
SOVITS_ROOT = ROOT / "third_party" / "GPT-SoVITS"
sys.path.insert(0, str(SOVITS_ROOT))
sys.path.insert(0, str(SOVITS_ROOT / "GPT_SoVITS"))
sys.path.insert(0, str(SOVITS_ROOT / "GPT_SoVITS" / "eres2net"))

# ========== 环境变量(GPT-SoVITS 要读) ==========
os.environ["is_half"] = "True"      # fp16
os.environ["is_share"] = "False"

# 权重路径(GPT-SoVITS v2)
PRETRAINED = SOVITS_ROOT / "GPT_SoVITS" / "pretrained_models"
GPT_PATH = PRETRAINED / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
SOVITS_PATH = PRETRAINED / "gsv-v2final-pretrained" / "s2G2333k.pth"

os.environ["gpt_path"] = str(GPT_PATH)
os.environ["sovits_path"] = str(SOVITS_PATH)
os.environ["cnhubert_base_path"] = str(PRETRAINED / "chinese-hubert-base")
os.environ["bert_path"] = str(PRETRAINED / "chinese-roberta-wwm-ext-large")

# ========== 检查文件 ==========
for name, p in [("GPT", GPT_PATH), ("SoVITS", SOVITS_PATH)]:
    if not p.exists():
        print(f"❌ 找不到 {name}: {p}")
        sys.exit(1)
    print(f"✅ {name}: {p.name}")

# ========== 导入 GPT-SoVITS 推理 ==========
print("\n📦 导入 GPT-SoVITS ...")
try:
    from GPT_SoVITS.inference_webui import (
        change_gpt_weights,
        change_sovits_weights,
        get_tts_wav,
    )
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\n❌ 导入失败: {e}")
    sys.exit(1)

print("✅ 导入成功")

# ========== 加载权重 ==========
print("\n⏳ 加载 GPT ...")
change_gpt_weights(gpt_path=str(GPT_PATH))
print("✅ GPT 就绪")

print("\n⏳ 加载 SoVITS ...")
change_sovits_weights(sovits_path=str(SOVITS_PATH))
print("✅ SoVITS 就绪")

# ========== 合成 ==========
REF_WAV = ROOT / "assets" / "voices" / "default_female_ja.wav"
REF_TEXT = "こんにちは、今日はいい天気ですね。散歩に行きませんか。"
TARGET_TEXT = "初めまして、私はアイです。よろしくお願いします。"

print(f"\n🎙️ 合成中 ...")
print(f"   参考音频: {REF_WAV.name}")
print(f"   目标文本: {TARGET_TEXT}")

gen = get_tts_wav(
    ref_wav_path=str(REF_WAV),
    prompt_text=REF_TEXT,
    prompt_language="日文",
    text=TARGET_TEXT,
    text_language="日文",
    how_to_cut="不切",
    top_k=15,
    top_p=1.0,
    temperature=1.0,
    ref_free=False,
)

# get_tts_wav 是生成器,yield (sr, audio)
sr, audio = None, None
for item in gen:
    sr, audio = item

# ========== 保存 ==========
import soundfile as sf
OUT = ROOT / "output" / "test_sovits.wav"
OUT.parent.mkdir(exist_ok=True)
sf.write(str(OUT), audio, sr)

print(f"\n✅ 完成! → {OUT}")
print(f"   采样率: {sr} Hz")
print(f"   时长: {len(audio)/sr:.2f}s")