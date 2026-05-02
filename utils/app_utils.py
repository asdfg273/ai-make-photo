# app_utils.py
import re
import itertools

# === 全局常量 ===
OUTPUT_DIR = "photo"

# === 提示词魔法预设 ===
PROMPT_PRESETS = {
    "默认精美": {"p": "杰作, 最高画质, 极其精致的细节, 绝美光影, 8k分辨率, 电影级打光", "n": "低画质, 畸形, 扭曲, 糟糕的人体结构, 错误的比例, 模糊, 水印, 签名"},
    "二次元动漫": {"p": "杰作, 极致画质, 动漫风格, 绚丽的色彩, 京阿尼风格, 精致的五官, 唯美背景", "n": "真实照片, 3d, 丑陋, 崩坏, 多余的手指, 恐怖, 猎奇"},
    "电影级真实": {"p": "RAW照片, 极其真实的自然光, 电影感构图, 8k uhd, dslr, 柔和焦点, 胶片颗粒, 毛孔细节", "n": "动漫, 卡通, 画作, 插画, 虚假, 塑料质感, 过度平滑"},
    "赛博朋克": {"p": "赛博朋克风格, 霓虹灯, 未来城市, 高科技机甲, 杰作, 赛博格, 强烈的色彩对比", "n": "古风, 乡村, 简单背景, 模糊, 阳光明媚"},
    "唯美水墨": {"p": "中国水墨画风格, 意境深远, 留白, 极其优美的毛笔线条, 大师级杰作, 传统色彩", "n": "写实, 3d, 鲜艳的霓虹色, 机械, 现代"}
}

def parse_dynamic_prompt(prompt_text):
    """解析动态组合提示词，例如: {白发|黑发} 少女"""
    prompt_text = re.sub(r'<lora:[^>]+>', '', prompt_text)
    pattern = r'\{([^}]+)\}'
    matches = re.findall(pattern, prompt_text)
    if not matches:
        return [prompt_text]
    options_list = [match.split('|') for match in matches]
    combinations = list(itertools.product(*options_list))
    final_prompts = []
    for combo in combinations:
        temp_prompt = prompt_text
        for replacement in combo:
            temp_prompt = re.sub(pattern, replacement.strip(), temp_prompt, count=1)
        final_prompts.append(temp_prompt)
    return final_prompts