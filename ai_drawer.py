import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像

import torch
from diffusers import StableDiffusionPipeline
import json
import jieba
from deep_translator import GoogleTranslator

# ==========================================
# 第一部分：翻译与词典 (保持不变)
# ==========================================
DICT_FILE = "zh_to_en_dict.json"
default_dict = {
    "高质量": "best quality", "大师作": "masterpiece", "细节": "highly detailed",
    "女孩": "1girl", "男孩": "1boy", "单人": "solo",
    "低画质": "low quality", "最差画质": "worst quality", "坏手": "bad hands"
}

dictionary = {}

def load_dictionary():
    global dictionary
    if os.path.exists(DICT_FILE):
        try:
            with open(DICT_FILE, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
        except:
            dictionary = {}
    for k, v in default_dict.items():
        if k not in dictionary:
            dictionary[k] = v
    for word in dictionary.keys():
        jieba.add_word(word)

def save_dictionary():
    try:
        with open(DICT_FILE, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=4)
    except: pass

def translate_to_english(text):
    if not text.strip(): return ""
    has_new_word = False
    text = text.replace('，', ',').replace('。', ',').replace('\n', ',')
    segments = [seg.strip() for seg in text.split(',') if seg.strip()]
    final_parts = []
    for seg in segments:
        if all(ord(c) < 128 for c in seg): 
            final_parts.append(seg)
            continue
        if seg in dictionary: 
            final_parts.append(dictionary[seg])
            continue
        words = jieba.lcut(seg)
        seg_trans_words = []
        for word in words:
            word = word.strip()
            if not word or word in ["在", "的", "了", "着", "地", "个", "是", "不要"]: continue 
            if word in dictionary:
                seg_trans_words.append(dictionary[word])
            else:
                try:
                    res = GoogleTranslator(source='zh-CN', target='en').translate(word)
                    if res and res.strip():
                        seg_trans_words.append(res)
                        dictionary[word] = res
                        jieba.add_word(word) 
                        has_new_word = True
                    else: seg_trans_words.append(word)
                except: seg_trans_words.append(word)
        if seg_trans_words: final_parts.append(" ".join(seg_trans_words))
    if has_new_word: save_dictionary()
    return ", ".join(final_parts)

# ==========================================
# 第二部分：提示词与模型
# ==========================================
chinese_prompt = "大师作, 最高画质, 杰作, 单人, 一个美丽的女孩, 极其精致的面部, 完美的五官, 迷人的大眼睛, 看着镜头, 脸部特写, 唯美光影"
chinese_negative_prompt = "低画质, 最差画质, 畸形, 扭曲的脸, 坏眼, 斗鸡眼, 不对称的眼睛, 模糊, 恐怖, 肢体残缺, 多余的手指"

# 【重要】确保这里是你下载的那个 1.5 模型文件名！
MODEL_FILE_NAME = "meinamix_v12Final.safetensors"

# ==========================================
# 第三部分：100% 稳定的 CPU 绘图模式
# ==========================================
if __name__ == "__main__":
    print("1. 正在翻译你的中文...")
    load_dictionary()
    english_prompt = translate_to_english(chinese_prompt)
    english_negative = translate_to_english(chinese_negative_prompt)
    print(f"【翻译完成】: {english_prompt}")
    
    print("\n2. 正在加载 AI 模型 (CPU 绝对稳定模式)...")
    # CPU 原生支持 32 位浮点数，完全不会有精度报错
    pipe = StableDiffusionPipeline.from_single_file(
        f"./{MODEL_FILE_NAME}",
        torch_dtype=torch.float32, 
        use_safetensors=True,
        safety_checker=None,       
        requires_safety_checker=False
    )
    pipe.safety_checker = None
    
    # 全部强制使用 CPU！
    pipe.to("cpu")
    
    # 开启内存优化，防止撑爆你的电脑内存
    pipe.enable_attention_slicing()

    print("\n3. CPU 开始暴走画图！因为没有显卡加速，大约需要 5 到 10 分钟。")
    print("   去泡杯咖啡，刷会手机，只要看到进度条在动，就必定能出图！")
    with torch.no_grad():
        image = pipe(
            prompt=english_prompt,
            negative_prompt=english_negative,
            num_inference_steps=40, # 20 步足够了
            guidance_scale=7.0,     
            width=512,  
            height=512  
        ).images[0]

    image.save("my_output_image.png")
    print("\n4. 🎉 完美出图！快去文件夹里查看 my_output_image.png 吧！ 🎉")