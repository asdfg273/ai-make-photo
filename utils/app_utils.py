# app_utils.py
import re
import itertools

# === 全局常量 ===
from utils.paths import OUTPUT_DIR

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
