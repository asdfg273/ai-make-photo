# app_utils.py
import re
import itertools

# === 全局常量 ===

# <lora:文件名:权重> / <lora:文件名> —— 社区通用写法
# 权重可省略，范围会被夹到 0~2
_LORA_TAG_RE = re.compile(r"<lora:([^:>]+?)(?::([\d.]+))?\s*>", re.IGNORECASE)


def parse_lora_tags(prompt_text):
    """从提示词里提取 `<lora:xxx:0.8>` 标签。

    返回 (去掉标签后的提示词, [{'name':..., 'weight':...}, ...])

    ⚠️ 旧实现里 `parse_dynamic_prompt` 只是把这些标签**静默删掉**，
    却没有把它们接到 LoRA 槽位上 —— 用户按社区习惯写 `<lora:xxx:0.8>`
    会以为生效了，实际什么都没发生。
    """
    if not prompt_text:
        return prompt_text, []

    found = []

    def _sub(m):
        name = (m.group(1) or "").strip()
        w = m.group(2)
        try:
            weight = float(w) if w not in (None, "") else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        weight = max(0.0, min(2.0, weight))
        if name:
            found.append({"name": name, "weight": weight})
        return ""

    cleaned = _LORA_TAG_RE.sub(_sub, prompt_text)
    # 清掉因删除标签留下的多余逗号/空白
    cleaned = re.sub(r"\s*,\s*(?=,)", "", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,\n\t")
    return cleaned, found


def parse_dynamic_prompt(prompt_text):
    """解析动态组合提示词，例如: {白发|黑发} 少女

    注意：`<lora:...>` 标签只在这里被移除；它们会被
    parse_lora_tags() 单独提取并挂到 LoRA 槽位。
    """
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
