import json
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AppConfig:
    config_file = "app_config.json"
    
    # 基础参数默认值
    default_steps: int = 30
    default_strength: float = 0.6
    default_lora_weight: float = 0.7
    adetailer_enabled: bool = False
    theme: str = "darkly"
    
    # 历史记录
    recent_models: List[str] = field(default_factory=list)
    recent_prompts: List[str] = field(default_factory=list)

    @classmethod
    def load(cls):
        """高容错加载：忽略废弃字段，补全缺失字段"""
        if os.path.exists(cls.config_file):
            try:
                with open(cls.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    config = cls()
                    # 只更新存在的字段，完美向下兼容
                    for key, value in data.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    return config
            except Exception as e:
                print(f"⚠️ 配置文件加载失败: {e}，将使用默认配置")
        return cls()

    def save(self):
        """保存配置到本地"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.__dict__, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存配置失败: {e}")

    def add_recent_model(self, model_name):
        if model_name and model_name not in self.recent_models:
            self.recent_models.insert(0, model_name)
            self.recent_models = self.recent_models[:10]  # 最多存10个
            self.save()  # 立即保存

    def add_recent_prompt(self, prompt):
        if prompt and prompt not in self.recent_prompts:
            self.recent_prompts.insert(0, prompt[:200])
            self.recent_prompts = self.recent_prompts[:20] # 最多存20个
            self.save()  # 立即保存