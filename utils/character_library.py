# utils/character_library.py
"""
角色库管理模块
- 角色数据结构
- 存储/加载
- 缩略图生成
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from PIL import Image
import copy
import shutil

logger = logging.getLogger(__name__)


@dataclass
class Character:
    """角色数据结构"""
    name: str
    ref_image_path: str = ""
    
    # LoRA 配置
    lora_path: str = ""
    lora_weight: float = 0.75
    
    # IP-Adapter 配置
    ip_adapter_scale: float = 0.6
    
    # ControlNet 配置（可选）
    controlnet_image_path: str = ""
    controlnet_type: str = ""  # openpose, canny, depth 等
    controlnet_scale: float = 1.0
    
    # 提示词
    base_prompt: str = ""
    negative_prompt: str = ""
    
    # 特征缓存（可选，用于加速）
    extracted_features: Optional[dict] = None
    
    # 元数据
    uuid: str = ""  # 唯一标识
    created_at: str = ""
    modified_at: str = ""
    thumbnail_path: str = ""
    tags: List[str] = field(default_factory=list)  # 分类标签
    notes: str = ""  # 备注
    
    def __post_init__(self):
        if not self.uuid:
            import uuid
            self.uuid = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.modified_at:
            self.modified_at = self.created_at
    
    def to_dict(self) -> dict:
        """转为字典（用于 JSON 序列化）"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Character':
        """从字典恢复"""
        # 处理旧版本兼容（extracted_features 可能是 None）
        if data.get('extracted_features') is None:
            data['extracted_features'] = None
        return cls(**data)


class CharacterLibrary:
    """角色库管理器"""
    
    def __init__(self, library_dir: str = "data/characters"):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        self.thumbnails_dir = self.library_dir / ".thumbnails"
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        self.characters: dict[str, Character] = {}  # uuid -> Character
        self._load_all()
    
    def _load_all(self):
        """加载所有角色"""
        self.characters.clear()
        for json_file in self.library_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                char = Character.from_dict(data)
                self.characters[char.uuid] = char
                logger.info(f"✅ 加载角色: {char.name}")
            except Exception as e:
                logger.error(f"❌ 加载角色失败 {json_file}: {e}")
    
    def save_character(self, char: Character) -> bool:
        """保存角色"""
        try:
            char.modified_at = datetime.now().isoformat()
            
            # 生成缩略图
            if char.ref_image_path and Path(char.ref_image_path).exists():
                char.thumbnail_path = self._generate_thumbnail(
                    char.ref_image_path, char.uuid
                )
            
            # 保存 JSON
            json_path = self.library_dir / f"{char.uuid}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(char.to_dict(), f, ensure_ascii=False, indent=2)
            
            self.characters[char.uuid] = char
            logger.info(f"💾 保存角色: {char.name}")
            return True
        except Exception as e:
            logger.error(f"❌ 保存角色失败: {e}")
            return False
    
    def delete_character(self, uuid: str) -> bool:
        """删除角色"""
        try:
            char = self.characters.get(uuid)
            if not char:
                return False
            
            # 删除 JSON
            json_path = self.library_dir / f"{uuid}.json"
            if json_path.exists():
                json_path.unlink()
            
            # 删除缩略图
            if char.thumbnail_path and Path(char.thumbnail_path).exists():
                Path(char.thumbnail_path).unlink()
            
            del self.characters[uuid]
            logger.info(f"🗑️ 删除角色: {char.name}")
            return True
        except Exception as e:
            logger.error(f"❌ 删除角色失败: {e}")
            return False
    
    def get_character(self, uuid: str) -> Optional[Character]:
        """获取角色"""
        return self.characters.get(uuid)
    
    def list_characters(self, tag: Optional[str] = None) -> List[Character]:
        """列出所有角色（可按标签筛选）"""
        chars = list(self.characters.values())
        if tag:
            chars = [c for c in chars if tag in c.tags]
        return sorted(chars, key=lambda c: c.modified_at, reverse=True)
    
    def _generate_thumbnail(self, image_path: str, uuid: str) -> str:
        """生成缩略图"""
        try:
            thumb_path = self.thumbnails_dir / f"{uuid}.jpg"
            img = Image.open(image_path)
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img.convert('RGB').save(thumb_path, 'JPEG', quality=85)
            return str(thumb_path)
        except Exception as e:
            logger.error(f"❌ 生成缩略图失败: {e}")
            return ""
    
    def export_character(self, uuid: str, export_path: str) -> bool:
        try:
            src = self.characters.get(uuid)
            if not src:
                return False
            char = copy.deepcopy(src)              # ← 关键：操作副本
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)

            for attr in ("ref_image_path", "controlnet_image_path"):
                p = getattr(char, attr, "")
                if p and Path(p).exists():
                    dest = export_dir / Path(p).name
                    shutil.copy2(p, dest)
                    setattr(char, attr, dest.name)  # 副本里改，安全

            json_path = export_dir / f"{char.name}.character.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(char.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"📦 导出角色: {char.name} → {export_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 导出角色失败: {e}")
            return False
    
    def import_character(self, import_path: str) -> Optional[Character]:
        """导入角色卡"""
        try:
            import_dir = Path(import_path)
            if import_dir.is_file() and import_dir.suffix == '.json':
                json_file = import_dir
                import_dir = import_dir.parent
            else:
                json_files = list(import_dir.glob("*.character.json"))
                if not json_files:
                    json_files = list(import_dir.glob("*.json"))
                if not json_files:
                    raise FileNotFoundError("未找到角色 JSON 文件")
                json_file = json_files[0]
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            char = Character.from_dict(data)
            
            # 复制参考图到库目录
            if char.ref_image_path:
                import shutil
                src = import_dir / char.ref_image_path
                if src.exists():
                    dest = self.library_dir / f"{char.uuid}_ref{src.suffix}"
                    shutil.copy2(src, dest)
                    char.ref_image_path = str(dest)
            
            # 复制 ControlNet 图
            if char.controlnet_image_path:
                import shutil
                src = import_dir / char.controlnet_image_path
                if src.exists():
                    dest = self.library_dir / f"{char.uuid}_cn{src.suffix}"
                    shutil.copy2(src, dest)
                    char.controlnet_image_path = str(dest)
            
            # 重新生成 UUID（避免冲突）
            import uuid
            char.uuid = str(uuid.uuid4())
            
            self.save_character(char)
            logger.info(f"📥 导入角色: {char.name}")
            return char
        except Exception as e:
            logger.error(f"❌ 导入角色失败: {e}")
            return None




