# presets.py
"""
🎨 场景级预设系统
每个预设包含:
  p           - 正向提示词
  n           - 负向提示词
  params      - 所有可调参数 (步数/提示词强度/采样器/分辨率/CLIP跳过/ADetailer模型...)
"""

# ============================================================
# ADetailer 模型常量 (与 UI 下拉选项对齐)
# ============================================================
AD_FACE_ANIME = "face_yolov8n.pt"        # 动漫人脸模型
AD_FACE_REAL  = "face_yolov8s.pt"        # 真人人脸模型 (精度更高、速度略慢)
AD_HAND       = "hand_yolov8n.pt"        # 手部修正模型

# 通用负面提示词（全局复用，避免重复编写）
NEG_COMMON = (
    "低分辨率, 人体结构崩坏, 手部畸形, 手指缺失, 手指多余, "
    "画面裁切, 画质极差, 低画质, 压缩失真, 水印, 签名, "
    "用户名, 模糊画面, 丑陋五官, 形体扭曲, 面部绘制崩坏, 手部绘制崩坏"
)
# 二次元专用负面词
NEG_ANIME = NEG_COMMON + ", 真人照片, 3D建模, 写实风格, 实景拍摄"
# 真人写实专用负面词
NEG_REAL  = NEG_COMMON + ", 动漫风格, 卡通, 插画, 手绘, 2D画面, 三维动画, 人偶"


PROMPT_PRESETS = {
    # ============================================================
    # 🎯 通用基础类
    # ============================================================
    "默认精美": {
        "p": "杰作, 顶级画质, 极致细节, 8K超清, 电影级光影, 画面锐利",
        "n": NEG_COMMON,
        "params": {
            "steps": 28, "cfg": 7.0, "sampler": "DPM++ 2M Karras",
            "width": 768, "height": 1024, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "极致细节": {
        "p": "(杰作:1.3), (顶级画质:1.3), (极致细节:1.2), 纹理丰富, 高清质感, "
             "精准对焦, 专业布光, 景深效果, 8K超高清画面",
        "n": NEG_COMMON + ", 简约背景, 线稿草图",
        "params": {
            "steps": 40, "cfg": 7.5, "sampler": "DPM++ 2M Karras",
            "width": 832, "height": 1216, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "清新治愈": {
        "p": "柔和色调, 治愈氛围, 明亮光线, 干净画面, 简约构图, 温馨场景, 高清细节",
        "n": NEG_COMMON + ", 暗沉, 压抑, 杂乱背景",
        "params": {
            "steps": 26, "cfg": 6.8, "sampler": "DPM++ 2M Karras",
            "width": 768, "height": 1024, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "高冷氛围感": {
        "p": "冷色调, 高级质感, 清冷氛围, 极简光影, 高级构图, 电影感, 精致五官",
        "n": NEG_COMMON + ", 鲜艳色彩, 可爱风格, 喧闹画面",
        "params": {
            "steps": 30, "cfg": 7.2, "sampler": "DPM++ 2M Karras",
            "width": 832, "height": 1152, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    # ============================================================
    # 🌸 二次元 / 动漫风格
    # ============================================================
    "厚涂油画二次元": {
        "p": "杰作, 顶级画质, 油画风格, 厚重笔触, 丰富肌理, 明暗对比强烈, "
             "高级插画, 绘画质感, 精致面容, 灵动眼眸, 少女, 细腻肌肤",
        "n": NEG_ANIME + ", 平涂色彩, 赛璐璐上色, 纯线稿",
        "params": {
            "steps": 35, "cfg": 7.5, "sampler": "DPM++ 2M Karras",
            "width": 832, "height": 1216, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "清新日系二次元": {
        "p": "杰作, 顶级画质, 日系动漫风格, 赛璐璐上色, 色彩鲜艳, "
             "干净线条, 柔和光线, 少女, 直视镜头, 简约背景",
        "n": NEG_ANIME,
        "params": {
            "steps": 25, "cfg": 7.0, "sampler": "Euler a",
            "width": 768, "height": 1152, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "标准赛璐璐画风": {
        "p": "杰作, 顶级画质, 赛璐璐上色, 动画截图质感, "
             "色彩明快, 锐利线条, 平涂上色, 动画宣传图风格",
        "n": NEG_ANIME + ", 手绘质感, 油画笔触, 写实光影",
        "params": {
            "steps": 24, "cfg": 6.5, "sampler": "Euler a",
            "width": 768, "height": 1152, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "暗黑哥特二次元": {
        "p": "杰作, 顶级画质, 暗黑幻想风, 哥特风格, 戏剧化光影, "
             "浓重阴影, 阴郁氛围, 细节繁复, 动漫风格, 空灵气质, 忧郁意境",
        "n": NEG_ANIME + ", 明亮画面, 欢快氛围, 简约背景",
        "params": {
            "steps": 32, "cfg": 8.0, "sampler": "DPM++ 2M Karras",
            "width": 832, "height": 1216, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "国风二次元": {
        "p": "国风二次元, 古风服饰, 水墨描边, 古典意境, 淡雅配色, "
             "长发飘逸, 古风妆容, 亭台楼阁背景, 唯美画面",
        "n": NEG_ANIME + ", 现代服饰, 街头风格, 浓妆艳抹",
        "params": {
            "steps": 33, "cfg": 7.8, "sampler": "DPM++ 2M Karras",
            "width": 768, "height": 1152, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "Q版萌系": {
        "p": "Q版人物, 萌系画风, 大头小身, 可爱表情, 圆润线条, "
             "粉嫩色彩, 卡通质感, 治愈可爱, 简约背景",
        "n": NEG_ANIME + ", 写实比例, 成熟风格, 暗黑氛围",
        "params": {
            "steps": 22, "cfg": 6.2, "sampler": "Euler a",
            "width": 768, "height": 768, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    # ============================================================
    # 📷 真人写实 / 人像摄影
    # ============================================================
    "电影级写实": {
        "p": "原片质感, 超写实摄影, 肌肤纹理极致细节, 电影级光影, "
             "浅景深, 85mm定焦镜头, 专业人像摄影, 胶片颗粒感, 杰作, 顶级画质",
        "n": NEG_REAL,
        "params": {
            "steps": 30, "cfg": 5.5, "sampler": "DPM++ 2M Karras",
            "width": 896, "height": 1152, "clip_skip": 1,
            "ad_model": AD_FACE_REAL,
        },
    },

    "室内人像写真": {
        "p": "原片质感, 专业人像写真, 柔和自然光, 真实皮肤纹理, "
             "精致眼眸, 虚化背景, 50mm定焦镜头, 大光圈效果, 影棚质感, 超写实",
        "n": NEG_REAL + ", 过曝画面, 塑料假肤, 人偶质感",
        "params": {
            "steps": 30, "cfg": 6.0, "sampler": "DPM++ 2M Karras",
            "width": 832, "height": 1216, "clip_skip": 1,
            "ad_model": AD_FACE_REAL,
        },
    },

    "复古胶片风": {
        "p": "胶卷摄影, 柯达人像卷, 复古调色, 胶片颗粒, 轻微柔焦, "
             "怀旧氛围, 超写实, 暖调柔光, 优质画面",
        "n": NEG_REAL + ", 数码感, 色彩过饱和, 高动态范围",
        "params": {
            "steps": 28, "cfg": 5.0, "sampler": "DPM++ 2M Karras",
            "width": 832, "height": 1152, "clip_skip": 1,
            "ad_model": AD_FACE_REAL,
        },
    },

    "时尚杂志大片": {
        "p": "高端时尚摄影, 杂志封面风格, 影棚布光, 精致妆容, "
             "优雅姿态, 超写实, 极致细节, 精准对焦, 8K超清",
        "n": NEG_REAL + ", 休闲穿搭, 业余拍摄, 粗糙画质",
        "params": {
            "steps": 32, "cfg": 6.5, "sampler": "DPM++ 2M Karras",
            "width": 896, "height": 1152, "clip_skip": 1,
            "ad_model": AD_FACE_REAL,
        },
    },

    "户外街拍": {
        "p": "城市街拍, 自然抓拍, 户外自然光, 生活化氛围, "
             "真实质感, 动态姿态, 街头背景, 纪实摄影风格",
        "n": NEG_REAL + ", 摆拍过重, 浓妆, 影棚背景",
        "params": {
            "steps": 29, "cfg": 5.8, "sampler": "DPM++ 2M Karras",
            "width": 1024, "height": 896, "clip_skip": 1,
            "ad_model": AD_FACE_REAL,
        },
    },

    # ============================================================
    # 🎨 纯艺术画风
    # ============================================================
    "水彩手绘": {
        "p": "水彩画, 柔和马卡龙色系, 纸张纹理, 松散笔触, 梦幻氛围, "
             "艺术感, 精致细节, 色彩晕染, 传统手绘媒介",
        "n": NEG_COMMON + ", 数码绘画, 锐利边缘, 3D渲染",
        "params": {
            "steps": 30, "cfg": 7.0, "sampler": "Euler a",
            "width": 832, "height": 1152, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "游戏概念原画": {
        "p": "概念设计, 数字绘画, 场景原画, 宏大构图, 戏剧光影, "
             "艺术站热门风格, 细节拉满, 电影质感, 杰作",
        "n": NEG_COMMON + ", 画面简单, 扁平画风",
        "params": {
            "steps": 35, "cfg": 8.0, "sampler": "DPM++ 2M Karras",
            "width": 1024, "height": 768, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "中式水墨丹青": {
        "p": "中国水墨画, 传统水墨技法, 单色墨色, 雅致构图, "
             "行云流水笔触, 东方美学, 极简留白, 艺术佳作",
        "n": NEG_COMMON + ", 色彩艳丽, 照片写实, 3D效果, 西式画风",
        "params": {
            "steps": 28, "cfg": 7.5, "sampler": "Euler a",
            "width": 768, "height": 1152, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "赛博朋克艺术": {
        "p": "赛博朋克, 霓虹灯光, 未来都市, 全息投影, 雨水路面反光, "
             "电影光影, 极致细节, 洋红与青蓝撞色, 银翼杀手风格, 8K超清",
        "n": NEG_COMMON + ", 白昼场景, 乡村风景, 复古风格",
        "params": {
            "steps": 32, "cfg": 7.5, "sampler": "DPM++ 2M Karras",
            "width": 1024, "height": 768, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "蒸汽朋克": {
        "p": "蒸汽朋克风格, 黄铜机械, 维多利亚时代, 齿轮零件, 蒸汽管道, "
             "暖棕复古色调, 精密机械细节, 氛围感拉满, 高画质",
        "n": NEG_COMMON + ", 现代科技, 科幻风, 极简干净画面",
        "params": {
            "steps": 32, "cfg": 7.5, "sampler": "DPM++ 2M Karras",
            "width": 896, "height": 1152, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "油画写实人像": {
        "p": "古典油画, 油彩肌理, 厚重颜料, 欧式古典光影, "
             "复古色调, 古典肖像, 笔触细腻, 美术馆画作质感",
        "n": NEG_COMMON + ", 数码感, 线稿, 卡通风格",
        "params": {
            "steps": 34, "cfg": 7.6, "sampler": "DPM++ 2M Karras",
            "width": 832, "height": 1216, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    # ============================================================
    # 🌆 风景 / 场景构图
    # ============================================================
    "梦幻奇幻风景": {
        "p": "绝美自然风光, 魔幻氛围, 黄金时刻光线, 薄雾缭绕, "
             "空灵光影, 极致细节, 电影质感, 奇幻秘境, 8K超清",
        "n": NEG_COMMON + ", 人物, 人像, 肖像",
        "params": {
            "steps": 30, "cfg": 7.0, "sampler": "DPM++ 2M Karras",
            "width": 1216, "height": 832, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "都市霓虹夜景": {
        "p": "城市夜景, 霓虹招牌, 湿润街道, 路面反光, "
             "电影氛围, 光斑散景, 专业摄影, 极致细节, 8K, 超写实",
        "n": NEG_REAL + ", 白天场景, 空旷无人",
        "params": {
            "steps": 30, "cfg": 6.5, "sampler": "DPM++ 2M Karras",
            "width": 1216, "height": 832, "clip_skip": 1,
            "ad_model": AD_FACE_REAL,
        },
    },

    "极简留白构图": {
        "p": "极简构图, 大量留白, 干净画面, 单一主体, "
             "柔和浅色系背景, 专业摄影, 简约雅致, 优质画面",
        "n": NEG_COMMON + ", 元素杂乱, 背景拥挤, 构图复杂",
        "params": {
            "steps": 25, "cfg": 6.0, "sampler": "DPM++ 2M Karras",
            "width": 1024, "height": 1024, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "古风山水意境": {
        "p": "古风山水, 青山绿水, 云雾缭绕, 江南风光, "
             "亭台楼阁, 诗意氛围, 中式园林, 唯美风景",
        "n": NEG_COMMON + ", 现代建筑, 人群, 车辆, 嘈杂场景",
        "params": {
            "steps": 31, "cfg": 7.3, "sampler": "DPM++ 2M Karras",
            "width": 1216, "height": 832, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },

    "森系林间": {
        "p": "原始森林, 林间柔光, 丁达尔光效, 苔藓植被, "
             "自然气息, 静谧氛围, 绿野仙踪, 高清风景",
        "n": NEG_COMMON + ", 人造建筑, 垃圾杂物, 人工改造痕迹",
        "params": {
            "steps": 29, "cfg": 6.9, "sampler": "DPM++ 2M Karras",
            "width": 1024, "height": 896, "clip_skip": 2,
            "ad_model": AD_FACE_ANIME,
        },
    },
}

# ============================================================
# ⭐ 首页推荐预设（下拉框置顶展示）
# ============================================================
RECOMMENDED_PRESETS = [
    "默认精美",
    "厚涂油画二次元",
    "电影级写实",
    "游戏概念原画",
    "中式水墨丹青",
    "都市霓虹夜景",
]
import json
import os
import logging

logger = logging.getLogger(__name__)

from utils.paths import PROJECT_ROOT as _PROJECT_ROOT

USER_PRESETS_FILE = os.path.join(_PROJECT_ROOT, "user_presets.json")

# 内置预设保留一份原始拷贝（防止被用户覆盖）
_BUILTIN_PRESETS = dict(PROMPT_PRESETS)


def load_user_presets() -> dict:
    """从 user_presets.json 加载用户自定义预设"""
    if not os.path.exists(USER_PRESETS_FILE):
        return {}
    try:
        with open(USER_PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"⚠️ {USER_PRESETS_FILE} 不是 dict 结构，忽略")
            return {}
        return data
    except Exception as e:
        logger.error(f"❌ 加载用户预设失败: {e}")
        return {}


def save_user_presets(user_dict: dict) -> bool:
    """保存用户预设到 user_presets.json"""
    try:
        with open(USER_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(user_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 用户预设已保存 ({len(user_dict)} 个)")
        return True
    except Exception as e:
        logger.error(f"❌ 保存用户预设失败: {e}")
        return False


def get_all_presets() -> dict:
    """获取合并后的预设（内置 + 用户）"""
    merged = dict(_BUILTIN_PRESETS)
    user = load_user_presets()
    # 用户预设可以覆盖同名内置
    merged.update(user)
    return merged


def is_builtin_preset(name: str) -> bool:
    """判断是否内置预设"""
    return name in _BUILTIN_PRESETS


# 启动时立即合并一次，让 PROMPT_PRESETS 包含用户预设
PROMPT_PRESETS.update(load_user_presets())