_REGISTRY = {}

def register(cls):
    inst = cls()
    for aid in cls.arch_ids:
        _REGISTRY[aid] = inst
    return cls

def get_loader(arch_id):
    loader = _REGISTRY.get(arch_id)
    if loader is None:
        raise ValueError(f"架构 {arch_id} 没有注册加载器")
    return loader

from .single_file import SingleFileLoader   # noqa: E402
from .anima import AnimaLoader              # noqa: E402
register(SingleFileLoader)
register(AnimaLoader)