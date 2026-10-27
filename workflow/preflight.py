# workflow/preflight.py
"""运行前预检（WorkflowPreflight）。

为什么需要它
------------
工作流最容易让人恼火的地方不是"跑得慢"，而是**跑到一半才报错**：
模型路径写错 → 加载 20 秒后炸；LoRA 是 SDXL 的但底模是 SD1.5 →
采样时才抛 shape mismatch；输出目录不可写 → 出完图才说保存失败。

所以运行前把所有能静态/廉价发现的问题一次报完，并按严重程度分三档：

    错误(error)   无法运行，必须先处理
    警告(warning) 能跑，但会影响效果或速度
    提示(hint)    建议操作（不改也能跑）

对外主入口：`run_preflight(wf, host=None, ai=None) -> PreflightResult`

设计约束
--------
· **不做重活**：不加载模型、不推理。只查文件是否存在、架构是否匹配、
  依赖是否装上、目录是否可写。整个预检应在百毫秒级。
· **任何单项检查失败都不该让预检本身崩**：各自 try/except，降级成警告。
· 结论给 UI 直接用（`to_text()`），也给程序用（结构化字段）。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from workflow import adapters, graph as graph_mod, registry
from workflow.memory_policy import DeviceCapabilities, resolve as resolve_policy

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"
HINT = "hint"

LEVEL_LABELS = {ERROR: "❌ 错误", WARNING: "⚠️ 警告", HINT: "💡 提示"}


@dataclass
class PreflightIssue:
    level: str
    code: str
    message: str
    node_id: str = ""
    fix: str = ""

    def to_dict(self) -> dict:
        return {"level": self.level, "code": self.code,
                "message": self.message, "node_id": self.node_id,
                "fix": self.fix}


@dataclass
class PreflightResult:
    ok: bool = True
    errors: List[PreflightIssue] = field(default_factory=list)
    warnings: List[PreflightIssue] = field(default_factory=list)
    hints: List[PreflightIssue] = field(default_factory=list)
    required_models: List[dict] = field(default_factory=list)
    estimated_vram_mb: int = 0
    estimated_time: float = 0.0
    checks_run: int = 0
    policy: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)

    # ---- 便捷构造 ----
    def add(self, level: str, code: str, message: str,
            node_id: str = "", fix: str = ""):
        issue = PreflightIssue(level, code, message, node_id, fix)
        if level == ERROR:
            self.errors.append(issue)
            self.ok = False
        elif level == WARNING:
            self.warnings.append(issue)
        else:
            self.hints.append(issue)
        return issue

    # ---- 输出 ----
    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": [i.to_dict() for i in self.errors],
            "warnings": [i.to_dict() for i in self.warnings],
            "hints": [i.to_dict() for i in self.hints],
            "required_models": self.required_models,
            "estimated_vram_mb": self.estimated_vram_mb,
            "estimated_time": self.estimated_time,
            "checks_run": self.checks_run,
            "policy": self.policy,
        }

    def to_text(self, max_each: int = 12, *, translate=None) -> str:
        """生成预检报告；后台默认保留中文，界面可注入翻译函数。"""
        tr = translate or (lambda text: text)
        lines: List[str] = []
        if self.ok:
            lines.append(tr("✅ 预检通过，可以运行"))
        else:
            lines.append(tr("❌ 预检未通过（{count} 个错误）").format(
                count=len(self.errors)))

        for level, items in ((ERROR, self.errors),
                             (WARNING, self.warnings),
                             (HINT, self.hints)):
            if not items:
                continue
            lines.append("")
            lines.append(tr("{label}（{count}）").format(
                label=tr(LEVEL_LABELS[level]), count=len(items)))
            for it in items[:max_each]:
                where = f"[{it.node_id}] " if it.node_id else ""
                lines.append(f"  · {where}{tr(it.message)}")
                if it.fix:
                    lines.append(f"      → {tr(it.fix)}")
            if len(items) > max_each:
                lines.append(tr("  …还有 {count} 条").format(
                    count=len(items) - max_each))

        lines.append("")
        line = tr("预计显存 {value} MB").format(value=self.estimated_vram_mb)
        if self.estimated_time:
            line += tr(" · 预计耗时约 {seconds}s").format(
                seconds=f"{self.estimated_time:.0f}")
        if self.policy:
            line += f" · {self.policy}"
        lines.append(line)
        if self.required_models:
            miss = [m for m in self.required_models if not m.get("present")]
            lines.append(tr("资源：需要 {required} 项，缺 {missing} 项").format(
                required=len(self.required_models), missing=len(miss)))
        return "\n".join(lines)


# ======================================================================
#  各项检查
# ======================================================================
def _find_checkpoint(name: str) -> Optional[str]:
    """按 load_model 的规则找底模文件的绝对路径。"""
    if not name:
        return None
    name = name.strip()
    if os.path.isabs(name) and os.path.exists(name):
        return name
    from utils import paths
    subs = ["sd15", "sdxl", "sd3", "pony", "flux", "anima"]
    cands = [os.path.join(paths.MODEL_DIR, name)] + [
        os.path.join(paths.MODEL_DIR, s, name) for s in subs]
    exts = ["", ".safetensors", ".ckpt", ".pt"]
    for c in cands:
        for e in exts:
            p = c + e
            if os.path.isfile(p):
                return p
    # 兜底：递归找同名文件
    try:
        for root, _d, files in os.walk(paths.MODEL_DIR):
            if name in files:
                return os.path.join(root, name)
    except OSError:
        pass
    return None


def _find_lora(name: str) -> Optional[str]:
    if not name:
        return None
    name = name.strip()
    if name in ("无", "none", "None", ""):
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    from utils import paths
    exts = ["", ".safetensors", ".ckpt", ".pt"]
    try:
        for root, _d, files in os.walk(paths.LORA_DIR):
            for f in files:
                if f == name or os.path.splitext(f)[0] == os.path.splitext(name)[0]:
                    return os.path.join(root, f)
    except OSError:
        pass
    for root, _d, files in os.walk(paths.LORA_DIR):
        for f in files:
            for e in exts:
                if f == name + e:
                    return os.path.join(root, f)
    return None


def _detect_arch(path: str) -> str:
    try:
        from core import arch as arch_mod
        return arch_mod.detect(path).arch_id
    except Exception as e:
        logger.debug(f"架构检测失败 {path}: {e}")
        return "unknown"


def _check_structure(wf, res: PreflightResult):
    """工作流结构 + 节点输入完整性（复用 graph.validate）。"""
    try:
        issues = graph_mod.validate(wf)
    except Exception as e:
        res.add(ERROR, "structure", f"工作流校验异常: {e}")
        return
    for i in issues:
        if i.level == "error":
            res.add(ERROR, "structure", i.message, i.node_id)
        else:
            res.add(WARNING, "structure", i.message, i.node_id)
    if not wf.nodes:
        res.add(ERROR, "structure", "工作流是空的，没有可执行的节点")


def _check_models(wf, res: PreflightResult, ai) -> List[dict]:
    """底模：存在性 + 架构 + 与节点能力是否匹配。"""
    found: List[dict] = []
    loaders = [n for n in wf.nodes if n.type == "LoadCheckpoint"]
    if not loaders:
        res.add(WARNING, "no_checkpoint",
                "工作流里没有 LoadCheckpoint 节点",
                fix="如果你打算用当前已加载的模型，可以忽略这条")
        return found

    archs = set()
    for n in loaders:
        name = (n.params.get("path") or n.inputs.get("path") or "")
        if not isinstance(name, str):
            name = str(name)
        name = name.strip()
        if not name:
            res.add(ERROR, "model_path_empty",
                    "没有指定模型文件", n.id,
                    fix="在 LoadCheckpoint 的「模型」里从下拉框选一个本地底模")
            continue
        path = _find_checkpoint(name)
        if not path:
            res.add(ERROR, "model_missing",
                    f"找不到模型文件「{name}」", n.id,
                    fix="放到 models/puton/ 让程序自动归位，或从下拉框重新选择")
            found.append({"type": "checkpoint", "name": name, "present": False})
            continue
        arch = _detect_arch(path)
        found.append({"type": "checkpoint", "name": os.path.basename(path),
                      "present": True, "arch": arch, "path": path})
        if arch == "unknown":
            res.add(WARNING, "model_arch_unknown",
                    f"无法识别「{os.path.basename(path)}」的架构", n.id,
                    fix="仍可尝试运行；若失败请确认它是不是完整底模")
        else:
            archs.add(arch)
            ad = adapters.get_adapter(arch)
            if not ad.caps.txt2img:
                res.add(ERROR, "arch_unsupported",
                        f"架构 {arch} 当前不支持生成", n.id)
        # clip_skip 适用性
        cs = int(n.params.get("clip_skip") or 0)
        if cs > 0 and arch in ("sdxl", "pony", "sd3", "flux", "anima"):
            res.add(HINT, "clip_skip_ignored",
                    f"{arch} 上 CLIP Skip 不生效（管线固定使用倒数第二层）", n.id,
                    fix="可以忽略，或把 CLIP Skip 设为 0")

    if len(archs) > 1:
        res.add(WARNING, "mixed_arch",
                f"工作流里同时用了多种架构的底模: {', '.join(sorted(archs))}",
                fix="不同架构的节点通常共用同一个 UNet，混用可能互相覆盖")

    # 节点能力 vs 架构
    if archs:
        arch = sorted(archs)[0]
        ad = adapters.get_adapter(arch)
        # 用统一的**能力名**（与 ModelCapabilities.supports 同一套命名）
        need = {"HiresFix": "hires_fix", "ADetailer": "adetailer",
                "ControlNetApply": "controlnet", "IPAdapterApply": "ip_adapter",
                "LoadLoRA": "lora", "Inpaint": "inpaint",
                "VAEEncode": "img2img", "Sampler": "txt2img"}
        for n in wf.nodes:
            feat = need.get(n.type)
            if feat and not ad.supports(feat):
                res.add(ERROR, "capability",
                        f"{arch} 架构不支持 {n.type}（{feat}）", n.id,
                        fix=f"删掉这个节点，或改用支持该功能的底模")
    return found


def _check_loras(wf, res: PreflightResult, model_arch: str):
    """LoRA：存在性 + 是否适用于当前底模（最容易踩的坑）。"""
    names = []
    for n in wf.nodes:
        if n.type != "LoadLoRA":
            continue
        for i in (1, 2, 3):
            nm = (n.params.get(f"lora_{i}") or "").strip()
            if not nm or nm in ("无", "none", "None"):
                continue
            names.append((n.id, nm))
    if not names:
        return

    exp_sub = {"sd15": "sd1.5", "sd21": "sd1.5", "sdxl": "sdxl", "pony": "sdxl",
               "flux": "flux", "sd3": "sd3", "anima": "anima"}.get(model_arch, "")

    for node_id, nm in names:
        path = _find_lora(nm)
        if not path:
            res.add(ERROR, "lora_missing",
                    f"找不到 LoRA「{nm}」", node_id,
                    fix="放到 loras/ 对应架构子目录，或用下拉框重新选择")
            continue
        larch = _detect_arch(path)
        # LoRA 的 arch_id 形如 lora_sd15 / lora_sdxl / lora_dit
        l_sub = {"lora_sd15": "sd1.5", "lora_sd21": "sd1.5",
                 "lora_sdxl": "sdxl"}.get(larch)
        if l_sub and exp_sub and l_sub != exp_sub:
            res.add(ERROR, "lora_arch_mismatch",
                    f"LoRA「{nm}」是 {l_sub} 用的，但主模型是 {model_arch}"
                    f"（{exp_sub}）", node_id,
                    fix="换一个对应架构的 LoRA，或换底模")
        elif larch == "lora_dit":
            res.add(WARNING, "lora_dit",
                    f"LoRA「{nm}」看起来是 DiT 架构的，与 {model_arch} 不匹配",
                    node_id)
        elif larch in ("lora", "unknown"):
            res.add(HINT, "lora_arch_unknown",
                    f"无法确定 LoRA「{nm}」适用的架构", node_id,
                    fix="若出图异常请检查它是否与底模同代")


def _check_vae(wf, res: PreflightResult, model_arch: str):
    """VAE：存在性 + 兼容性（按文件名/体积粗判）。"""
    for n in wf.nodes:
        if n.type != "LoadVAE":
            continue
        name = (n.params.get("path") or "").strip()
        if not name:
            res.add(ERROR, "vae_path_empty", "没有指定 VAE 文件", n.id,
                    fix="从下拉框选一个本地 VAE，或删掉这个节点")
            continue
        from utils import paths
        found = None
        for root in (paths.MODEL_DIR, paths.WEIGHTS_DIR):
            for r, _d, files in os.walk(root):
                for f in files:
                    if f == name or os.path.splitext(f)[0] == os.path.splitext(name)[0]:
                        found = os.path.join(r, f)
                        break
                if found:
                    break
            if found:
                break
        if not found:
            res.add(ERROR, "vae_missing", f"找不到 VAE「{name}」", n.id)
            continue
        low = os.path.basename(found).lower()
        size_mb = os.path.getsize(found) / 1024 ** 2
        looks_sdxl = ("xl" in low) or size_mb > 250
        if model_arch in ("sd15", "sd21") and looks_sdxl:
            res.add(ERROR, "vae_mismatch",
                    f"VAE「{name}」看起来是 SDXL 用的（{size_mb:.0f}MB），"
                    f"与 SD1.5 底模不兼容", n.id,
                    fix="换成 SD1.5 的 VAE，或删掉这个节点")
        elif model_arch in ("sdxl", "pony") and not looks_sdxl:
            res.add(WARNING, "vae_maybe_sd15",
                    f"VAE「{name}」体积只有 {size_mb:.0f}MB，可能不是 SDXL 的",
                    n.id)


def _check_controlnet(wf, res: PreflightResult, model_arch: str):
    """ControlNet：类型是否被支持 + 本地是否已安装。"""
    applied = [n for n in wf.nodes if n.type == "ControlNetApply"]
    if not applied:
        return
    is_sdxl = model_arch in ("sdxl", "pony")
    supported = ("openpose", "canny", "depth")

    # 从上游 ControlNetPreprocess 推类型；没有就用默认
    for n in applied:
        kind = "openpose"
        for lk in wf.links:
            if lk[2] == n.id and lk[3] == "control":
                src = wf.node(lk[0])
                if src is not None and src.type == "ControlNetPreprocess":
                    kind = (src.params.get("kind") or "openpose")
        if kind not in supported:
            res.add(ERROR, "cn_kind_unsupported",
                    f"ControlNet 类型「{kind}」目前无法加载"
                    f"（只支持 {'/'.join(supported)}）", n.id,
                    fix="改用 openpose / canny / depth；"
                        "其它类型目前只能单独做预处理出图")
            continue
        try:
            from core.model_manager import ModelManager
            src_path, how = ModelManager.describe_local_controlnet(
                kind, is_sdxl=is_sdxl)
        except Exception as e:
            src_path, how = None, f"探测失败: {e}"
        if not src_path:
            res.add(ERROR, "cn_missing",
                    f"本地没有可用的 ControlNet 权重（{kind}"
                    f"{'，SDXL' if is_sdxl else '，SD1.5'}）", n.id,
                    fix="在「扩展市场 → 控制」里下载，或把权重放进 controlnets/")
        else:
            res.add(HINT, "cn_ok", f"ControlNet {kind} 已就绪（{how}）", n.id)


def _check_assets(wf, res: PreflightResult):
    """节点声明的可下载资产 / 外部工具是否齐备。"""
    try:
        from workflow import resource_manager as RM
        reqs = RM.collect(wf)
    except Exception as e:
        res.add(WARNING, "assets_probe_failed", f"资源探测失败: {e}")
        return
    for r in reqs:
        if r.kind == "asset" and not r.present:
            res.add(ERROR, "asset_missing",
                    f"缺少组件：{r.label or r.key}"
                    + (f"（约 {r.size_mb / 1024:.1f}GB）" if r.size_mb else ""),
                    (r.nodes or [""])[0],
                    fix="在「工具 → 🛒 扩展市场」里下载（程序不会自动联网下载）")
        elif r.kind == "tool" and not r.present:
            res.add(ERROR, "tool_missing",
                    f"缺少外部工具：{r.key}", (r.nodes or [""])[0],
                    fix="在扩展市场安装，或自行准备")


def _check_vram(wf, res: PreflightResult, policy, cap):
    """显存：按节点声明估算，并说明会自动采取什么策略。"""
    try:
        specs = [(n.id, registry.get(n.type)) for n in wf.nodes
                 if registry.exists(n.type)]
        from workflow.memory_policy import plan_for_nodes
        plan = plan_for_nodes(policy, specs, cap=cap)
    except Exception as e:
        res.add(WARNING, "vram_probe_failed", f"显存估算失败: {e}")
        return
    res.plan = plan
    res.estimated_vram_mb = int(plan.get("total_peak_mb", 0))
    res.policy = policy.describe()

    for w in plan.get("warnings", []):
        res.add(WARNING, "vram", w)
    if cap.has_cuda and res.estimated_vram_mb > cap.total_gb * 1024 * 0.9:
        res.add(WARNING, "vram_tight",
                f"节点声明显存合计约 {res.estimated_vram_mb} MB，"
                f"占设备 {cap.total_gb * 1024:.0f} MB 的比例偏高",
                fix="程序会自动启用分块/CPU offload，速度会下降")
    if policy.cpu_offload:
        res.add(WARNING, "vram_offload",
                f"当前显存 {cap.total_gb:.1f}GB，已自动启用 CPU Offload",
                fix="这是正常的省显存模式；显存更大时可切「高性能」预设提速")
    n_cn = len([n for n in wf.nodes if n.type == "ControlNetApply"])
    if n_cn >= 2:
        res.add(WARNING, "vram_many_cn",
                f"当前工作流包含 {n_cn} 个 ControlNet，预计速度较慢")


def _check_output_dir(wf, res: PreflightResult, host):
    """输出目录：能不能写 + **有没有被静默回退**。

    ⚠️ 只检查 `host_output_dir()` 是不够的：`resolve_output_dir()` 在配置目录
    不可用时会**静默回退**到 photo/，于是"Z:/不存在的盘"这种配置永远测不出
    问题 —— 用户以为图存进了自己选的目录，其实全落在 photo/。
    所以这里额外比对「配置值 vs 实际解析值」，把回退暴露出来。
    """
    from utils import paths
    cfg = getattr(host, "config", None) if host is not None else None
    raw = (getattr(cfg, "output_dir", "") or "").strip()

    try:
        from utils.paths import resolve_output_dir
        resolved = str(resolve_output_dir(cfg))
    except Exception as e:
        res.add(ERROR, "output_dir", f"输出目录解析失败: {e}",
                fix="在「输出设置 → 输出目录」换一个可写目录")
        return

    # 历史占位值 "outputs/" 不算"用户自定义目录" —— paths.resolve_output_dir
    # 有意把它当作"未设置"处理，这里不能因此误报回退。
    legacy = raw.replace("\\", "/").rstrip("/").lower() in ("", "outputs")
    if raw and not legacy:
        want = raw
        if not os.path.isabs(want):
            want = os.path.join(paths.PROJECT_ROOT, want)
        same = (os.path.normcase(os.path.abspath(want))
                == os.path.normcase(os.path.abspath(resolved)))
        if not same:
            res.add(WARNING, "output_dir_fallback",
                    f"配置的输出目录「{raw}」不可用，实际会写到 {resolved}",
                    fix="换一个可写目录（检查盘符是否存在、权限是否足够）")

    # 实际写一个临时文件验证（最可靠的判据）
    try:
        os.makedirs(resolved, exist_ok=True)
        probe = os.path.join(resolved, f".preflight_{os.getpid()}.tmp")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except Exception as e:
        res.add(ERROR, "output_dir",
                f"输出目录不可写（{resolved}）: {e}",
                fix="在「输出设置 → 输出目录」换一个可写目录")


def _check_resource_refs(wf, res: PreflightResult):
    """按 `resource_id` 检查工作流依赖的模型资源（跨机器可迁移的那一套）。

    这一步和 `_check_models` 的区别：
      · `_check_models` 是"按路径找文件"的老逻辑，兜底用；
      · 这里走 `resource_store`，按**稳定 resource_id** 解析，
        能区分「缺失 / 位置变了 / 架构不符 / 校验失败」，
        并把可下载的辅助组件挑出来给 UI 渲染资源卡片。
    `resource_store` 不可用时静默跳过，不影响其它检查。
    """
    try:
        from workflow import resource_store as RS
    except Exception as e:
        logger.debug(f"resource_store 不可用，跳过资源引用检查: {e}")
        return
    try:
        refs = RS.collect_refs(wf)
    except Exception as e:
        res.add(WARNING, "resources_probe_failed", f"资源依赖检查失败: {e}")
        return
    if not refs:
        return
    try:
        rep = RS.missing_report(refs)
    except Exception as e:
        res.add(WARNING, "resources_report_failed", f"资源报告生成失败: {e}")
        return

    res.required_models.extend(
        [dict(r.get("ref") or {}, status=r.get("status"))
         for r in (rep.get("missing", []) + rep.get("mismatched", []))])

    # ⚠️ 底模/LoRA/VAE 缺失报**错误**（跑不了）；辅助组件缺失也报错误，
    #    但会附带"去扩展市场下载"的建议，UI 直接用 downloadable 渲染卡片。
    for r in rep.get("missing", []):
        ref = r.get("ref") or {}
        rtype = (ref.get("model_type") or "?").lower()
        name = ref.get("filename") or ref.get("resource_id") or "?"
        is_aux = rtype in ("controlnet", "adetailer", "ip_adapter",
                           "upscaler", "engine")
        fix = ("在「🗂 资源依赖」里点下载（扩展市场）" if is_aux
               else f"自行下载后放到 {TYPE_HINT.get(rtype, 'models/')}，"
                    "再点「🔄 重新检测」")
        res.add(ERROR, "resource_missing",
                f"缺少{_type_cn(rtype)}「{name}」（工作流按资源 ID 依赖它）",
                (r.get("node_id") or ""), fix)

    for r in rep.get("mismatched", []):
        ref = r.get("ref") or {}
        name = ref.get("filename") or ref.get("resource_id") or "?"
        want = ref.get("architecture") or "?"
        res.add(ERROR, "resource_mismatch",
                f"资源「{name}」与工作流记录的架构不符（工作流要求 {want}）"
                f"：{r.get('detail', '')}",
                (r.get("node_id") or ""),
                fix="换成对应架构的文件，或在工作流里改回正确的资源")

    moved = [r for r in rep.get("present", [])
             if r.get("status") == "found_elsewhere"]
    if moved:
        names = "、".join((r.get("ref") or {}).get("filename", "?")
                          for r in moved[:3])
        res.add(WARNING, "resource_moved",
                f"有 {len(moved)} 项资源不在原位置，已按文件名找到：{names}",
                fix="能跑，但建议重新保存工作流以更新资源位置")


TYPE_HINT = {
    "checkpoint": "models/（可放 models/puton/ 自动归位）",
    "lora": "loras/（按架构分子目录）",
    "vae": "models/ 或 weights/",
    "controlnet": "controlnets/",
    "upscaler": "upscalers/",
    "engine": "models/flux/",
}


def _type_cn(t: str) -> str:
    return {"checkpoint": "底模", "lora": "LoRA", "vae": "VAE",
            "controlnet": "ControlNet", "upscaler": "放大/插帧模型",
            "engine": "推理引擎", "adetailer": "ADetailer",
            "ip_adapter": "IP-Adapter"}.get(t, t or "模型")


def _check_dependencies(wf, res: PreflightResult):
    missing = []
    for mod, label in (("torch", "PyTorch"), ("diffusers", "diffusers"),
                       ("PIL", "Pillow")):
        try:
            __import__(mod)
        except Exception:
            missing.append(label)
    if missing:
        res.add(ERROR, "deps", f"缺少必需依赖: {', '.join(missing)}",
                fix="pip install -r requirements.txt")


def _estimate_time(wf, policy) -> float:
    """很粗的耗时估计（秒）。只用于给用户一个量级感。"""
    steps = 28
    n_samples = 0
    for n in wf.nodes:
        if n.type == "Sampler":
            n_samples += 1
            try:
                steps = int(n.params.get("steps", steps) or steps)
            except Exception:
                pass
    if not n_samples:
        return 0.0
    # 经验值：512x768 / 28 步 / SD1.5 全 GPU ≈ 0.35s/步
    per_step = 0.35
    if policy.cpu_offload:
        per_step *= 1.6
    if policy.sequential_offload:
        per_step *= 8.0
    # 重活节点按"等效步数"折算
    extra = 0
    for n in wf.nodes:
        if n.type == "HiresFix":
            extra += steps * 0.8
        elif n.type == "ADetailer":
            extra += 24 * 0.5
        elif n.type == "ControlNetApply":
            extra += steps * 0.25
        elif n.type == "IPAdapterApply":
            extra += steps * 0.15
    return (steps + extra) * per_step * max(1, n_samples)


# ======================================================================
def run_preflight(wf, host=None, ai=None, *, preset: str = "auto",
                  cap: Optional[DeviceCapabilities] = None) -> PreflightResult:
    """执行全部预检。任何时候都不抛异常。"""
    res = PreflightResult()
    if wf is None:
        res.add(ERROR, "no_workflow", "没有可检查的工作流（wf 为空）")
        return res
    ai = ai if ai is not None else getattr(host, "ai", None)
    try:
        registry.load_builtin_nodes()
    except Exception:
        pass

    arch = ""
    try:
        arch = (getattr(ai, "current_arch_id", "") or "") if ai else ""
    except Exception:
        arch = ""

    checks = [
        ("structure", lambda: _check_structure(wf, res)),
        ("output", lambda: _check_output_dir(wf, res, host)),
        ("deps", lambda: _check_dependencies(wf, res)),
        ("assets", lambda: _check_assets(wf, res)),
        ("resources", lambda: _check_resource_refs(wf, res)),
    ]
    for name, fn in checks:
        try:
            fn()
            res.checks_run += 1
        except Exception as e:
            logger.warning(f"[preflight] {name} 检查异常: {e}")
            res.add(WARNING, f"{name}_failed", f"{name} 检查未能完成: {e}")

    # 需要底模信息的两组检查
    try:
        models = _check_models(wf, res, ai)
        res.required_models.extend(models)
        res.checks_run += 1
    except Exception as e:
        logger.warning(f"[preflight] 模型检查异常: {e}")
        models = []
        res.add(WARNING, "models_failed", f"模型检查未能完成: {e}")

    model_arch = ""
    for m in models:
        if m.get("present") and m.get("arch") and m["arch"] != "unknown":
            model_arch = m["arch"]
            break
    if not model_arch:
        model_arch = arch

    for name, fn in (
        ("loras", lambda: _check_loras(wf, res, model_arch)),
        ("vae", lambda: _check_vae(wf, res, model_arch)),
        ("controlnet", lambda: _check_controlnet(wf, res, model_arch)),
    ):
        try:
            fn()
            res.checks_run += 1
        except Exception as e:
            logger.warning(f"[preflight] {name} 检查异常: {e}")
            res.add(WARNING, f"{name}_failed", f"{name} 检查未能完成: {e}")

    # 显存策略
    try:
        cap = cap or DeviceCapabilities.detect()
        p = (wf.metadata or {}).get("memory_preset") or preset
        policy = resolve_policy(p, cap, arch_id=model_arch)
        _check_vram(wf, res, policy, cap)
        res.estimated_time = _estimate_time(wf, policy)
        res.checks_run += 1
    except Exception as e:
        logger.warning(f"[preflight] 显存检查异常: {e}")
        res.add(WARNING, "vram_failed", f"显存检查未能完成: {e}")

    # 没接 SaveImage 的提示
    try:
        if wf.nodes and not any(n.type == "SaveImage" for n in wf.nodes):
            res.add(HINT, "no_save",
                    "工作流里没有 SaveImage 节点，结果不会落盘",
                    fix="加一个 SaveImage 节点")
    except Exception:
        pass

    res.ok = not res.errors
    return res
