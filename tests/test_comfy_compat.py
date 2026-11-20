# tests/test_comfy_compat.py
# ============================================================
#  ComfyUI 工作流兼容层的回归
# ------------------------------------------------------------
#  锁四件事：
#    1. 两种格式（API / UI）都能识别并导入；
#    2. 折叠节点（EmptyLatentImage / CLIPSetLastLayer）的值真的进了
#       下游节点的参数，而不是被悄悄丢掉；
#    3. 认不出来的节点必须出现在报告里 —— 不许静默少节点；
#    4. 采样器 / sigma 名字在 ComfyUI 与本项目之间双向都对得上。
# ============================================================
import _bootstrap  # noqa: F401  （路径 + 配置隔离）

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_bootstrap.isolate_config_globally()

from workflow import comfy_compat as cc  # noqa: E402
from workflow.registry import load_builtin_nodes  # noqa: E402

load_builtin_nodes()

FAILS = []


def check(name):
    def deco(fn):
        try:
            fn()
        except Exception as e:
            import traceback
            FAILS.append((name, f"{e}"))
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
        else:
            print(f"[PASS] {name}")
        return fn
    return deco


def _api_graph() -> dict:
    """一个最小可用的 ComfyUI API 图：底模 → 正负提示 → 采样 → 解码 → 存图。"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sd15/meinamix_v12Final.safetensors"}},
        "2": {"class_type": "CLIPSetLastLayer",
              "inputs": {"clip": ["1", 0], "stop_at_clip_layer": -2}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": "a cat on a windowsill"}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": "blurry, low quality"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["3", 0],
                         "negative": ["4", 0], "latent_image": ["5", 0],
                         "seed": 4242, "steps": 24, "cfg": 6.5,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras",
                         "denoise": 1.0}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["1", 1]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "comfy_demo"}},
    }


@check("test_detect_format")
def test_detect_format():
    assert cc.detect_format(_api_graph()) == cc.FORMAT_API
    assert cc.detect_format({"nodes": [], "links": []}) == cc.FORMAT_UI
    try:
        cc.detect_format({"foo": 1})
    except ValueError as e:
        assert "无法识别" in str(e)
    else:
        raise AssertionError("非法格式应抛 ValueError")


@check("test_import_api_graph")
def test_import_api_graph():
    wf, rep = cc.import_prompt(_api_graph())
    types = sorted(n.type for n in wf.nodes)
    assert types == ["LoadCheckpoint", "NegativeTextEncode", "Sampler",
                     "SaveImage", "TextEncode", "VAEDecode"], types
    assert not rep.unsupported, rep.unsupported
    # 连线：Sampler.model ← LoadCheckpoint.model
    sampler = wf.node("c6")
    assert sampler.inputs["model"] == ["c1", "model"], sampler.inputs
    assert sampler.inputs["positive"] == ["c3", "conditioning"]
    assert sampler.inputs["negative"] == ["c4", "conditioning"]
    # VAEDecode 的 samples→latent、vae→model 别名
    dec = wf.node("c7")
    assert dec.inputs["latent"] == ["c6", "latent"], dec.inputs
    assert dec.inputs["model"] == ["c1", "vae"], dec.inputs
    # SaveImage 的 images→image
    assert wf.node("c8").inputs["image"] == ["c7", "image"]


@check("test_folded_empty_latent_size")
def test_folded_empty_latent_size():
    wf, rep = cc.import_prompt(_api_graph())
    sampler = wf.node("c6")
    assert sampler.params["width"] == 832, sampler.params
    assert sampler.params["height"] == 1216
    assert any("EmptyLatentImage" in f for f in rep.folded), rep.folded


@check("test_folded_clip_skip")
def test_folded_clip_skip():
    wf, rep = cc.import_prompt(_api_graph())
    ckpt = wf.node("c1")
    assert ckpt.params["clip_skip"] == 2, ckpt.params
    assert any("clip_skip" in f for f in rep.folded), rep.folded


@check("test_text_roles_follow_wiring")
def test_text_roles_follow_wiring():
    wf, _ = cc.import_prompt(_api_graph())
    assert wf.node("c3").type == "TextEncode"
    assert wf.node("c4").type == "NegativeTextEncode"
    assert wf.node("c3").params["text"] == "a cat on a windowsill"


@check("test_sampler_and_sigma_names")
def test_sampler_and_sigma_names():
    wf, _ = cc.import_prompt(_api_graph())
    params = wf.node("c6").params
    assert params["sampler"] == "DPM++ 2M", params
    assert params["scheduler"] == "Karras", params
    assert params["seed"] == 4242 and params["steps"] == 24
    assert abs(params["strength"] - 1.0) < 1e-9


@check("test_lora_loader_maps_to_load_lora")
def test_lora_loader_maps_to_load_lora():
    graph = _api_graph()
    graph["9"] = {"class_type": "LoraLoader",
                  "inputs": {"model": ["1", 0], "clip": ["1", 0],
                             "lora_name": "mychar.safetensors",
                             "strength_model": 0.75, "strength_clip": 0.6}}
    graph["6"]["inputs"]["model"] = ["9", 0]
    wf, rep = cc.import_prompt(graph)
    lora = wf.node("c9")
    assert lora is not None and lora.type == "LoadLoRA"
    assert lora.params["lora_1"] == "mychar.safetensors"
    assert abs(lora.params["weight_1"] - 0.75) < 1e-9
    assert wf.node("c6").inputs["model"] == ["c9", "model"]


@check("test_unsupported_nodes_are_reported")
def test_unsupported_nodes_are_reported():
    graph = _api_graph()
    graph["10"] = {"class_type": "SomeWeirdNode",
                   "inputs": {"anything": 1}}
    wf, rep = cc.import_prompt(graph)
    assert ("10", "SomeWeirdNode") in rep.unsupported, rep.unsupported
    assert "SomeWeirdNode" in rep.summary_text()
    assert len(wf.nodes) == 6      # 不认识的节点不进图，但被报告了


@check("test_ui_format_imports_like_api")
def test_ui_format_imports_like_api():
    ui = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple",
             "widgets_values": ["sd15/x.safetensors"]},
            {"id": 2, "type": "CLIPTextEncode",
             "widgets_values": ["a red apple"],
             "inputs": [{"name": "clip", "link": 11}]},
            {"id": 3, "type": "CLIPTextEncode",
             "widgets_values": ["ugly"],
             "inputs": [{"name": "clip", "link": 12}]},
            {"id": 4, "type": "EmptyLatentImage",
             "widgets_values": [512, 768, 1]},
            {"id": 5, "type": "KSampler",
             "widgets_values": [7, "randomize", 20, 7.0, "euler", "normal", 1.0],
             "inputs": [{"name": "model", "link": 13},
                        {"name": "positive", "link": 14},
                        {"name": "negative", "link": 15},
                        {"name": "latent_image", "link": 16}]},
            {"id": 6, "type": "VAEDecode",
             "inputs": [{"name": "samples", "link": 17},
                        {"name": "vae", "link": 18}]},
            {"id": 7, "type": "SaveImage",
             "widgets_values": ["ui_demo"],
             "inputs": [{"name": "images", "link": 19}]},
        ],
        "links": [
            [11, 1, 0, 2, 0, "CLIP"],
            [12, 1, 0, 3, 0, "CLIP"],
            [13, 1, 0, 5, 0, "MODEL"],
            [14, 2, 0, 5, 1, "CONDITIONING"],
            [15, 3, 0, 5, 2, "CONDITIONING"],
            [16, 4, 0, 5, 3, "LATENT"],
            [17, 5, 0, 6, 0, "LATENT"],
            [18, 1, 1, 6, 1, "VAE"],
            [19, 6, 0, 7, 0, "IMAGE"],
        ],
    }
    wf, rep = cc.import_prompt(ui)
    assert rep.kind == cc.FORMAT_UI
    types = sorted(n.type for n in wf.nodes)
    assert types == ["LoadCheckpoint", "NegativeTextEncode", "Sampler",
                     "SaveImage", "TextEncode", "VAEDecode"], types
    sampler = wf.node("c5")
    assert sampler.params["width"] == 512 and sampler.params["height"] == 768
    assert sampler.params["sampler"] == "Euler", sampler.params
    assert sampler.params["scheduler"] == "默认", sampler.params
    assert wf.node("c6").inputs["model"] == ["c1", "vae"]


@check("test_reroute_is_transparent")
def test_reroute_is_transparent():
    graph = _api_graph()
    graph["11"] = {"class_type": "Reroute", "inputs": {"": ["1", 2]}}
    graph["8"]["inputs"]["images"] = ["11", 0]
    graph["11"]["inputs"] = {"": ["7", 0]}
    wf, rep = cc.import_prompt(graph)
    assert wf.node("c8").inputs["image"] == ["c7", "image"], wf.node("c8").inputs


@check("test_export_round_trip")
def test_export_round_trip():
    wf, _ = cc.import_prompt(_api_graph())
    data, rep = cc.export_prompt(wf)
    assert data, rep.summary_text()
    assert not rep.unsupported, rep.unsupported
    wf2, rep2 = cc.import_prompt(data)
    assert sorted(n.type for n in wf2.nodes) == sorted(n.type for n in wf.nodes)

    def _sampler(w):
        found = w.by_type("Sampler")
        assert len(found) == 1, [n.type for n in w.nodes]
        return found[0].params

    s1, s2 = _sampler(wf), _sampler(wf2)
    for key in ("steps", "cfg", "seed", "sampler", "scheduler",
                "width", "height"):
        assert s1[key] == s2[key], (key, s1.get(key), s2.get(key))
    # 连线也要复原
    dec1 = wf.by_type("VAEDecode")[0]
    dec2 = wf2.by_type("VAEDecode")[0]
    assert len(dec1.inputs) == len(dec2.inputs) == 2


@check("test_export_reports_unmappable_nodes")
def test_export_reports_unmappable_nodes():
    from workflow.schema import Node, Workflow

    wf = Workflow(name="只有本项目才有的节点")
    wf.add(Node(id="t1", type="TiledHiresFix", inputs={}, params={}))
    data, rep = cc.export_prompt(wf)
    assert not data
    assert ("t1", "TiledHiresFix") in rep.unsupported, rep.unsupported


@check("test_report_summary_is_actionable")
def test_report_summary_is_actionable():
    graph = _api_graph()
    graph["99"] = {"class_type": "TotallyUnknown", "inputs": {}}
    _, rep = cc.import_prompt(graph)
    text = rep.summary_text()
    assert "不支持" in text and "TotallyUnknown" in text
    assert not rep.ok


@check("test_editor_has_comfy_actions")
def test_editor_has_comfy_actions():
    """编辑器必须真的挂了导入/导出动作，且映射只在 comfy_compat 里一份。"""
    src = io.open(ROOT / "ui" / "workflow_editor" / "window.py",
                  encoding="utf-8").read()
    for need in ('act("导入 ComfyUI"', 'act("导出 ComfyUI"',
                 "def on_import_comfy", "def on_export_comfy",
                 "comfy.load_file(", "comfy.export_prompt("):
        assert need in src, f"window.py 缺少 {need!r}"
    assert "class_type" not in src, "界面里不该出现 class_type（映射归 comfy_compat）"
    # 节点库要标红危险节点
    pal = io.open(ROOT / "ui" / "workflow_editor" / "palette.py",
                  encoding="utf-8").read()
    assert 'RISKY_NODE_TYPES = {"PythonScript"}' in pal
    assert "RISKY_NODE_TYPES:" in pal.replace(" in RISKY_NODE_TYPES", " in RISKY_NODE_TYPES")
    assert '"高级"' in pal, "分类顺序里要留出「高级」"


@check("test_subdir_filenames_are_reduced_to_basename")
def test_subdir_filenames_are_reduced_to_basename():
    """ComfyUI 常写 "SDXL\\x.safetensors"，本项目按文件名查模型目录。"""
    api = {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "SDXL\\mix2_4.safetensors"}},
        "5": {"class_type": "LoraLoader",
              "inputs": {"model": ["4", 0], "clip": ["4", 1],
                         "lora_name": "char/red.safetensors",
                         "strength_model": 0.8, "strength_clip": 0.8}},
    }
    wf, rep = cc.import_prompt(api)
    ck = wf.by_type("LoadCheckpoint")[0]
    lora = wf.by_type("LoadLoRA")[0]
    assert ck.params["path"] == "mix2_4.safetensors", ck.params
    assert lora.params["lora_1"] == "red.safetensors", lora.params
    # 归一必须留痕，不能悄悄改
    assert any("子目录" in w for w in rep.warnings), rep.warnings


print("=" * 60)
if FAILS:
    print(f"总计：FAIL {len(FAILS)}")
    for name, err in FAILS:
        print(f"  - {name}: {err}")
    sys.exit(1)
print("全部通过")
sys.exit(0)
