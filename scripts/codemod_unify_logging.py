# codemod_unify_logging.py
# ============================================================
#  一次性代码迁移:把业务代码里的 print(...) 收编进 logging。
#  - 基于 tokenize 定位完整语句,保留原格式与注释
#  - 按内容关键词自动分级 (debug / info / warning / error)
#  - 多参数 print 改为 " ".join(map(str, (...))) 保持输出语义
#  - 剥离 flush=True;带 sep=/end= 的语句跳过不处理
#  - 文件缺少 logger 时自动补 `logger = logging.getLogger(__name__)`
#  用法:
#    python scripts/codemod_unify_logging.py          # dry-run 预览
#    python scripts/codemod_unify_logging.py --apply  # 实际写入
# ============================================================
import io
import os
import re
import sys
import tokenize

TARGET_FILES = [
    "main.py",
    "core/config_manager.py",
    "core/model_manager.py",
    "ui/disclaimer.py",
    "ui/gallery_panel.py",
    "ui/preset_manager.py",
    "ui/ui_builder.py",
    "ui/video_panel_mixin.py",
    "utils/app_events.py",
    "utils/app_generation.py",
    "utils/chattts_patch.py",
    "utils/extension_manager.py",
    "utils/image_processor.py",
    "utils/model_downloader.py",
    "utils/prompt_enhancer.py",
    "utils/rife_interpolate.py",
    "utils/tiled_diffusion.py",
    "utils/tts_engine.py",
    "utils/video_gen.py",
    "photo_turn/mixin_filters.py",
    "photo_turn/pro_editor_qt.py",
]

ERROR_KW   = ("❌", "Traceback", "traceback", "错误堆栈")
WARNING_KW = ("⚠️", "⚠", "警告", "未找到", "降级", "不存在", "失败", "错误", "异常")
DEBUG_KW   = ("🟢", "[SHOW]", "[APPLY]", "[GALLERY", "[CONNECT]",
              "[PRELOAD", "[DEBUG", "step ", "[TASK]")

def pick_level(inner: str) -> str:
    # 作者标注优先:❌=error > ⚠️=warning > 调试标记=debug > info
    for kw in ERROR_KW:
        if kw in inner:
            return "error"
    for kw in WARNING_KW:
        if kw in inner:
            return "warning"
    for kw in DEBUG_KW:
        if kw in inner:
            return "debug"
    return "info"


def split_top_level_args(s: str):
    """按顶层逗号切分参数(忽略括号/字符串内的逗号)"""
    args, depth, cur = [], 0, []
    in_str, str_ch, esc = False, "", False
    for ch in s:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == str_ch:
                in_str = False
            continue
        if ch in "\"'":
            in_str, str_ch = True, ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1; cur.append(ch)
        elif ch in ")]}":
            depth -= 1; cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    tail = "".join(cur)
    if tail.strip():
        args.append(tail)
    return args


def find_print_spans(source: str):
    """返回 [(start_offset, end_offset, inner_text), ...] 按出现顺序"""
    spans = []
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    line_starts = [0]
    for m in re.finditer("\n", source):
        line_starts.append(m.end())

    def off(line, col):
        return line_starts[line - 1] + col

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if (tok.type == tokenize.NAME and tok.string == "print"
                and i + 1 < len(tokens)
                and tokens[i + 1].type == tokenize.OP
                and tokens[i + 1].string == "("):
            prev = tokens[i - 1] if i > 0 else None
            if prev is not None and prev.type == tokenize.OP and prev.string == ".":
                i += 1
                continue
            # 找匹配的右括号
            depth, j = 0, i + 1
            while j < len(tokens):
                t = tokens[j]
                if t.type == tokenize.OP:
                    if t.string in "([{":
                        depth += 1
                    elif t.string in ")]}":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            if j >= len(tokens):
                break
            start = off(tok.start[0], tok.start[1])
            open_paren = off(tokens[i + 1].start[0], tokens[i + 1].start[1])
            end = off(tokens[j].end[0], tokens[j].end[1])
            inner = source[open_paren + 1: end - 1]
            spans.append((start, end, inner))
            i = j + 1
        else:
            i += 1
    return spans


def transform_inner(inner: str):
    """返回 (level, new_inner) 或 None 表示跳过"""
    if re.search(r"\b(sep|end)\s*=", inner):
        return None  # 特殊格式输出,保留 print
    level = "info"
    if re.search(r"\bfile\s*=\s*sys\.stderr", inner):
        level = "error"
    cleaned = re.sub(r",?\s*flush\s*=\s*True", "", inner)
    cleaned = re.sub(r",?\s*file\s*=\s*sys\.stderr", "", cleaned).strip()
    if cleaned.endswith(","):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return (level if level != "info" else "debug", '""')
    args = split_top_level_args(cleaned)
    kw_level = pick_level(cleaned)
    if level == "info":
        level = kw_level
    if len(args) == 1:
        return level, args[0].strip()
    joined = ", ".join(a.strip() for a in args)
    return level, f'" ".join(map(str, ({joined})))'


def ensure_logger(source: str) -> str:
    if re.search(r"^logger\s*=", source, re.M):
        return source
    if re.search(r"import.*\blogger\b", source):  # from ... import logger (单行)
        return source
    # from ... import (\n ... logger ... ) 多行形式
    if re.search(r"from\s+\S+\s+import\s*\([^)]*\blogger\b[^)]*\)", source, re.S):
        return source
        return source
    lines = source.splitlines(keepends=True)
    # 只认顶格(列 0)的 import/from 行,插入到最后一个之后;
    # 多行 import(以未闭合括号结尾)要跳到整条语句结束,
    # 避免插进类体/函数体或 import 语句中间
    insert_at = None
    has_logging = False
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if re.match(r"import\s+logging\b", line):
            has_logging = True
        if re.match(r"(import|from)\s+\S", line):
            depth = line.count("(") - line.count(")")
            while depth > 0 and idx + 1 < len(lines):
                idx += 1
                depth += lines[idx].count("(") - lines[idx].count(")")
            insert_at = idx + 1
        idx += 1
    block = []
    if not has_logging:
        block.append("import logging\n")
    block.append("\nlogger = logging.getLogger(__name__)\n")
    lines[insert_at:insert_at] = block
    return "".join(lines)


def process(path: str, apply: bool):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    spans = find_print_spans(src)
    if not spans:
        return 0, 0
    out, last, converted, skipped = [], 0, 0, 0
    for start, end, inner in spans:
        r = transform_inner(inner)
        if r is None:
            skipped += 1
            continue
        level, new_inner = r
        out.append(src[last:start])
        out.append(f"logger.{level}({new_inner})")
        last = end
        converted += 1
    out.append(src[last:])
    new_src = ensure_logger("".join(out))
    if apply and converted:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_src)
    return converted, skipped


def main():
    apply = "--apply" in sys.argv
    total_c = total_s = 0
    for rel in TARGET_FILES:
        if not os.path.exists(rel):
            print(f"[MISS] {rel}")
            continue
        c, s = process(rel, apply)
        total_c += c; total_s += s
        print(f"{'[WRITE]' if apply else '[DRY ]'} {rel}: 转换 {c}, 跳过 {s}")
    print(f"\n合计: 转换 {total_c}, 跳过 {total_s}")


if __name__ == "__main__":
    main()
