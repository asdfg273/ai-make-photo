# -*- coding: utf-8 -*-
"""全量检查：语法 + 未定义名"""
import ast
import builtins
import os

root = r"E:\pythonprojects\ai  make photo"
skip = {"venv", "third_party", "__pycache__", ".git"}
BUILTINS = set(dir(builtins))
IGNORE = {"__file__", "__name__", "__doc__", "__package__"}


def collect_defs(tree):
    defined = set()

    def _add_args(args):
        for a in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
            defined.add(a.arg)
        if args.vararg:
            defined.add(args.vararg.arg)
        if args.kwarg:
            defined.add(args.kwarg.arg)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            _add_args(node.args)
        elif isinstance(node, ast.Lambda):
            # ⚠️ 以前漏了这里：lambda 的参数域没被收集，
            # 导致 `lambda v: ...` 里的 v 被误报成"未定义名"，
            # 真信号（如 Path 未 import）反而被一堆误报淹没。
            _add_args(node.args)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    defined.add(a.asname or a.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
    return defined


hits = []
total = 0
syntax_bad = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for fn in filenames:
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        total += 1
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError as e:
            syntax_bad.append((p, e.lineno, e.msg))
            continue
        defined = collect_defs(tree)
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in defined or node.id in BUILTINS or node.id in IGNORE:
                    continue
                key = (p, node.id)
                if key in seen:
                    continue
                seen.add(key)
                hits.append((p, node.lineno, node.id))

print("== 语法 ==")
print("检查文件数:", total, "| 语法失败:", len(syntax_bad))
for p, ln, msg in syntax_bad:
    print("  ", p, "行", ln, ":", msg)
print("== 疑似未定义名 ==")
print("命中数:", len(hits))
for p, ln, name in hits:
    print("  ", p, ":", ln, ":", name)
