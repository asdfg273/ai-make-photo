import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import _bootstrap  # noqa: F401
from _bootstrap import isolated_config  # noqa: F401
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication([])

import importlib.util

spec = importlib.util.spec_from_file_location(
    "t", ROOT / "tests" / "test_ipa_refs_ui.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
try:
    m.test_builder_emits_role_nodes_when_refs_configured()
    print("RESULT: PASS")
except Exception:
    print("RESULT: FAIL")
    traceback.print_exc()
sys.stdout.flush()
os._exit(0)
