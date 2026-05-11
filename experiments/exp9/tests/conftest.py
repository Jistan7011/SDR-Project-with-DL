from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "sourcecode"
EXP2_SOURCE_ROOT = SOURCE_ROOT.parents[1] / "exp2" / "sourcecode"
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != EXP2_SOURCE_ROOT]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

loaded_src = sys.modules.get("src")
loaded_file = Path(getattr(loaded_src, "__file__", "")).resolve() if loaded_src else None
if loaded_file and SOURCE_ROOT not in loaded_file.parents:
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
