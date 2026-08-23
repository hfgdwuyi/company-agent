# -*- coding: utf-8 -*-
"""清理 data/raw 下的重入库副本（形如 {stem}-{8hex}.{ext}），只保留原始语料。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pat = re.compile(r"-[0-9a-f]{8}\.(pdf|docx|txt|md)$")
removed = 0
for f in sorted(RAW.iterdir()):
    if pat.search(f.name):
        f.unlink()
        print("removed:", f.name)
        removed += 1
print(f"removed {removed} copies; remaining {len(list(RAW.iterdir()))} files")
