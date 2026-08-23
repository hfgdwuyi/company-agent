# -*- coding: utf-8 -*-
"""清理重复入库的文档副本（doc_id 形如 {stem}-{8hex}-{8hex}）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from company_rag import knowledge_base, vector_store  # noqa: E402
from company_rag.config import settings  # noqa: E402

pat = re.compile(r"-[0-9a-f]{8}-[0-9a-f]{8}$")
docs = vector_store.list_documents(settings.default_collection)
removed = 0
for d in docs:
    if pat.search(d["doc_id"]):
        n = knowledge_base.delete_document(settings.default_collection, d["doc_id"])
        print(f"deleted {d['doc_id']} ({n} chunks)")
        removed += 1
print("removed copies:", removed)
print("remaining:", len(vector_store.list_documents(settings.default_collection)))
