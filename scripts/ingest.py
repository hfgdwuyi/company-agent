# -*- coding: utf-8 -*-
"""知识库 CLI：批量入库 / 清单 / 删除。

用法：
  python scripts/ingest.py data/raw                      # 批量入库目录
  python scripts/ingest.py some.pdf                      # 入库单个文件
  python scripts/ingest.py --list                        # 文档清单
  python scripts/ingest.py --delete <doc_id>             # 删除文档
  python scripts/ingest.py data/raw --collection mykb    # 指定集合
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from company_rag import knowledge_base, vector_store  # noqa: E402
from company_rag.config import settings  # noqa: E402
from company_rag.document_parser import SUPPORTED_EXTENSIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="企业知识库 CLI")
    parser.add_argument("target", nargs="?", help="文件或目录路径")
    parser.add_argument("--collection", default=settings.default_collection)
    parser.add_argument("--list", action="store_true", help="列出文档")
    parser.add_argument("--delete", metavar="DOC_ID", help="删除指定文档")
    parser.add_argument("--stats", action="store_true", help="显示知识库统计")
    args = parser.parse_args()

    if args.list:
        docs = knowledge_base.list_documents(args.collection)
        print(f"共 {len(docs)} 份文档:")
        for d in docs:
            print(f"  {d['filename']}  doc_id={d['doc_id']}  chunks={d['chunk_count']}")
        return 0

    if args.delete:
        n = knowledge_base.delete_document(args.collection, args.delete)
        print(f"删除 {args.delete}: 移除 {n} 个片段")
        return 0

    if args.stats:
        print(f"集合 {args.collection}: {vector_store.count_chunks(args.collection)} 个片段 / {len(vector_store.list_documents(args.collection))} 份文档")
        return 0

    if not args.target:
        parser.print_help()
        return 1

    target = Path(args.target)
    if target.is_dir():
        metas = knowledge_base.ingest_directory(target, collection=args.collection)
        ok = sum(1 for m in metas if m.status == "ingested")
        print(f"\n入库完成: {ok}/{len(metas)} 成功")
        for m in metas:
            flag = "✅" if m.status == "ingested" else "❌"
            print(f"  {flag} {m.filename}  {m.chunk_count} 块  {m.status}" + (f" ({m.error})" if m.error else ""))
    elif target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"不支持的类型 {target.suffix}，支持: {sorted(SUPPORTED_EXTENSIONS)}")
            return 1
        meta = knowledge_base.ingest_path(target, collection=args.collection)
        print(f"{'✅' if meta.status=='ingested' else '❌'} {meta.filename}: {meta.chunk_count} 块, {meta.pages} 页, 状态={meta.status}")
        if meta.error:
            print(f"   错误: {meta.error}")
    else:
        print(f"路径不存在: {target}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
