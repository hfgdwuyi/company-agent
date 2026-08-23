# -*- coding: utf-8 -*-
"""测试 pymupdf 提取 PDF 图片。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import fitz

doc = fitz.open(r"C:\AI\company-rag\data\raw\attention_is_all_you_need.pdf")
print("pages:", len(doc))
total = 0
for pno in range(len(doc)):
    page = doc[pno]
    imgs = page.get_images(full=True)
    if imgs:
        for i, img in enumerate(imgs):
            xref = img[0]
            info = doc.extract_image(xref)
            total += 1
            print(f"page {pno+1}: img{i} xref={xref} {info['ext']} {len(info['image'])//1024}KB")
print("total images:", total)
