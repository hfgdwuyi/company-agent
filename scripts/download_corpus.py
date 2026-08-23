# -*- coding: utf-8 -*-
"""下载真实测试语料：arXiv 论文 PDF + 国新办发布会实录 PDF + 法律法规文本。
保存到 data/raw/ 下。"""
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def download(url, dest, timeout=120):
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print(f"skip (exists): {os.path.basename(dest)}")
        return True
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            f.write(r.read())
        size = os.path.getsize(dest)
        print(f"OK  {os.path.basename(dest)}  {size/1024:.0f} KB  <- {url}")
        return size > 5000
    except Exception as e:
        print(f"FAIL {os.path.basename(dest)}: {e}")
        try:
            os.remove(dest)
        except OSError:
            pass
        return False


# ---------- arXiv 真实论文（英文，稳定 URL） ----------
ARXIV = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
    "retrieval_augmented_generation.pdf": "https://arxiv.org/pdf/2005.11401",
    "llama2_open_foundation_models.pdf": "https://arxiv.org/pdf/2307.09288",
    "bert_pretraining_deep_bidirectional.pdf": "https://arxiv.org/pdf/1810.04805",
    "bge_m3_embedding.pdf": "https://arxiv.org/pdf/2402.03216",
    "self_rag_learning_to_retrieve.pdf": "https://arxiv.org/pdf/2310.11511",
    "lost_in_the_middle.pdf": "https://arxiv.org/pdf/2307.03172",
    "toolformer_language_models_teach.pdf": "https://arxiv.org/pdf/2302.04761",
}

# ---------- 国新办新闻发布会文字实录 PDF（真实中文文档） ----------
SCIO = {
    "scio_rural_road_2024.pdf": "http://www.scio.gov.cn/live/2024/35161/qwxz/202412/P020241202347663859724.pdf",
    "scio_marine_ecology_2024.pdf": "http://www.scio.gov.cn/live/2024/34295/qwxz/202407/P020240712508820336399.pdf",
}

print("---- arXiv 下载开始 ----")
ok = 0
for name, url in ARXIV.items():
    if download(url, os.path.join(RAW, name)):
        ok += 1
print(f"arXiv: {ok}/{len(ARXIV)}")

print("---- 国新办 PDF 下载开始 ----")
ok2 = 0
for name, url in SCIO.items():
    if download(url, os.path.join(RAW, name)):
        ok2 += 1
print(f"SCIO: {ok2}/{len(SCIO)}")

print("---- 法规全文（gov.cn HTML → TXT/MD）开始 ----")
import re
import html as html_mod

LAWS_HTML = {
    "data_security_law": "http://www.gov.cn/xinwen/2021-06/11/content_5616919.htm",
    "personal_info_protection_law": "http://www.gov.cn/xinwen/2021-08/20/content_5631930.htm",
}


def html_to_text(html_text):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|h\d|li|tr)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_mod.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


ok3 = 0
for name, url in LAWS_HTML.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8", errors="ignore")
        text = html_to_text(raw)
        txt_path = os.path.join(RAW, f"{name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        md_path = os.path.join(RAW, f"{name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n" + text)
        print(f"OK  {name}.txt/.md  {len(text)} chars")
        ok3 += 1
    except Exception as e:
        print(f"FAIL {name}: {e}")
print(f"LAWS: {ok3}/{len(LAWS_HTML)}")
print("DONE")
