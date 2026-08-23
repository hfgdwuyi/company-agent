# -*- coding: utf-8 -*-
"""抓取真实中文法规全文（政府官网 HTML → TXT/MD），补充中文语料。"""
from __future__ import annotations

import html as html_mod
import re
import ssl
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 法规名 -> (URL, 标题)
LAWS = {
    "data_security_law": (
        "https://sjj.yanan.gov.cn/zfxxgk/fdzdgknr/fgwj/flfg/1656607936505516033.html",
        "中华人民共和国数据安全法",
    ),
    "personal_info_protection_law": (
        "http://www.npc.gov.cn/npc/c2/c30834/202108/t20210820_313088.html",
        "中华人民共和国个人信息保护法",
    ),
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def strip_html(t: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", t, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</(p|div|h\d|li|tr|td)>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html_mod.unescape(t)
    lines = [ln.strip() for ln in t.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40, context=ctx) as r:
        data = r.read()
    for enc in ("utf-8", "gb18030"):
        try:
            d = data.decode(enc)
            if "\ufffd" not in d and len(d) > 500:
                return strip_html(d)
        except UnicodeDecodeError:
            continue
    return strip_html(data.decode("utf-8", errors="replace"))


def main() -> int:
    ok = 0
    for name, (url, title) in LAWS.items():
        try:
            text = fetch_text(url)
            # 截取正文（法规标题之后），避免导航噪音
            idx = text.find(title)
            body = text[idx:] if idx != -1 else text
            body = body.strip()
            (RAW / f"{name}.txt").write_text(body, encoding="utf-8")
            (RAW / f"{name}.md").write_text(f"# {title}\n\n{body}", encoding="utf-8")
            print(f"OK  {name}: {len(body)} 字符 <- {url}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
    print(f"laws: {ok}/{len(LAWS)}")
    return 0 if ok == len(LAWS) else 1


if __name__ == "__main__":
    sys.exit(main())
