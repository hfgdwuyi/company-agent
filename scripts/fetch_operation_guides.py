# -*- coding: utf-8 -*-
"""抓取机械操作/保养指南文本（筑楼人 + 团标 PDF），补充"操作指南"类训练数据。"""
from __future__ import annotations

import html as html_mod
import re
import ssl
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parents[1] / "data" / "finetune" / "guides"
RAW.mkdir(parents=True, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


def strip_html(t: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", t, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</(p|div|h\d|li|tr|td)>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html_mod.unescape(t)
    lines = [ln.strip() for ln in t.splitlines()]
    return "\n".join(ln for ln in lines if ln)


# 筑楼人页面（操作/保养指导书）
ZHULOUREN = {
    "maintenance_guide_mech": "http://www.zhulouren.com/n231866.html",  # 机械设备维护保养作业指书
    "maintenance_guide_equip": "https://www.zhulouren.com/n458526.html",  # 设备保养作业指导书
}

# 团标 PDF（ICS 标准）
STANDARDS = {
    "ics_equipment_maintenance": "https://www.ttbz.org.cn/upload/file/20251212/6390115621801322237777396.pdf",
}


def main() -> int:
    ok = 0
    for name, url in ZHULOUREN.items():
        try:
            data = fetch(url)
            text = None
            for enc in ("utf-8", "gb18030"):
                try:
                    d = data.decode(enc)
                    if "\ufffd" not in d and len(d) > 300:
                        text = strip_html(d)
                        break
                except UnicodeDecodeError:
                    continue
            if not text:
                continue
            # 截取正文：找"指导书/规程"标题之后
            idx = text.find("指导书")
            if idx == -1:
                idx = text.find("规程")
            body = text[idx:] if idx != -1 else text
            (RAW / f"{name}.txt").write_text(body[:8000], encoding="utf-8")
            print(f"OK  {name}: {len(body[:8000])} 字符")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__} {str(e)[:80]}")

    for name, url in STANDARDS.items():
        try:
            data = fetch(url, timeout=120)
            p = RAW / f"{name}.pdf"
            p.write_bytes(data)
            print(f"OK  {name}: {len(data)//1024} KB (PDF)")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__} {str(e)[:80]}")

    print(f"guides: {ok}/{len(ZHULOUREN) + len(STANDARDS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
