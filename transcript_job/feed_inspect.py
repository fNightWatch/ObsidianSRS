from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

STAMP = "20221005222839"
FEED = "https://cloud.mave.digital/36719"
URL = f"https://web.archive.org/web/{STAMP}id_/{FEED}"

r = requests.get(URL, timeout=120)
r.raise_for_status()
root = ET.fromstring(r.content)
items = []
for item in root.findall("./channel/item"):
    title = html.unescape((item.findtext("title") or "").strip())
    desc_raw = html.unescape(item.findtext("description") or "")
    desc = re.sub(r"<[^>]+>", " ", desc_raw)
    desc = re.sub(r"\s+", " ", desc).strip()
    pub = item.findtext("pubDate") or ""
    try:
        dt = parsedate_to_datetime(pub)
        iso = dt.date().isoformat()
    except Exception:
        iso = ""
    enclosure = item.find("enclosure")
    items.append({
        "date": iso,
        "pub": pub,
        "title": title,
        "description": desc,
        "link": item.findtext("link") or "",
        "guid": item.findtext("guid") or "",
        "enclosure": enclosure.attrib if enclosure is not None else {},
    })

print("items", len(items))
print("range", min((x["date"] for x in items if x["date"]), default=""), max((x["date"] for x in items if x["date"]), default=""))

print("\n=== OCT-NOV 2020 ===")
for x in sorted(items, key=lambda z: z["date"]):
    if "2020-09-01" <= x["date"] <= "2020-12-31":
        print(json.dumps(x, ensure_ascii=False))

print("\n=== KEYWORDS ===")
for x in items:
    hay = (x["title"] + " " + x["description"] + " " + x["link"] + " " + x["guid"]).casefold()
    if any(k in hay for k in ["мгу", "зарубеж", "практик", "личн", "обучен", "университет", "lBjpEnIHb1s".casefold(), "JK1KAqvhCno".casefold()]):
        print(json.dumps(x, ensure_ascii=False))

print("\n=== OLDEST 30 ===")
for x in sorted(items, key=lambda z: z["date"])[:30]:
    print(json.dumps(x, ensure_ascii=False))
