from __future__ import annotations

import html
import json
import re
import time
import xml.etree.ElementTree as ET

import requests

FEED = "https://cloud.mave.digital/36719"
TARGETS = [
    "Зарубежные практики и работа после МГУ",
    "Личная жизнь и обучение в МГУ",
]

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 podcast-feed-recovery/1.0"})

cdx = s.get(
    "https://web.archive.org/cdx/search/cdx",
    params={
        "url": FEED,
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,length,digest",
        "filter": "statuscode:200",
        "collapse": "digest",
        "limit": "200",
    },
    timeout=90,
)
cdx.raise_for_status()
rows = cdx.json()
print("captures", max(0, len(rows) - 1))
print(json.dumps(rows[:10], ensure_ascii=False))

found = {}
for row in rows[1:]:
    timestamp, original = row[0], row[1]
    url = f"https://web.archive.org/web/{timestamp}id_/{original}"
    try:
        response = s.get(url, timeout=90)
        response.raise_for_status()
        print("snapshot", timestamp, len(response.content), response.headers.get("content-type"))
        root = ET.fromstring(response.content)
        for item in root.findall("./channel/item"):
            title = html.unescape((item.findtext("title") or "").strip())
            description = html.unescape(item.findtext("description") or "")
            haystack = title + " " + re.sub(r"<[^>]+>", " ", description)
            for target in TARGETS:
                if target.casefold() in haystack.casefold() and target not in found:
                    enclosure = item.find("enclosure")
                    found[target] = {
                        "snapshot": timestamp,
                        "title": title,
                        "pubDate": item.findtext("pubDate") or "",
                        "link": item.findtext("link") or "",
                        "guid": item.findtext("guid") or "",
                        "enclosure": enclosure.attrib if enclosure is not None else {},
                    }
                    print("FOUND", target, json.dumps(found[target], ensure_ascii=False))
        if len(found) == len(TARGETS):
            break
    except Exception as exc:
        print("snapshot_error", timestamp, type(exc).__name__, str(exc))
    time.sleep(1)

print("RESULT", json.dumps(found, ensure_ascii=False, indent=2))
if len(found) != len(TARGETS):
    raise SystemExit(2)
