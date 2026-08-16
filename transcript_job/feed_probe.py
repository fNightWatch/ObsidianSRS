from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

import requests

FEED = "https://cloud.mave.digital/36719"
TARGETS = {
    "lBjpEnIHb1s": "Зарубежные практики и работа после МГУ",
    "JK1KAqvhCno": "Личная жизнь и обучение в МГУ",
}

response = requests.get(FEED, timeout=60)
response.raise_for_status()
print("feed", response.status_code, len(response.content), response.headers.get("content-type"))
root = ET.fromstring(response.content)
items = root.findall("./channel/item")
print("items", len(items))

for video_id, needle in TARGETS.items():
    matches = []
    for item in items:
        title = html.unescape((item.findtext("title") or "").strip())
        description = html.unescape(item.findtext("description") or "")
        if needle.casefold() in title.casefold() or needle.casefold() in re.sub(r"<[^>]+>", " ", description).casefold():
            enclosure = item.find("enclosure")
            matches.append((title, item.findtext("pubDate") or "", enclosure.attrib if enclosure is not None else {}, item.findtext("link") or ""))
    print("\nTARGET", video_id, needle, "matches", len(matches))
    for title, date, enclosure, link in matches:
        print("title", title)
        print("pubDate", date)
        print("link", link)
        print("enclosure", enclosure)
        url = enclosure.get("url")
        if url:
            try:
                head = requests.head(url, allow_redirects=True, timeout=60)
                print("head", head.status_code, head.url, head.headers.get("content-type"), head.headers.get("content-length"), head.headers.get("accept-ranges"))
            except Exception as exc:
                print("head_error", type(exc).__name__, str(exc))
