from __future__ import annotations

import json
import re

import requests

IDS = ["lBjpEnIHb1s", "JK1KAqvhCno"]
VIEWER = "https://www.ytdataviewer.com/"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/136 Safari/537.36"

s = requests.Session()
s.headers.update({"User-Agent": UA})
page = s.get(VIEWER, timeout=45)
page.raise_for_status()
match = re.search(r'window\.YOUTUBE_API_KEY\s*=\s*"([^"]+)"', page.text)
if not match:
    match = re.search(r'const\s+API_KEY\s*=\s*"([^"]+)"', page.text)
if not match:
    raise SystemExit("API key not found")
key = match.group(1)

r = s.get(
    "https://www.googleapis.com/youtube/v3/videos",
    params={"part": "snippet", "id": ",".join(IDS), "key": key},
    headers={"Referer": VIEWER, "Origin": "https://www.ytdataviewer.com"},
    timeout=45,
)
r.raise_for_status()
for item in r.json().get("items") or []:
    snippet = item.get("snippet") or {}
    description = snippet.get("description") or ""
    print("\n===", item.get("id"), "===")
    print("title", snippet.get("title"))
    print("publishedAt", snippet.get("publishedAt"))
    print("DESCRIPTION_BEGIN")
    print(description)
    print("DESCRIPTION_END")
    print("URLS")
    for url in re.findall(r'https?://[^\s<>()\[\]{}"\']+', description):
        print(url.rstrip(".,;:!?)"))
