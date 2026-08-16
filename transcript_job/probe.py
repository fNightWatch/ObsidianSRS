import html
import re
import urllib.parse
import requests

SITES = [
    "https://tubealfred.com/tools/youtube-metadata-viewer",
    "https://yttools.co/youtube-video-info-tool",
    "https://www.ytinfo.online/",
    "https://www.ytdataviewer.com/",
]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/136 Safari/537.36"
s = requests.Session(); s.headers.update({"User-Agent": UA})


def interesting(text: str):
    found = set()
    patterns = [
        r'https?://[^"\'`\\\s<>]+',
        r'(?<![A-Za-z0-9_])/(?:api|v1|v2)/[^"\'`\\\s<>]+',
        r'fetch\((.{0,300})',
        r'axios\.(?:get|post)\((.{0,300})',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.I | re.S):
            value = m.group(0).replace("\\u0026", "&")
            low = value.lower()
            if any(x in low for x in ("youtube", "metadata", "video", "api", "fetch")):
                found.add(value[:500])
    return sorted(found)

for page in SITES:
    try:
        r = s.get(page, timeout=30); r.raise_for_status(); source = r.text
        print("\n### PAGE", page, r.status_code, len(source), r.headers.get("content-type"))
        print("forms", re.findall(r'<form[^>]*action=["\']([^"\']+)', source, re.I)[:20])
        print("inline", *interesting(source)[:80], sep="\n")
        scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)', source, re.I)
        print("scripts", len(scripts))
        base = f"{urllib.parse.urlparse(page).scheme}://{urllib.parse.urlparse(page).netloc}"
        for src in scripts:
            url = urllib.parse.urljoin(base, html.unescape(src))
            try:
                js = s.get(url, timeout=30).text
            except Exception:
                continue
            hits = interesting(js)
            if hits:
                print("\nSCRIPT", url, len(js))
                print(*hits[:120], sep="\n")
    except Exception as exc:
        print("\n### ERROR", page, type(exc).__name__, exc)
