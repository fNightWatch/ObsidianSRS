from __future__ import annotations

import html
import json
import re
import urllib.parse

import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/136 Safari/537.36"
IDS = ["JK1KAqvhCno", "lBjpEnIHb1s"]
S = requests.Session()
S.headers.update({
    "User-Agent": UA,
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Referer": "https://www.google.com/",
})


def compact(value: str) -> str:
    return html.unescape(value).replace("\\/", "/").replace("\\u0026", "&")


def interesting(text: str) -> list[str]:
    values = set()
    patterns = [
        r'https?://[^"\'`\\\s<>]+',
        r'(?<![A-Za-z0-9_])/(?:api|ajax|youtube|download|video|media|task|job|proxy|convert)[^"\'`\\\s<>]{0,250}',
        r'(?:fetch|axios\.(?:get|post)|XMLHttpRequest|\.ajax)\s*\(.{0,500}',
        r'<form[^>]{0,1000}>',
        r'<(?:a|button|video|audio|source)[^>]{0,1200}>',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I | re.S):
            value = compact(match.group(0))[:1500]
            low = value.casefold()
            if any(token in low for token in (
                "youtube", "download", "скач", "video", "audio", "media", "api", "fetch", "ajax",
                "mave", "episode", "podcast", ".mp3", ".m4a", ".mp4", "googlevideo", "enclosure",
            )):
                values.add(value)
    return sorted(values)


def dump_page(label: str, url: str, referer: str | None = None) -> None:
    headers = {"Referer": referer} if referer else None
    try:
        response = S.get(url, headers=headers, timeout=50, allow_redirects=True)
        print("\n###", label, response.status_code, response.url, len(response.content), response.headers.get("content-type"))
        print("cookies", dict(response.cookies))
        text = response.text
        print("forms", re.findall(r'<form[^>]*action=["\']([^"\']+)', text, re.I)[:50])
        print("scripts", re.findall(r'<script[^>]+src=["\']([^"\']+)', text, re.I)[:100])
        print("links", [compact(x) for x in re.findall(r'<a[^>]+href=["\']([^"\']+)', text, re.I)[:200]])
        print("interesting")
        print(*interesting(text)[:300], sep="\n")
        base = response.url
        for src in re.findall(r'<script[^>]+src=["\']([^"\']+)', text, re.I):
            script_url = urllib.parse.urljoin(base, html.unescape(src))
            try:
                script = S.get(script_url, headers=headers, timeout=50)
                hits = interesting(script.text)
                if hits:
                    print("\nSCRIPT", script.status_code, script_url, len(script.content))
                    print(*hits[:300], sep="\n")
            except Exception as exc:
                print("SCRIPT_ERROR", script_url, type(exc).__name__, str(exc))
    except Exception as exc:
        print("\n###", label, "ERROR", type(exc).__name__, str(exc))


for video_id in IDS:
    for referer in (
        "https://www.google.com/",
        "https://qna.center/question/7311394",
        "https://www.youtube.com/",
    ):
        dump_page(
            f"qna-{video_id}-{urllib.parse.urlparse(referer).netloc}",
            f"https://video.qna.center/youtube/{video_id}",
            referer,
        )

dump_page("mave-root", "https://glebsolomin.mave.digital/")
dump_page("mave-feed", "https://cloud.mave.digital/36719")

print("\n### WAYBACK CDX")
try:
    response = S.get(
        "https://web.archive.org/cdx/search/cdx",
        params={
            "url": "glebsolomin.mave.digital/*",
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype,length,digest",
            "filter": "statuscode:200",
            "collapse": "urlkey",
            "limit": "5000",
        },
        timeout=120,
    )
    response.raise_for_status()
    rows = response.json()
    print("rows", max(0, len(rows) - 1))
    for row in rows[1:]:
        print("CDX", json.dumps(row, ensure_ascii=False))
    candidates = []
    for row in rows[1:]:
        timestamp, original = row[0], row[1]
        low = original.casefold()
        if any(x in low for x in ("ep-", "feed", "rss", "api", "json", "podcast", "episode")) or original.rstrip("/") == "https://glebsolomin.mave.digital":
            candidates.append((timestamp, original))
    for timestamp, original in candidates[:200]:
        archive_url = f"https://web.archive.org/web/{timestamp}id_/{original}"
        try:
            archived = S.get(archive_url, timeout=90)
            print("\nARCHIVE", timestamp, original, archived.status_code, len(archived.content), archived.headers.get("content-type"))
            hits = interesting(archived.text)
            if hits:
                print(*hits[:300], sep="\n")
            for video_id in IDS:
                if video_id in archived.text:
                    print("FOUND_VIDEO_ID", video_id, archive_url)
            for needle in ("Личная жизнь", "Зарубежные практики", "Иван Венчиков", "Борис Спирин"):
                if needle.casefold() in archived.text.casefold():
                    print("FOUND_TEXT", needle, archive_url)
        except Exception as exc:
            print("ARCHIVE_ERROR", timestamp, original, type(exc).__name__, str(exc))
except Exception as exc:
    print("CDX_ERROR", type(exc).__name__, str(exc))
