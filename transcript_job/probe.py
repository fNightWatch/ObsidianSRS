import re
import requests

VID = "7lQTfrOTROo"
keys = [
    ("youtube_web_public", "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"),
]

for name, key in keys:
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "part": "snippet,statistics,contentDetails",
                "id": VID,
                "key": key,
            },
            timeout=30,
        )
        print("\n===", name, r.status_code, len(r.content), "===")
        print(r.text[:5000])
    except Exception as exc:
        print(name, type(exc).__name__, exc)

url = "https://www.ytdataviewer.com/_astro/VideoDataViewer.astro_astro_type_script_index_0_lang.B0qetQFI.js"
r = requests.get(url, timeout=30)
print("\n=== ytdataviewer_js", r.status_code, len(r.content), "===")
for key in sorted(set(re.findall(r"AIza[0-9A-Za-z_-]{30,}", r.text))):
    print("public_browser_key", key)
