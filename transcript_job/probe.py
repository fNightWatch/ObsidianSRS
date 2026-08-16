import re
import requests

page_url = "https://www.ytdataviewer.com/"
js_url = "https://www.ytdataviewer.com/_astro/VideoDataViewer.astro_astro_type_script_index_0_lang.B0qetQFI.js"
for name, url in (("PAGE", page_url), ("JS", js_url)):
    r = requests.get(url, timeout=30); r.raise_for_status(); text = r.text
    print("\n===", name, len(text), "===")
    needles = ["googleapis.com/youtube/v3/videos", "apiKey", "api-key", "api_key", "youtubeApi", "YOUTUBE_API", "data-key", "key="]
    shown = set()
    for needle in needles:
        pos = 0
        while True:
            pos = text.find(needle, pos)
            if pos < 0: break
            chunk = text[max(0, pos-3000):min(len(text), pos+3000)]
            if chunk not in shown:
                print("\n---", needle, "at", pos, "---\n", chunk)
                shown.add(chunk)
            pos += len(needle)
    print("candidate strings")
    for value in sorted(set(re.findall(r'[A-Za-z0-9_-]{30,80}', text))):
        if value.startswith(("AIza", "ABCD", "youtube")) or "key" in value.lower():
            print(value)
