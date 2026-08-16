import json
import requests

VID = "7lQTfrOTROo"
URL = f"https://www.youtube.com/watch?v={VID}"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/136 Safari/537.36"
s = requests.Session()
s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9,ru;q=0.8"})

def probe(name, method, url, **kwargs):
    try:
        r = s.request(method, url, timeout=25, **kwargs)
        print("\n===", name, r.status_code, r.headers.get("content-type"), len(r.content), "===")
        print(r.text[:1500].replace("\x00", ""))
    except Exception as e:
        print("\n===", name, "ERROR", type(e).__name__, str(e), "===")

probe("summynews", "GET", "https://www.summynews.com/api.php", params={"v": VID, "format": "json", "lang": "ru"})
probe("notegpt", "GET", "https://notegpt.io/api/v2/video-transcript", params={"platform": "youtube", "video_id": VID}, headers={"User-Agent": UA, "Referer": "https://notegpt.io/youtube-transcript-generator"})
probe("tubetext", "GET", "https://tubetext.vercel.app/youtube/transcript-with-timestamps", params={"video_id": VID})
probe("oembed", "GET", "https://www.youtube.com/oembed", params={"url": URL, "format": "json"})
probe("ryd", "GET", "https://returnyoutubedislikeapi.com/votes", params={"videoId": VID})
for host in [
    "https://inv.thepixora.com/api/v1/videos/",
    "https://inv.nadeko.net/api/v1/videos/",
    "https://invidious.nerdvpn.de/api/v1/videos/",
    "https://pipedapi.tokhmi.xyz/streams/",
    "https://pipedapi.moomoo.me/streams/",
    "https://api-piped.mha.fi/streams/",
    "https://piped-api.garudalinux.org/streams/",
    "https://pipedapi.syncpundit.io/streams/",
]:
    probe(host.split('/')[2], "GET", host + VID)
