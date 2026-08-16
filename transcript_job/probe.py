import json
import requests

VID = "7lQTfrOTROo"
clients = [
    (
        "web",
        {
            "clientName": "WEB",
            "clientVersion": "2.20231219.04.00",
            "hl": "en",
            "gl": "US",
        },
        "com.google.android.youtube/2.20231219.04.00",
    ),
    (
        "android",
        {
            "clientName": "ANDROID",
            "clientVersion": "20.10.38",
            "androidSdkVersion": 32,
            "osName": "Android",
            "osVersion": "12",
            "hl": "en",
            "gl": "US",
        },
        "com.google.android.youtube/20.10.38 (Linux; U; Android 12) gzip",
    ),
    (
        "ios",
        {
            "clientName": "IOS",
            "clientVersion": "20.10.4",
            "deviceMake": "Apple",
            "deviceModel": "iPhone16,2",
            "osName": "iPhone",
            "osVersion": "18.3.2.22D82",
            "hl": "en",
            "gl": "US",
        },
        "com.google.ios.youtube/20.10.4 (iPhone16,2; U; CPU iOS 18_3_2 like Mac OS X)",
    ),
]

for name, client, ua in clients:
    body = {
        "context": {"client": client},
        "videoId": VID,
        "contentCheckOk": True,
        "racyCheckOk": True,
    }
    try:
        r = requests.post(
            "https://www.youtube.com/youtubei/v1/player",
            json=body,
            headers={
                "User-Agent": ua,
                "Content-Type": "application/json",
                "X-YouTube-Client-Name": client["clientName"],
                "X-YouTube-Client-Version": client["clientVersion"],
            },
            timeout=30,
        )
        print("\n===", name, r.status_code, r.headers.get("content-type"), len(r.content), "===")
        try:
            data = r.json()
            print("playability", json.dumps(data.get("playabilityStatus"), ensure_ascii=False)[:1500])
            print("videoDetails", json.dumps(data.get("videoDetails"), ensure_ascii=False)[:2000])
            print("microformat", json.dumps(data.get("microformat"), ensure_ascii=False)[:2000])
            print("captions_keys", list((data.get("captions") or {}).keys()))
        except Exception:
            print(r.text[:3000])
    except Exception as exc:
        print("\n===", name, "ERROR", type(exc).__name__, str(exc), "===")
