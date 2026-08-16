import requests

KEY = "AIzaSyCacWhXdBEFeVczq4NV1WgWf6KS7zLhLQE"
r = requests.get(
    "https://www.googleapis.com/youtube/v3/videos",
    params={
        "part": "snippet,statistics",
        "id": "7lQTfrOTROo",
        "key": KEY,
    },
    headers={"Referer": "https://www.ytdataviewer.com/"},
    timeout=30,
)
print(r.status_code, len(r.content))
print(r.text[:5000])
