from __future__ import annotations

import json
import urllib.parse

import requests

SHOW_ID = "1540559311"
TARGETS = [
    "Личная жизнь и обучение в МГУ",
    "Зарубежные практики и работа после МГУ",
    "JK1KAqvhCno",
    "lBjpEnIHb1s",
]
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 ApplePodcastsProbe/1.0"})


def dump(label: str, url: str, params: dict[str, str]) -> dict:
    response = S.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    print("\n###", label, response.url, "count", data.get("resultCount"))
    print(json.dumps(data, ensure_ascii=False, indent=2)[:500000])
    return data


for country in ("ru", "us", "gb"):
    data = dump(
        f"lookup-{country}",
        "https://itunes.apple.com/lookup",
        {
            "id": SHOW_ID,
            "entity": "podcastEpisode",
            "limit": "200",
            "country": country,
        },
    )
    episodes = [x for x in data.get("results") or [] if x.get("wrapperType") == "podcastEpisode"]
    print("EPISODES", country, len(episodes))
    for episode in sorted(episodes, key=lambda x: x.get("releaseDate") or "")[:25]:
        print("EARLY", json.dumps(episode, ensure_ascii=False))
    for episode in episodes:
        hay = json.dumps(episode, ensure_ascii=False).casefold()
        if any(target.casefold() in hay for target in TARGETS):
            print("MATCH", country, json.dumps(episode, ensure_ascii=False))
            media = episode.get("episodeUrl")
            if media:
                try:
                    head = S.head(media, allow_redirects=True, timeout=60)
                    print("HEAD", head.status_code, head.url, dict(head.headers))
                except Exception as exc:
                    print("HEAD_ERROR", type(exc).__name__, str(exc))

for country in ("ru", "us", "gb"):
    for term in TARGETS[:2]:
        dump(
            f"search-{country}-{term}",
            "https://itunes.apple.com/search",
            {
                "term": term,
                "entity": "podcastEpisode",
                "limit": "200",
                "country": country,
            },
        )

dump("show", "https://itunes.apple.com/lookup", {"id": SHOW_ID, "country": "ru"})
