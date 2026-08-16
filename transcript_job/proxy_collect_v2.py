from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

import proxy_collect as b

PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/all/data.txt",
]
_proxy_lock = threading.Lock()
_proxies: list[str] | None = None


def proxy_session(proxy: str) -> requests.Session:
    s = requests.Session()
    value = proxy if "://" in proxy else "http://" + proxy
    s.proxies.update({"http": value, "https": value})
    s.headers.update({
        "User-Agent": b.UA,
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Cookie": "CONSENT=YES+cb.20210328-17-p0.en+FX+471",
    })
    return s


def discover_proxies(sample_id: str) -> list[str]:
    global _proxies
    with _proxy_lock:
        if _proxies is not None:
            return _proxies
        candidates: list[str] = []
        for source in PROXY_SOURCES:
            try:
                text = b.session().get(source, timeout=25).text
                for line in text.splitlines():
                    value = line.strip()
                    if not value or value.startswith("#"):
                        continue
                    if re.fullmatch(r"(?:https?://)?[^\s:]+:\d+", value):
                        candidates.append(value)
            except Exception as exc:
                print(f"proxy list {source}: {type(exc).__name__}:{exc}", flush=True)
        candidates = list(dict.fromkeys(candidates))[:500]
        random.Random(160826).shuffle(candidates)
        target = f"https://www.youtube.com/watch?v={sample_id}&hl=en&gl=US"

        def test(proxy: str) -> str | None:
            try:
                r = proxy_session(proxy).get(target, timeout=8)
                if r.status_code == 200 and "ytInitialPlayerResponse" in r.text and "google.com/sorry" not in r.url:
                    return proxy
            except Exception:
                pass
            return None

        good: list[str] = []
        with cf.ThreadPoolExecutor(max_workers=48) as pool:
            futures = [pool.submit(test, x) for x in candidates[:350]]
            for future in cf.as_completed(futures):
                value = future.result()
                if value:
                    good.append(value)
                    print(f"working proxy {len(good)}", flush=True)
                    if len(good) >= 16:
                        for f in futures:
                            f.cancel()
                        break
        _proxies = good
        print(f"youtube proxies usable: {len(good)}", flush=True)
        return good


def decode_player(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for marker in (
        "var ytInitialPlayerResponse = ",
        "ytInitialPlayerResponse = ",
        'window["ytInitialPlayerResponse"] = ',
        '"ytInitialPlayerResponse":',
    ):
        pos = text.find(marker)
        if pos < 0:
            continue
        start = text.find("{", pos + len(marker))
        if start < 0:
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict) and (value.get("videoDetails") or value.get("playabilityStatus")):
                return value
        except Exception:
            pass
    raise RuntimeError("player response not found")


def caption_tracks(player: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return player["captions"]["playerCaptionsTracklistRenderer"]["captionTracks"]
    except Exception:
        return []


def metadata_from_player(player: dict[str, Any]) -> b.Metadata:
    details = player.get("videoDetails") or {}
    renderer = ((player.get("microformat") or {}).get("playerMicroformatRenderer") or {})
    title = b.clean(details.get("title") or renderer.get("title", {}).get("simpleText"))
    try:
        views = int(details.get("viewCount")) if details.get("viewCount") is not None else None
    except Exception:
        views = None
    published = b.normalize_date(renderer.get("publishDate") or renderer.get("uploadDate"))
    return b.Metadata(title, views, published)


def youtube_via_proxy(video: b.Video, need_transcript: bool) -> tuple[list[b.Segment], b.Metadata, str]:
    errors: list[str] = []
    proxies = discover_proxies(video.video_id)
    order = list(proxies)
    random.Random(video.video_id).shuffle(order)
    for proxy in order[:10]:
        try:
            s = proxy_session(proxy)
            r = s.get(f"https://www.youtube.com/watch?v={video.video_id}&hl=en&gl=US", timeout=18)
            r.raise_for_status()
            player = decode_player(r.text)
            meta = metadata_from_player(player)
            if not need_transcript:
                return [], meta, ""
            tracks = caption_tracks(player)
            tracks.sort(key=lambda x: (
                0 if b.clean(x.get("languageCode")) == video.channel else 1,
                0 if b.clean(x.get("languageCode")).startswith(video.channel) else 1,
                0 if x.get("kind") != "asr" else 1,
            ))
            for track in tracks:
                lang = b.clean(track.get("languageCode"))
                if lang and not lang.startswith(video.channel):
                    continue
                url = b.clean(track.get("baseUrl"))
                if not url:
                    continue
                sep = "&" if "?" in url else "?"
                rr = s.get(url + sep + "fmt=json3", timeout=18)
                rr.raise_for_status()
                try:
                    parts = b.find_segments(rr.json())
                except Exception:
                    parts = b.parse_caption_response(rr)
                if parts:
                    return parts, meta, lang or video.channel
            errors.append(f"{proxy}:no matching captions")
        except Exception as exc:
            errors.append(f"{proxy}:{type(exc).__name__}:{exc}")
    raise RuntimeError("; ".join(errors[-5:]) or "no working youtube proxy")


def invidious_data(video: b.Video) -> tuple[dict[str, Any], str]:
    errors = []
    for base in b.discover_invidious(video.video_id):
        try:
            r = b.session().get(f"{base}/api/v1/videos/{video.video_id}", timeout=20)
            r.raise_for_status()
            data = r.json()
            if data.get("title"):
                return data, base
        except Exception as exc:
            errors.append(f"{base}:{type(exc).__name__}:{exc}")
    raise RuntimeError("; ".join(errors[-5:]) or "no invidious data")


def invidious_meta(data: dict[str, Any]) -> b.Metadata:
    try:
        views = int(data.get("viewCount")) if data.get("viewCount") is not None else None
    except Exception:
        views = None
    return b.Metadata(
        b.clean(data.get("title")),
        views,
        b.normalize_date(data.get("published") or data.get("publishedText") or data.get("premiereTimestamp")),
    )


def invidious_captions(video: b.Video, data: dict[str, Any], base: str) -> tuple[list[b.Segment], str]:
    captions = data.get("captions") or []
    captions.sort(key=lambda x: (
        0 if b.clean(x.get("languageCode")) == video.channel else 1,
        0 if b.clean(x.get("languageCode")).startswith(video.channel) else 1,
    ))
    for cap in captions:
        lang = b.clean(cap.get("languageCode"))
        if lang and not lang.startswith(video.channel):
            continue
        url = urljoin(base + "/", b.clean(cap.get("url")))
        if not url:
            continue
        r = b.session().get(url, timeout=20)
        r.raise_for_status()
        parts = b.parse_caption_response(r)
        if parts:
            return parts, lang or video.channel
    return [], ""


def process(video: b.Video, initial: b.Metadata | None) -> b.Record:
    record = b.Record(video=video, title=video.source_title)
    if initial:
        b.merge_meta(record, initial)

    try:
        parts, meta, lang = b.fetch_notegpt(video)
        b.merge_meta(record, meta)
        if parts:
            record.segments = parts
            record.language = lang
            record.source = "notegpt"
    except Exception as exc:
        record.errors.append(f"notegpt:{type(exc).__name__}:{exc}")

    inv_data: dict[str, Any] | None = None
    inv_base = ""
    if not record.published or record.views is None or not record.segments:
        try:
            inv_data, inv_base = invidious_data(video)
            b.merge_meta(record, invidious_meta(inv_data))
            if not record.segments:
                parts, lang = invidious_captions(video, inv_data, inv_base)
                if parts:
                    record.segments = parts
                    record.language = lang
                    record.source = "invidious"
        except Exception as exc:
            record.errors.append(f"invidious:{type(exc).__name__}:{exc}")

    if not record.published or record.views is None or not record.segments:
        try:
            parts, meta, lang = youtube_via_proxy(video, not record.segments)
            b.merge_meta(record, meta)
            if parts:
                record.segments = parts
                record.language = lang
                record.source = "youtube-proxy"
        except Exception as exc:
            record.errors.append(f"youtube-proxy:{type(exc).__name__}:{exc}")

    if record.views is None:
        record.views = b.fallback_views(video.video_id)
    return record


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("transcript_job/videos.tsv"))
    p.add_argument("--output", type=Path, default=Path("gleb_solomin_transcripts"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    videos = b.read_videos(a.input)
    metadata = b.fetch_lemnos(videos)
    with cf.ThreadPoolExecutor(max_workers=2) as prep:
        prep.submit(b.discover_invidious, videos[0].video_id)
        prep.submit(discover_proxies, videos[0].video_id)
    records: list[b.Record] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = {pool.submit(process, v, metadata.get(v.video_id)): v for v in videos}
        for n, future in enumerate(cf.as_completed(futures), 1):
            v = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = b.Record(video=v, title=v.source_title, errors=[f"fatal:{type(exc).__name__}:{exc}"])
            records.append(record)
            print(f"{n}/{len(videos)} {v.channel}:{v.order} {v.video_id} {b.state(record)} t={bool(record.segments)} d={bool(record.published)}", flush=True)
    b.write_all(records, a.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
