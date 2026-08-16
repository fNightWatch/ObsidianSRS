from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import html
import json
import re
import shutil
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from youtube_transcript_api import YouTubeTranscriptApi

AS_OF = "2026-08-16"
TACTIQ_URL = "https://tactiq-apps-prod.tactiq.io/transcript"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/136 Safari/537.36"


@dataclass(frozen=True)
class Video:
    channel: str
    order: int
    video_id: str
    url: str
    source_title: str


@dataclass
class Segment:
    start: float
    duration: float
    text: str


@dataclass
class Record:
    video: Video
    title: str = ""
    views: int | None = None
    published: str = ""
    language: str = ""
    source: str = ""
    segments: list[Segment] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        status=1,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Cookie": "CONSENT=YES+cb",
    })
    return s


def clean(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text).strip()


def json_string(raw: str) -> str:
    try:
        return json.loads('"' + raw + '"')
    except Exception:
        return raw.replace("\\u0026", "&").replace("\\/", "/")


def dedupe(parts: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for item in sorted(parts, key=lambda x: x.start):
        item.text = clean(item.text)
        if not item.text:
            continue
        if out and item.text.casefold() == out[-1].text.casefold():
            continue
        out.append(item)
    return out


def parse_json3(data: dict[str, Any]) -> list[Segment]:
    parts: list[Segment] = []
    for event in data.get("events") or []:
        text = "".join(str(x.get("utf8") or "") for x in event.get("segs") or [])
        parts.append(Segment(
            float(event.get("tStartMs") or 0) / 1000,
            float(event.get("dDurationMs") or 0) / 1000,
            text,
        ))
    return dedupe(parts)


def parse_xml(text: str) -> list[Segment]:
    parts = []
    for m in re.finditer(r'<text\s+start="([\d.]+)"(?:\s+dur="([\d.]+)")?[^>]*>(.*?)</text>', text, re.S):
        parts.append(Segment(float(m.group(1)), float(m.group(2) or 0), clean(m.group(3))))
    return dedupe(parts)


def fetch_watch(s: requests.Session, video: Video) -> tuple[str, int | None, str, list[dict[str, str]]]:
    r = s.get(video.url, params={"hl": "en", "gl": "US", "bpctr": "9999999999"}, timeout=14)
    r.raise_for_status()
    text = r.text

    title = ""
    for pattern in (
        r'"videoDetails":\{"videoId":"[^"]+","title":"((?:\\.|[^"\\])*)"',
        r'<meta\s+name="title"\s+content="([^"]+)"',
        r'<title>(.*?)</title>',
    ):
        m = re.search(pattern, text, re.S)
        if m:
            title = clean(json_string(m.group(1)))
            title = re.sub(r"\s*-\s*YouTube\s*$", "", title)
            if title:
                break

    views = None
    for pattern in (r'"viewCount":"(\d+)"', r'itemprop="interactionCount"\s+content="(\d+)"'):
        m = re.search(pattern, text)
        if m:
            views = int(m.group(1))
            break

    published = ""
    for pattern in (
        r'"publishDate":"(\d{4}-\d{2}-\d{2})"',
        r'"uploadDate":"(\d{4}-\d{2}-\d{2})"',
        r'itemprop="datePublished"\s+content="(\d{4}-\d{2}-\d{2})"',
    ):
        m = re.search(pattern, text)
        if m:
            published = m.group(1)
            break

    tracks: list[dict[str, str]] = []
    for m in re.finditer(
        r'"baseUrl":"((?:\\.|[^"\\])*)"[^{}]{0,1500}?"languageCode":"((?:\\.|[^"\\])*)"',
        text,
        re.S,
    ):
        tracks.append({"url": json_string(m.group(1)), "lang": json_string(m.group(2))})
    return title, views, published, tracks


def oembed_title(s: requests.Session, video: Video) -> str:
    try:
        r = s.get(
            "https://www.youtube.com/oembed",
            params={"url": video.url, "format": "json"},
            timeout=8,
        )
        r.raise_for_status()
        return clean(r.json().get("title"))
    except Exception:
        return ""


def fallback_views(s: requests.Session, video_id: str) -> int | None:
    try:
        r = s.get("https://returnyoutubedislikeapi.com/votes", params={"videoId": video_id}, timeout=9)
        r.raise_for_status()
        value = r.json().get("viewCount")
        return int(value) if value is not None else None
    except Exception:
        return None


def fallback_date(s: requests.Session, video_id: str) -> str:
    for base in (
        "https://pipedapi.adminforge.de/streams/",
        "https://pipedapi.reallyaweso.me/streams/",
        "https://yewtu.be/api/v1/videos/",
        "https://inv.nadeko.net/api/v1/videos/",
    ):
        try:
            r = s.get(base + video_id, timeout=9)
            r.raise_for_status()
            data = r.json()
            value = clean(data.get("uploadDate"))
            m = re.search(r"\d{4}-\d{2}-\d{2}", value)
            if m:
                return m.group(0)
        except Exception:
            pass
    return ""


def tactiq(s: requests.Session, video: Video) -> tuple[list[Segment], str]:
    errors = []
    for code in ([video.channel, "ru-RU"] if video.channel == "ru" else [video.channel, "en-US"]):
        try:
            r = s.post(
                TACTIQ_URL,
                json={"langCode": code, "videoUrl": video.url},
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://tactiq.io",
                    "Referer": "https://tactiq.io/",
                },
                timeout=16,
            )
            r.raise_for_status()
            data = r.json()
            captions = data.get("captions") if isinstance(data, dict) else data
            parts = dedupe([
                Segment(
                    float(x.get("start") or x.get("offset") or 0),
                    float(x.get("dur") or x.get("duration") or 0),
                    x.get("text") or "",
                )
                for x in (captions or [])
                if isinstance(x, dict)
            ])
            if parts:
                return parts, code
            errors.append(f"{code}:empty")
        except Exception as exc:
            errors.append(f"{code}:{type(exc).__name__}:{exc}")
    raise RuntimeError("; ".join(errors))


def caption_track(s: requests.Session, tracks: list[dict[str, str]], language: str) -> tuple[list[Segment], str]:
    ordered = sorted(tracks, key=lambda t: (0 if t["lang"] == language else 1 if t["lang"].startswith(language) else 2))
    for track in ordered:
        if not track["lang"].startswith(language):
            continue
        for fmt in ("json3", ""):
            try:
                url = track["url"]
                if fmt:
                    sep = "&" if "?" in url else "?"
                    url += sep + "fmt=" + fmt
                r = s.get(url, timeout=14)
                r.raise_for_status()
                parts = parse_json3(r.json()) if fmt == "json3" or r.text.lstrip().startswith("{") else parse_xml(r.text)
                if parts:
                    return parts, track["lang"]
            except Exception:
                pass
    raise RuntimeError("no usable caption track")


def transcript_api(video: Video) -> tuple[list[Segment], str]:
    api = YouTubeTranscriptApi()
    codes = [video.channel] + (["ru-RU"] if video.channel == "ru" else ["en-US", "en-GB"])
    try:
        got = api.fetch(video.video_id, languages=codes)
        parts = dedupe([Segment(float(x.start), float(x.duration), x.text) for x in got])
        if parts:
            return parts, clean(getattr(got, "language_code", video.channel)) or video.channel
    except Exception:
        pass
    tracks = list(api.list(video.video_id))
    tracks.sort(key=lambda x: (
        0 if str(x.language_code) == video.channel else 1,
        0 if str(x.language_code).startswith(video.channel) else 1,
        0 if not x.is_generated else 1,
    ))
    for track in tracks:
        if not str(track.language_code).startswith(video.channel):
            continue
        got = track.fetch()
        parts = dedupe([Segment(float(x.start), float(x.duration), x.text) for x in got])
        if parts:
            return parts, str(track.language_code)
    raise RuntimeError("no matching transcript")


def process(video: Video) -> Record:
    s = session()
    record = Record(video=video, title=video.source_title)
    tracks: list[dict[str, str]] = []
    try:
        title, views, published, tracks = fetch_watch(s, video)
        record.title = title or record.title
        record.views = views
        record.published = published
    except Exception as exc:
        record.errors.append(f"watch:{type(exc).__name__}:{exc}")

    if not record.title:
        record.title = oembed_title(s, video)
    if record.views is None:
        record.views = fallback_views(s, video.video_id)
    if not record.published:
        record.published = fallback_date(s, video.video_id)

    for name, fn in (
        ("tactiq", lambda: tactiq(s, video)),
        ("youtube-caption-track", lambda: caption_track(s, tracks, video.channel)),
        ("youtube-transcript-api", lambda: transcript_api(video)),
    ):
        try:
            record.segments, record.language = fn()
            record.source = name
            break
        except Exception as exc:
            record.errors.append(f"{name}:{type(exc).__name__}:{exc}")
    return record


def read_videos(path: Path) -> list[Video]:
    items: list[Video] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            items.append(Video(
                clean(row["channel"]),
                int(row["order"]),
                clean(row["video_id"]),
                clean(row["url"]),
                clean(row["source_title"]),
            ))
    ids = [x.video_id for x in items]
    if len(items) != 147 or len(set(ids)) != 147:
        raise ValueError(f"video integrity failure: total={len(items)} unique={len(set(ids))}")
    return items


def stamp(seconds: float) -> str:
    n = max(0, int(seconds))
    return f"{n // 3600:02d}:{n % 3600 // 60:02d}:{n % 60:02d}"


def status(record: Record) -> str:
    return "ok" if record.title and record.views is not None and record.published and record.segments else "error"


def write_record(record: Record) -> str:
    lines = [
        f"channel: {record.video.channel}",
        f"order: {record.video.order}",
        f"video_id: {record.video.video_id}",
        f"url: {record.video.url}",
        f"title: {record.title}",
        f"views: {'' if record.views is None else record.views}",
        f"views_as_of: {AS_OF}",
        f"published: {record.published}",
        f"transcript_language: {record.language}",
        f"transcript_source: {record.source}",
        f"status: {status(record)}",
    ]
    if status(record) != "ok":
        lines.append("error: " + clean(" | ".join(record.errors)))
    lines.append("")
    lines.extend(f"[{stamp(x.start)}] {x.text}" for x in record.segments)
    return "\n".join(lines).rstrip() + "\n"


def write_all(records: list[Record], output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    (output / "ru").mkdir(parents=True)
    (output / "en").mkdir(parents=True)

    index = ["channel\torder\tvideo_id\turl\ttitle\tviews\tviews_as_of\tpublished\ttranscript_language\ttranscript_source\tstatus\tfile"]
    errors = ["channel\torder\tvideo_id\terror"]
    combined = {"ru": [], "en": []}

    for record in sorted(records, key=lambda x: (x.video.channel != "ru", x.video.order)):
        rel = f"{record.video.channel}/{record.video.order:03d}_{record.video.video_id}.txt"
        body = write_record(record)
        (output / rel).write_text(body, encoding="utf-8")
        combined[record.video.channel].append(body.rstrip())
        values = [
            record.video.channel,
            str(record.video.order),
            record.video.video_id,
            record.video.url,
            record.title,
            "" if record.views is None else str(record.views),
            AS_OF,
            record.published,
            record.language,
            record.source,
            status(record),
            rel,
        ]
        index.append("\t".join(clean(x) for x in values))
        if status(record) != "ok":
            errors.append("\t".join([
                record.video.channel,
                str(record.video.order),
                record.video.video_id,
                clean(" | ".join(record.errors)),
            ]))

    (output / "index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")
    (output / "errors.tsv").write_text("\n".join(errors) + "\n", encoding="utf-8")
    (output / "all_ru.txt").write_text("\n\n".join(combined["ru"]) + "\n", encoding="utf-8")
    (output / "all_en.txt").write_text("\n\n".join(combined["en"]) + "\n", encoding="utf-8")
    stats = {
        "total": len(records),
        "ru": sum(x.video.channel == "ru" for x in records),
        "en": sum(x.video.channel == "en" for x in records),
        "complete": sum(status(x) == "ok" for x in records),
        "transcripts": sum(bool(x.segments) for x in records),
        "metadata_complete": sum(bool(x.title and x.views is not None and x.published) for x in records),
        "errors": sum(status(x) != "ok" for x in records),
        "views_as_of": AS_OF,
    }
    (output / "stats.txt").write_text("\n".join(f"{k}: {v}" for k, v in stats.items()) + "\n", encoding="utf-8")
    shutil.copy2(__file__, output / "collect.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("transcript_job/videos.tsv"))
    parser.add_argument("--output", type=Path, default=Path("gleb_solomin_transcripts"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    videos = read_videos(args.input)
    records: list[Record] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process, video): video for video in videos}
        for n, future in enumerate(cf.as_completed(futures), 1):
            video = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = Record(video=video, title=video.source_title, errors=[f"fatal:{type(exc).__name__}:{exc}"])
            records.append(record)
            print(f"{n}/{len(videos)} {video.channel}:{video.order} {video.video_id} {status(record)}", flush=True)

    failed = [x for x in records if status(x) != "ok"]
    if failed:
        print(f"retrying {len(failed)} incomplete", flush=True)
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            retried = list(pool.map(lambda x: process(x.video), failed))
        replacements = {x.video.video_id: x for x in retried}
        records = [replacements.get(x.video.video_id, x) for x in records]

    write_all(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
