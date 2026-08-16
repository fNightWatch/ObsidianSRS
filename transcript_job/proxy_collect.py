from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import html
import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

AS_OF = "2026-08-16"
NOTEGPT = "https://notegpt.io/api/v2/video-transcript"
LEMNOS = "https://yt.lemnoslife.com/noKey/videos"
UA = "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Mobile Safari/537.36"

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

@dataclass
class Metadata:
    title: str = ""
    views: int | None = None
    published: str = ""

def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, connect=2, read=2, status=2, backoff_factor=.4,
                  status_forcelist=(408, 429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}))
    a = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", a)
    s.headers.update({"User-Agent": UA, "Accept": "application/json,text/plain,*/*"})
    return s

def clean(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text).strip()

def seconds(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        x = float(value)
        return x / 1000 if x > 100000 else x
    text = str(value).strip().replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        x = float(text)
        return x / 1000 if x > 100000 else x
    m = re.search(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)", text)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return 0.0

def dedupe(parts: Iterable[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for item in sorted(parts, key=lambda x: x.start):
        item.text = clean(item.text)
        if not item.text:
            continue
        if out and item.text.casefold() == out[-1].text.casefold():
            continue
        out.append(item)
    return out

TEXT_KEYS = ("text", "caption", "content", "sentence", "line", "value", "transcript")
START_KEYS = ("start", "startTime", "start_time", "offset", "timestamp", "time", "tStartMs")
DURATION_KEYS = ("duration", "dur", "length", "dDurationMs")
END_KEYS = ("end", "endTime", "end_time")

def dict_segment(obj: dict[str, Any]) -> Segment | None:
    text = ""
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and clean(value):
            text = value
            break
    if not text and isinstance(obj.get("segs"), list):
        text = "".join(str(x.get("utf8") or "") for x in obj["segs"] if isinstance(x, dict))
    if not clean(text):
        return None
    start = next((seconds(obj[k]) for k in START_KEYS if k in obj), 0.0)
    duration = next((seconds(obj[k]) for k in DURATION_KEYS if k in obj), 0.0)
    if not duration:
        end = next((seconds(obj[k]) for k in END_KEYS if k in obj), 0.0)
        duration = max(0.0, end - start)
    return Segment(start, duration, text)

def parse_timestamped_text(text: str) -> list[Segment]:
    parts: list[Segment] = []
    for line in text.replace("\r", "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[?((?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?)\]?\s*[-–—:]?\s*(.+)$", line)
        if m:
            parts.append(Segment(seconds(m.group(1)), 0, m.group(2)))
    if parts:
        return dedupe(parts)
    text = clean(text)
    return [Segment(0, 0, text)] if len(text) > 40 else []

def find_segments(obj: Any, depth: int = 0) -> list[Segment]:
    if depth > 9:
        return []
    if isinstance(obj, list):
        direct = [x for x in (dict_segment(v) for v in obj if isinstance(v, dict)) if x]
        if direct and sum(len(x.text) for x in direct) > 80:
            return dedupe(direct)
        strings = [clean(v) for v in obj if isinstance(v, str) and clean(v)]
        if strings and sum(map(len, strings)) > 80:
            return [Segment(float(i), 0, x) for i, x in enumerate(strings)]
        for value in obj:
            found = find_segments(value, depth + 1)
            if found:
                return found
    elif isinstance(obj, dict):
        for key in ("transcript", "captions", "subtitles", "segments", "sentences", "content", "data", "result"):
            if key in obj:
                found = find_segments(obj[key], depth + 1)
                if found:
                    return found
        direct = dict_segment(obj)
        if direct and len(direct.text) > 80:
            return parse_timestamped_text(direct.text)
        for value in obj.values():
            found = find_segments(value, depth + 1)
            if found:
                return found
    elif isinstance(obj, str):
        return parse_timestamped_text(obj)
    return []

def recursive_values(obj: Any, keys: set[str], depth: int = 0) -> list[Any]:
    if depth > 8:
        return []
    out: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in keys:
                out.append(value)
            out.extend(recursive_values(value, keys, depth + 1))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(recursive_values(value, keys, depth + 1))
    return out

def normalize_date(value: Any) -> str:
    text = clean(value)
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if m:
        return m.group(0)
    try:
        x = float(value)
        if x > 10_000_000_000:
            x /= 1000
        if x > 1_000_000_000:
            return datetime.fromtimestamp(x, timezone.utc).date().isoformat()
    except Exception:
        pass
    return ""

def metadata_from_object(obj: Any) -> Metadata:
    titles = recursive_values(obj, {"title", "videotitle", "video_title"})
    dates = recursive_values(obj, {"publisheddate", "publishdate", "publishedat", "uploaddate", "published"})
    views = recursive_values(obj, {"viewcount", "views", "view_count"})
    title = next((clean(x) for x in titles if isinstance(x, str) and clean(x)), "")
    published = next((normalize_date(x) for x in dates if normalize_date(x)), "")
    count = None
    for value in views:
        digits = re.sub(r"\D", "", str(value))
        if digits:
            count = int(digits)
            break
    return Metadata(title, count, published)

def fetch_notegpt(video: Video) -> tuple[list[Segment], Metadata, str]:
    r = session().get(NOTEGPT,
                      params={"platform": "youtube", "video_id": video.video_id},
                      headers={"Referer": "https://notegpt.io/youtube-transcript-generator", "Origin": "https://notegpt.io"},
                      timeout=35)
    r.raise_for_status()
    obj = r.json()
    parts = find_segments(obj)
    if not parts:
        raise RuntimeError("no transcript in response: " + clean(r.text)[:300])
    return parts, metadata_from_object(obj), video.channel

def fetch_lemnos(videos: list[Video]) -> dict[str, Metadata]:
    out: dict[str, Metadata] = {}
    s = session()
    for start in range(0, len(videos), 50):
        chunk = videos[start:start+50]
        try:
            r = s.get(LEMNOS, params={"part": "snippet,statistics", "id": ",".join(x.video_id for x in chunk)}, timeout=35)
            r.raise_for_status()
            for item in (r.json().get("items") or []):
                vid = clean(item.get("id")); snippet = item.get("snippet") or {}; stats = item.get("statistics") or {}
                title = clean(snippet.get("title")); published = normalize_date(snippet.get("publishedAt"))
                try: views = int(stats.get("viewCount")) if stats.get("viewCount") is not None else None
                except Exception: views = None
                if vid: out[vid] = Metadata(title, views, published)
        except Exception as exc:
            print(f"lemnos chunk {start}: {type(exc).__name__}:{exc}", flush=True)
    return out

_instances_lock = threading.Lock()
_instances: list[str] | None = None

def discover_invidious(sample_id: str) -> list[str]:
    global _instances
    with _instances_lock:
        if _instances is not None:
            return _instances
        candidates: list[str] = []
        try:
            data = session().get("https://api.invidious.io/instances.json", timeout=25).json()
            for host, details in data:
                if not isinstance(details, dict) or not details.get("api"): continue
                uri = clean(details.get("uri")) or ("https://" + host)
                if uri.startswith("https://") and ".onion" not in uri: candidates.append(uri.rstrip("/"))
        except Exception as exc:
            print(f"invidious list: {type(exc).__name__}:{exc}", flush=True)
        candidates += ["https://yewtu.be", "https://inv.nadeko.net", "https://invidious.nerdvpn.de", "https://invidious.privacyredirect.com", "https://inv.us.projectsegfau.lt"]
        candidates = list(dict.fromkeys(candidates))[:40]
        def test(base: str) -> str | None:
            try:
                r = session().get(f"{base}/api/v1/videos/{sample_id}", timeout=12); r.raise_for_status()
                return base if r.json().get("title") else None
            except Exception: return None
        good: list[str] = []
        with cf.ThreadPoolExecutor(max_workers=16) as pool:
            for value in pool.map(test, candidates):
                if value:
                    good.append(value)
                    if len(good) >= 8: break
        _instances = good
        print(f"invidious usable: {len(good)}", flush=True)
        return good

def parse_caption_response(r: requests.Response) -> list[Segment]:
    text = r.text
    if text.lstrip().startswith("{"):
        return find_segments(r.json())
    parts = []
    for m in re.finditer(r'<text\s+start="([\d.]+)"(?:\s+dur="([\d.]+)")?[^>]*>(.*?)</text>', text, re.S):
        parts.append(Segment(float(m.group(1)), float(m.group(2) or 0), m.group(3)))
    return dedupe(parts)

def fetch_invidious(video: Video) -> tuple[list[Segment], Metadata, str]:
    errors = []
    for base in discover_invidious(video.video_id):
        try:
            s = session(); r = s.get(f"{base}/api/v1/videos/{video.video_id}", timeout=20); r.raise_for_status(); data = r.json()
            meta = Metadata(clean(data.get("title")), int(data["viewCount"]) if data.get("viewCount") is not None else None,
                            normalize_date(data.get("published") or data.get("publishedText")))
            captions = data.get("captions") or []
            captions.sort(key=lambda x: (0 if clean(x.get("languageCode")) == video.channel else 1,
                                         0 if clean(x.get("languageCode")).startswith(video.channel) else 1))
            for cap in captions:
                lang = clean(cap.get("languageCode"))
                if lang and not lang.startswith(video.channel): continue
                url = urljoin(base + "/", clean(cap.get("url")))
                if not url: continue
                rr = s.get(url, timeout=20); rr.raise_for_status(); parts = parse_caption_response(rr)
                if parts: return parts, meta, lang or video.channel
            errors.append(f"{base}:no captions")
        except Exception as exc:
            errors.append(f"{base}:{type(exc).__name__}:{exc}")
    raise RuntimeError("; ".join(errors[-5:]) or "no invidious instance")

def fallback_views(video_id: str) -> int | None:
    try:
        r = session().get("https://returnyoutubedislikeapi.com/votes", params={"videoId": video_id}, timeout=20); r.raise_for_status()
        value = r.json().get("viewCount")
        return int(value) if value is not None else None
    except Exception: return None

def merge_meta(record: Record, meta: Metadata) -> None:
    record.title = meta.title or record.title
    record.views = meta.views if meta.views is not None else record.views
    record.published = meta.published or record.published

def process(video: Video, initial: Metadata | None) -> Record:
    r = Record(video=video, title=video.source_title)
    if initial: merge_meta(r, initial)
    for name, fn in (("notegpt", fetch_notegpt), ("invidious", fetch_invidious)):
        try:
            parts, meta, lang = fn(video); merge_meta(r, meta)
            if parts:
                r.segments = parts; r.language = lang; r.source = name; break
        except Exception as exc:
            r.errors.append(f"{name}:{type(exc).__name__}:{exc}")
    if r.views is None: r.views = fallback_views(video.video_id)
    return r

def read_videos(path: Path) -> list[Video]:
    items = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            items.append(Video(clean(row["channel"]), int(row["order"]), clean(row["video_id"]), clean(row["url"]), clean(row["source_title"])))
    ids = [x.video_id for x in items]
    if len(items) != 147 or len(set(ids)) != 147: raise ValueError(f"video integrity failure: {len(items)} / {len(set(ids))}")
    return items

def stamp(value: float) -> str:
    n = max(0, int(value)); return f"{n//3600:02d}:{n%3600//60:02d}:{n%60:02d}"

def state(r: Record) -> str:
    return "ok" if r.title and r.views is not None and r.published and r.segments else "error"

def render(r: Record) -> str:
    lines = [f"channel: {r.video.channel}", f"order: {r.video.order}", f"video_id: {r.video.video_id}", f"url: {r.video.url}",
             f"title: {r.title}", f"views: {'' if r.views is None else r.views}", f"views_as_of: {AS_OF}", f"published: {r.published}",
             f"transcript_language: {r.language}", f"transcript_source: {r.source}", f"status: {state(r)}"]
    if state(r) != "ok":
        missing = [k for k, ok in (("title", r.title), ("views", r.views is not None), ("published", r.published), ("transcript", r.segments)) if not ok]
        lines += ["missing: " + ",".join(missing), "error: " + clean(" | ".join(r.errors))]
    lines.append("")
    lines.extend(f"[{stamp(x.start)}] {x.text}" for x in r.segments)
    return "\n".join(lines).rstrip() + "\n"

def write_all(records: list[Record], output: Path) -> None:
    if output.exists(): shutil.rmtree(output)
    (output / "ru").mkdir(parents=True); (output / "en").mkdir(parents=True)
    index = ["channel\torder\tvideo_id\turl\ttitle\tviews\tviews_as_of\tpublished\ttranscript_language\ttranscript_source\tstatus\tfile"]
    errors = ["channel\torder\tvideo_id\tmissing\terror"]; combined = {"ru": [], "en": []}
    for r in sorted(records, key=lambda x: (x.video.channel != "ru", x.video.order)):
        rel = f"{r.video.channel}/{r.video.order:03d}_{r.video.video_id}.txt"; body = render(r)
        (output / rel).write_text(body, encoding="utf-8"); combined[r.video.channel].append(body.rstrip())
        values = [r.video.channel, str(r.video.order), r.video.video_id, r.video.url, r.title, "" if r.views is None else str(r.views), AS_OF,
                  r.published, r.language, r.source, state(r), rel]
        index.append("\t".join(clean(x) for x in values))
        if state(r) != "ok":
            missing = ",".join(k for k, ok in (("title", r.title), ("views", r.views is not None), ("published", r.published), ("transcript", r.segments)) if not ok)
            errors.append("\t".join([r.video.channel, str(r.video.order), r.video.video_id, missing, clean(" | ".join(r.errors))]))
    (output / "index.tsv").write_text("\n".join(index) + "\n", encoding="utf-8")
    (output / "errors.tsv").write_text("\n".join(errors) + "\n", encoding="utf-8")
    (output / "all_ru.txt").write_text("\n\n".join(combined["ru"]) + "\n", encoding="utf-8")
    (output / "all_en.txt").write_text("\n\n".join(combined["en"]) + "\n", encoding="utf-8")
    stats = {"total": len(records), "ru": sum(x.video.channel == "ru" for x in records), "en": sum(x.video.channel == "en" for x in records),
             "complete": sum(state(x) == "ok" for x in records), "transcripts": sum(bool(x.segments) for x in records),
             "titles": sum(bool(x.title) for x in records), "views": sum(x.views is not None for x in records),
             "published": sum(bool(x.published) for x in records), "errors": sum(state(x) != "ok" for x in records), "views_as_of": AS_OF}
    (output / "stats.txt").write_text("\n".join(f"{k}: {v}" for k, v in stats.items()) + "\n", encoding="utf-8")
    shutil.copy2(__file__, output / "collect.py")

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input", type=Path, default=Path("transcript_job/videos.tsv")); p.add_argument("--output", type=Path, default=Path("gleb_solomin_transcripts")); p.add_argument("--workers", type=int, default=10); a = p.parse_args()
    videos = read_videos(a.input); metadata = fetch_lemnos(videos); discover_invidious(videos[0].video_id); records: list[Record] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = {pool.submit(process, v, metadata.get(v.video_id)): v for v in videos}
        for n, future in enumerate(cf.as_completed(futures), 1):
            v = futures[future]
            try: r = future.result()
            except Exception as exc: r = Record(video=v, title=v.source_title, errors=[f"fatal:{type(exc).__name__}:{exc}"])
            records.append(r); print(f"{n}/{len(videos)} {v.channel}:{v.order} {v.video_id} {state(r)} t={bool(r.segments)} d={bool(r.published)}", flush=True)
    write_all(records, a.output); return 0

if __name__ == "__main__": raise SystemExit(main())
