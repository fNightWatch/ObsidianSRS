#!/usr/bin/env python3
"""Obsidian SRS helper.

Scans markdown notes for review markers, tracks review history in JSON,
computes urgency score, and syncs color tags in note frontmatter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MARKER_RE = re.compile(r"@!\s*(10|[1-9])\b")
SRS_TAG_PREFIX = "srs/"
TAG_BANDS = [
    (0.92, "critical-red"),
    (0.80, "red"),
    (0.65, "orange"),
    (0.50, "yellow"),
    (0.35, "lime"),
    (0.00, "green"),
]


@dataclass
class ReviewEvent:
    ts: str
    difficulty: int


class SRSStore:
    def __init__(self, vault: Path) -> None:
        self.vault = vault
        self.store_path = vault / ".obsidian_srs" / "index.json"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {"notes": {}}
        self.load()

    def load(self) -> None:
        if self.store_path.exists():
            try:
                self.data = json.loads(self.store_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                backup = self.store_path.with_suffix(".broken.json")
                self.store_path.rename(backup)
                self.data = {"notes": {}}
                print(f"[warn] Corrupted index moved to {backup}")

    def save(self) -> None:
        self.store_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def add_review(self, rel_path: str, difficulty: int, now: datetime) -> None:
        notes = self.data.setdefault("notes", {})
        note = notes.setdefault(rel_path, {"reviews": []})
        note.setdefault("reviews", []).append({"ts": now.isoformat(), "difficulty": difficulty})

    def get_reviews(self, rel_path: str) -> list[ReviewEvent]:
        notes = self.data.get("notes", {})
        raw = notes.get(rel_path, {}).get("reviews", [])
        result: list[ReviewEvent] = []
        for entry in raw:
            try:
                result.append(ReviewEvent(ts=entry["ts"], difficulty=int(entry["difficulty"])))
            except (KeyError, ValueError, TypeError):
                continue
        return result


def iter_notes(vault: Path) -> list[Path]:
    return [
        p
        for p in vault.rglob("*.md")
        if ".git" not in p.parts and ".obsidian" not in p.parts and ".obsidian_srs" not in p.parts
    ]


def extract_marker_difficulty(line: str) -> int | None:
    match = MARKER_RE.search(line)
    if not match:
        return None
    return int(match.group(1))


def replace_first_marker(content: str, difficulty: int) -> tuple[str, bool]:
    def repl(match: re.Match[str]) -> str:
        return f"@@ {difficulty}"

    new_content, count = MARKER_RE.subn(repl, content, count=1)
    return new_content, count > 0


def parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_urgency(reviews: list[ReviewEvent], now: datetime) -> float:
    if not reviews:
        return 1.0

    reviews = sorted(reviews, key=lambda r: r.ts)
    last = reviews[-1]
    days_since = (now - parse_ts(last.ts)).total_seconds() / 86400
    avg_difficulty = sum(r.difficulty for r in reviews) / len(reviews)
    mastery_bonus = math.log2(len(reviews) + 1) / 5  # more repeats lower urgency

    retention_window = 1.0 + avg_difficulty * 1.8 + mastery_bonus * 7
    raw_pressure = days_since / retention_window
    difficulty_pressure = (avg_difficulty - 1) / 9

    urgency = clamp(0.58 * raw_pressure + 0.42 * difficulty_pressure, 0.0, 1.0)
    return urgency


def urgency_to_band(urgency: float) -> str:
    for threshold, band in TAG_BANDS:
        if urgency >= threshold:
            return band
    return "green"


def upsert_srs_tag(content: str, band: str) -> str:
    lines = content.splitlines()
    new_tag = f"{SRS_TAG_PREFIX}{band}"

    if lines and lines[0].strip() == "---":
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is not None:
            frontmatter = lines[1:end_idx]
            other_tags: list[str] = []
            tags_line_idx = None
            for idx, line in enumerate(frontmatter):
                if line.strip().startswith("tags:"):
                    tags_line_idx = idx
                    raw = line.split(":", 1)[1].strip()
                    if raw.startswith("[") and raw.endswith("]"):
                        payload = raw[1:-1].strip()
                        if payload:
                            other_tags = [t.strip() for t in payload.split(",") if t.strip()]
                    elif raw:
                        other_tags = [raw]
                    break

            filtered = [t for t in other_tags if not t.startswith(SRS_TAG_PREFIX)]
            filtered.append(new_tag)
            tag_text = "tags: [" + ", ".join(filtered) + "]"
            if tags_line_idx is None:
                frontmatter.append(tag_text)
            else:
                frontmatter[tags_line_idx] = tag_text
            return "\n".join(["---", *frontmatter, "---", *lines[end_idx + 1 :]]) + "\n"

    # no or invalid frontmatter -> prepend
    return "\n".join(["---", f"tags: [{new_tag}]", "---", *lines]) + "\n"


def scan_and_index(vault: Path, store: SRSStore, now: datetime, dry_run: bool = False) -> int:
    indexed = 0
    for note in iter_notes(vault):
        content = note.read_text(encoding="utf-8")
        difficulty = None
        for line in content.splitlines():
            difficulty = extract_marker_difficulty(line)
            if difficulty is not None:
                break
        if difficulty is None:
            continue

        rel = note.relative_to(vault).as_posix()
        store.add_review(rel, difficulty, now)
        new_content, replaced = replace_first_marker(content, difficulty)
        if replaced:
            urgency = compute_urgency(store.get_reviews(rel), now)
            band = urgency_to_band(urgency)
            new_content = upsert_srs_tag(new_content, band)
            if not dry_run:
                note.write_text(new_content, encoding="utf-8")
            indexed += 1
            print(f"[indexed] {rel} diff={difficulty} urgency={urgency:.2f} band={band}")
    if not dry_run:
        store.save()
    return indexed


def sync_tags(vault: Path, store: SRSStore, now: datetime, dry_run: bool = False) -> int:
    updated = 0
    for rel in sorted(store.data.get("notes", {}).keys()):
        note = vault / rel
        if not note.exists():
            continue
        reviews = store.get_reviews(rel)
        urgency = compute_urgency(reviews, now)
        band = urgency_to_band(urgency)
        content = note.read_text(encoding="utf-8")
        new_content = upsert_srs_tag(content, band)
        if new_content != content:
            if not dry_run:
                note.write_text(new_content, encoding="utf-8")
            updated += 1
            print(f"[sync] {rel} urgency={urgency:.2f} band={band}")
    if not dry_run:
        store.save()
    return updated


def add_manual_review(vault: Path, store: SRSStore, rel_path: str, difficulty: int, now: datetime) -> None:
    note = vault / rel_path
    if not note.exists():
        raise FileNotFoundError(f"Note not found: {rel_path}")
    store.add_review(rel_path, difficulty, now)
    content = note.read_text(encoding="utf-8")
    urgency = compute_urgency(store.get_reviews(rel_path), now)
    band = urgency_to_band(urgency)
    note.write_text(upsert_srs_tag(content, band), encoding="utf-8")
    store.save()
    print(f"[review] {rel_path} diff={difficulty} urgency={urgency:.2f} band={band}")


def report(vault: Path, store: SRSStore, now: datetime, limit: int = 30) -> None:
    rows = []
    for rel in store.data.get("notes", {}):
        note = vault / rel
        if not note.exists():
            continue
        reviews = store.get_reviews(rel)
        if not reviews:
            continue
        urgency = compute_urgency(reviews, now)
        last = sorted(reviews, key=lambda r: r.ts)[-1]
        rows.append((urgency, rel, len(reviews), last.difficulty, last.ts, urgency_to_band(urgency)))
    rows.sort(reverse=True)

    print("urgency | band         | reviews | last_diff | last_ts                  | note")
    print("-" * 96)
    for urgency, rel, count, last_diff, last_ts, band in rows[:limit]:
        print(f"{urgency:6.2f} | {band:<12} | {count:7d} | {last_diff:9d} | {last_ts:<24} | {rel}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Obsidian SRS for markdown vaults")
    p.add_argument("--vault", default=".", help="Path to Obsidian vault root")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Find @! <difficulty> markers, index and replace with @@")
    scan.add_argument("--dry-run", action="store_true")

    sync = sub.add_parser("sync", help="Recompute urgency and update SRS tags")
    sync.add_argument("--dry-run", action="store_true")

    rev = sub.add_parser("review", help="Add review manually")
    rev.add_argument("--file", required=True, help="Note path relative to vault")
    rev.add_argument("--difficulty", required=True, type=int, choices=range(1, 11))

    rep = sub.add_parser("report", help="Show prioritized notes")
    rep.add_argument("--limit", type=int, default=30)
    return p


def main() -> None:
    args = build_parser().parse_args()
    vault = Path(args.vault).resolve()
    now = datetime.now(timezone.utc)
    store = SRSStore(vault)

    if args.cmd == "scan":
        count = scan_and_index(vault, store, now, dry_run=args.dry_run)
        print(f"Indexed files: {count}")
    elif args.cmd == "sync":
        count = sync_tags(vault, store, now, dry_run=args.dry_run)
        print(f"Synced files: {count}")
    elif args.cmd == "review":
        add_manual_review(vault, store, args.file, args.difficulty, now)
    elif args.cmd == "report":
        report(vault, store, now, limit=args.limit)


if __name__ == "__main__":
    main()
