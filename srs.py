#!/usr/bin/env python3
"""Obsidian SRS helper.

Scans markdown notes for review markers, tracks review history in JSON,
computes urgency score, syncs color tags in note frontmatter, and keeps
Obsidian-facing markdown notes in the vault up to date.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any

MARKER_RE = re.compile(r"@!\s*(10|[1-9])\b")
WIKILINK_RE = re.compile(r"^\[\[(?P<body>.+?)\]\]$")
CHECKBOX_PREFIX_RE = re.compile(r"^\s*[-*+]\s+\[(?: |x|X)\]\s*")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
SRS_TAG_PREFIX = "srs/"
DEFAULT_REPORT_LIMIT = 30
QUEUE_NOTE_NAME = "_SRS Queue.md"
COMMANDS_NOTE_NAME = "_SRS Commands.md"
COMMANDS_REGION_START = "<!-- SRS-COMMANDS:START -->"
COMMANDS_REGION_END = "<!-- SRS-COMMANDS:END -->"
BRIDGE_NOTE_NAMES = {QUEUE_NOTE_NAME, COMMANDS_NOTE_NAME}
TAG_BANDS = [
    (0.92, "critical-red"),
    (0.80, "red"),
    (0.65, "orange"),
    (0.50, "yellow"),
    (0.35, "lime"),
    (0.00, "green"),
]
UPDATE_COMMANDS = {"обновить", "обновить всё", "обновить все", "update", "refresh", "refresh all"}
SCAN_COMMANDS = {"сканировать", "scan"}
SYNC_COMMANDS = {"синхронизировать", "sync"}
REPORT_COMMANDS = {"отчёт", "отчет", "report"}
QUEUE_COMMANDS = {"очередь", "queue", "export"}
FOCUS_LIMIT = 8
MINIMUM_TODAY_LIMIT = 5
SESSION_LIMIT = 12
UNTRACKED_LIMIT = 10
MessageSink = Callable[[str], None]


@dataclass(frozen=True)
class ReviewEvent:
    ts: str
    difficulty: int


@dataclass(frozen=True)
class ReportRow:
    urgency: float
    rel_path: str
    review_count: int
    last_difficulty: int
    last_ts: str
    band: str


@dataclass(frozen=True)
class CommandExecution:
    command: str
    ok: bool
    message: str

    def to_payload(self) -> dict[str, str]:
        return {
            "command": self.command,
            "status": "ok" if self.ok else "error",
            "message": self.message,
        }


@dataclass(frozen=True)
class QueueExportSummary:
    path: str
    row_count: int
    untracked_count: int


class SRSStore:
    def __init__(self, vault: Path) -> None:
        self.vault = vault
        self.store_path = vault / ".obsidian_srs" / "index.json"
        self.data: dict[str, Any] = {"notes": {}, "meta": {}}
        self.load()

    def load(self) -> None:
        if not self.store_path.exists():
            return

        try:
            raw_data = json.loads(self.store_path.read_text(encoding="utf-8"))
            if not isinstance(raw_data, dict):
                raise ValueError("Store root must be an object.")
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            backup = self._broken_backup_path()
            try:
                self.store_path.rename(backup)
            except OSError as backup_exc:
                print(f"[warn] Corrupted index could not be moved: {backup_exc}")
                backup = self.store_path
            self.data = {"notes": {}, "meta": {}}
            print(f"[warn] Corrupted index moved to {backup}: {exc}")
            return

        self.data = raw_data
        self.data.setdefault("notes", {})
        self.data.setdefault("meta", {})

    def _broken_backup_path(self) -> Path:
        backup = self.store_path.with_suffix(".broken.json")
        if not backup.exists():
            return backup

        for counter in range(1, 1000):
            candidate = self.store_path.with_name(f"{self.store_path.stem}.broken.{counter}.json")
            if not candidate.exists():
                return candidate
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return self.store_path.with_name(f"{self.store_path.stem}.broken.{timestamp}.json")

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def add_review(self, rel_path: str, difficulty: int, now: datetime) -> None:
        notes = self.data.setdefault("notes", {})
        note = notes.setdefault(rel_path, {"reviews": []})
        note.setdefault("reviews", []).append(
            {"ts": resolve_now(now).isoformat(), "difficulty": difficulty}
        )

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

    def set_meta_timestamp(self, key: str, now: datetime) -> None:
        self.set_meta_value(key, resolve_now(now).isoformat())

    def get_meta_timestamp(self, key: str) -> str | None:
        value = self.data.get("meta", {}).get(key)
        return value if isinstance(value, str) else None

    def set_meta_value(self, key: str, value: Any) -> None:
        self.data.setdefault("meta", {})[key] = value

    def get_meta_value(self, key: str, default: Any | None = None) -> Any:
        return self.data.get("meta", {}).get(key, default)

    def tracked_notes(self) -> int:
        return len(self.data.get("notes", {}))


def local_timezone() -> tzinfo:
    return datetime.now().astimezone().tzinfo or timezone.utc


def resolve_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=local_timezone())
    return now.astimezone(timezone.utc)


def parse_runtime(value: str) -> datetime:
    candidate = value.strip()
    if not candidate:
        raise argparse.ArgumentTypeError("Simulation time cannot be empty.")

    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use ISO time like '2026-04-01 09:30' or '2026-04-01T09:30:00+03:00'."
        ) from exc

    return resolve_now(parsed)


def format_runtime(
    ts: str | None,
    *,
    fallback: str = "Never",
    target_tz: tzinfo | None = None,
) -> str:
    if not ts:
        return fallback
    return parse_ts(ts).astimezone(target_tz or local_timezone()).strftime("%Y-%m-%d %H:%M:%S")


def resolve_vault(vault: Path | str) -> Path:
    candidate = Path(vault).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Vault not found: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {candidate}")
    return candidate


def emit_message(message: str, emit: MessageSink | None = None) -> None:
    if emit is None:
        print(message)
        return
    emit(message)


def queue_note_path(vault: Path) -> Path:
    return vault / QUEUE_NOTE_NAME


def commands_note_path(vault: Path) -> Path:
    return vault / COMMANDS_NOTE_NAME


def resolve_note_path_in_vault(vault: Path, rel_path: str) -> tuple[Path, str]:
    raw_path = rel_path.strip()
    if not raw_path:
        raise ValueError("Note path cannot be empty.")

    vault_root = vault.resolve()
    candidate = (vault_root / raw_path).resolve()
    try:
        normalized_rel = candidate.relative_to(vault_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Note path must stay inside the vault: {rel_path}") from exc

    if not normalized_rel or normalized_rel == ".":
        raise ValueError("Note path must point to a markdown file inside the vault.")
    return candidate, normalized_rel


def iter_notes(vault: Path) -> list[Path]:
    notes: list[Path] = []
    for path in vault.rglob("*.md"):
        rel = path.relative_to(vault)
        if ".git" in rel.parts or ".obsidian" in rel.parts or ".obsidian_srs" in rel.parts:
            continue
        if len(rel.parts) == 1 and rel.name in BRIDGE_NOTE_NAMES:
            continue
        notes.append(path)
    return notes


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
    now = resolve_now(now)
    if not reviews:
        return 1.0

    reviews = sorted(reviews, key=lambda r: r.ts)
    last = reviews[-1]
    days_since = (now - parse_ts(last.ts)).total_seconds() / 86400
    avg_difficulty = sum(r.difficulty for r in reviews) / len(reviews)
    mastery_bonus = math.log2(len(reviews) + 1) / 5

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


def urgency_recommendation(urgency: float) -> str:
    if urgency >= 0.92:
        return "Повторить прямо сейчас: заметка уже в зоне максимального риска." 
    if urgency >= 0.80:
        return "Лучше закрыть сегодня, пока не накопился хвост." 
    if urgency >= 0.65:
        return "Хороший кандидат на ближайшую учебную сессию." 
    if urgency >= 0.50:
        return "Можно повторить после красных и оранжевых заметок." 
    return "Состояние стабильное: заметка не требует срочного внимания." 


def _clean_tag_value(value: str) -> str:
    return value.strip().strip('"').strip("'").lstrip("#").strip()


def _split_inline_tags(payload: str) -> list[str]:
    tags: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape = False

    for char in payload:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\" and quote:
            escape = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            current.append(char)
            continue
        if char == "," and quote is None:
            tag = _clean_tag_value("".join(current))
            if tag:
                tags.append(tag)
            current = []
            continue
        current.append(char)

    tag = _clean_tag_value("".join(current))
    if tag:
        tags.append(tag)
    return tags


def _format_yaml_tag(tag: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_/-]+", tag):
        return tag
    return json.dumps(tag, ensure_ascii=False)


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
            tags_end_idx = None
            for idx, line in enumerate(frontmatter):
                if line == line.lstrip() and line.strip().startswith("tags:"):
                    tags_line_idx = idx
                    raw = line.split(":", 1)[1].strip()
                    if raw.startswith("[") and raw.endswith("]"):
                        other_tags = _split_inline_tags(raw[1:-1].strip())
                    elif raw:
                        other_tags = [_clean_tag_value(raw)]
                    else:
                        scan_idx = idx + 1
                        while scan_idx < len(frontmatter):
                            continuation = frontmatter[scan_idx]
                            if continuation and continuation == continuation.lstrip():
                                break
                            item = continuation.strip()
                            if item.startswith("- "):
                                tag = _clean_tag_value(item[2:])
                                if tag:
                                    other_tags.append(tag)
                            scan_idx += 1
                        tags_end_idx = scan_idx
                    if tags_end_idx is None:
                        tags_end_idx = idx + 1
                    break

            filtered = [t for t in other_tags if not _clean_tag_value(t).startswith(SRS_TAG_PREFIX)]
            filtered.append(new_tag)
            tag_text = "tags: [" + ", ".join(_format_yaml_tag(tag) for tag in filtered) + "]"
            if tags_line_idx is None:
                frontmatter.append(tag_text)
            else:
                frontmatter[tags_line_idx:tags_end_idx] = [tag_text]
            return "\n".join(["---", *frontmatter, "---", *lines[end_idx + 1 :]]) + "\n"

    return "\n".join(["---", f"tags: [{new_tag}]", "---", *lines]) + "\n"


def scan_and_index(
    vault: Path,
    store: SRSStore,
    now: datetime,
    dry_run: bool = False,
    emit: MessageSink | None = None,
) -> int:
    now = resolve_now(now)
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
        existing_reviews = store.get_reviews(rel)
        current_reviews = [*existing_reviews, ReviewEvent(ts=now.isoformat(), difficulty=difficulty)]
        new_content, replaced = replace_first_marker(content, difficulty)
        if not replaced:
            continue

        if not dry_run:
            store.add_review(rel, difficulty, now)

        urgency = compute_urgency(current_reviews, now)
        band = urgency_to_band(urgency)
        new_content = upsert_srs_tag(new_content, band)
        if not dry_run:
            note.write_text(new_content, encoding="utf-8")
        indexed += 1
        emit_message(f"[indexed] {rel} diff={difficulty} urgency={urgency:.2f} band={band}", emit)

    if not dry_run:
        store.set_meta_timestamp("last_scan_ts", now)
        store.set_meta_timestamp("last_action_ts", now)
        store.save()
    return indexed


def sync_tags(
    vault: Path,
    store: SRSStore,
    now: datetime,
    dry_run: bool = False,
    emit: MessageSink | None = None,
) -> int:
    now = resolve_now(now)
    updated = 0
    for rel in sorted(store.data.get("notes", {}).keys()):
        try:
            note, _normalized_rel = resolve_note_path_in_vault(vault, rel)
        except ValueError:
            continue
        if not note.exists():
            continue
        reviews = store.get_reviews(rel)
        urgency = compute_urgency(reviews, now)
        band = urgency_to_band(urgency)
        content = note.read_text(encoding="utf-8")
        new_content = upsert_srs_tag(content, band)
        if new_content == content:
            continue
        if not dry_run:
            note.write_text(new_content, encoding="utf-8")
        updated += 1
        emit_message(f"[sync] {rel} urgency={urgency:.2f} band={band}", emit)

    if not dry_run:
        store.set_meta_timestamp("last_sync_ts", now)
        store.set_meta_timestamp("last_action_ts", now)
        store.save()
    return updated


def add_manual_review(
    vault: Path,
    store: SRSStore,
    rel_path: str,
    difficulty: int,
    now: datetime,
    emit: MessageSink | None = None,
) -> None:
    now = resolve_now(now)
    note, normalized_rel = resolve_note_path_in_vault(vault, rel_path)
    if not note.exists():
        raise FileNotFoundError(f"Note not found: {rel_path}")
    if not note.is_file() or note.suffix.lower() != ".md":
        raise ValueError(f"Note path must point to a markdown file: {rel_path}")

    store.add_review(normalized_rel, difficulty, now)
    content = note.read_text(encoding="utf-8")
    urgency = compute_urgency(store.get_reviews(normalized_rel), now)
    band = urgency_to_band(urgency)
    note.write_text(upsert_srs_tag(content, band), encoding="utf-8")
    store.set_meta_timestamp("last_review_ts", now)
    store.set_meta_timestamp("last_action_ts", now)
    store.save()
    emit_message(f"[review] {normalized_rel} diff={difficulty} urgency={urgency:.2f} band={band}", emit)


def build_report_rows(
    vault: Path,
    store: SRSStore,
    now: datetime,
    limit: int | None = DEFAULT_REPORT_LIMIT,
) -> list[ReportRow]:
    now = resolve_now(now)
    rows: list[ReportRow] = []
    for rel in store.data.get("notes", {}):
        try:
            note, normalized_rel = resolve_note_path_in_vault(vault, rel)
        except ValueError:
            continue
        if not note.exists():
            continue
        reviews = store.get_reviews(rel)
        if not reviews:
            continue
        urgency = compute_urgency(reviews, now)
        last = sorted(reviews, key=lambda r: r.ts)[-1]
        rows.append(
            ReportRow(
                urgency=urgency,
                rel_path=normalized_rel,
                review_count=len(reviews),
                last_difficulty=last.difficulty,
                last_ts=last.ts,
                band=urgency_to_band(urgency),
            )
        )

    rows.sort(key=lambda row: row.urgency, reverse=True)
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def report(
    vault: Path,
    store: SRSStore,
    now: datetime,
    limit: int = DEFAULT_REPORT_LIMIT,
    emit: MessageSink | None = None,
) -> list[ReportRow]:
    rows = build_report_rows(vault, store, now, limit=limit)
    emit_message("urgency | band         | reviews | last_diff | last_ts                  | note", emit)
    emit_message("-" * 96, emit)
    for row in rows:
        emit_message(
            f"{row.urgency:6.2f} | {row.band:<12} | {row.review_count:7d} | "
            f"{row.last_difficulty:9d} | {row.last_ts:<24} | {row.rel_path}",
            emit,
        )
    return rows


def _days_since(ts: str, now: datetime) -> float:
    return max(0.0, (resolve_now(now) - parse_ts(ts)).total_seconds() / 86400)


def _humanize_age(ts: str, now: datetime) -> str:
    days = _days_since(ts, now)
    if days < 1 / 24:
        return "только что"
    if days < 1:
        return f"{max(1, round(days * 24))} ч. назад"
    if days < 7:
        return f"{days:.1f} дн. назад"
    return f"{days:.0f} дн. назад"


def _note_to_wikilink(rel_path: str) -> str:
    relative_no_suffix = Path(rel_path).with_suffix("").as_posix()
    alias = Path(rel_path).stem
    return f"[[{relative_no_suffix}|{alias}]]"


def _tracked_and_untracked_notes(vault: Path, rows: list[ReportRow]) -> tuple[set[str], list[str]]:
    tracked = {row.rel_path for row in rows}
    untracked = [
        note.relative_to(vault).as_posix()
        for note in iter_notes(vault)
        if note.relative_to(vault).as_posix() not in tracked
    ]
    return tracked, untracked


def _minimum_today(rows: list[ReportRow]) -> list[ReportRow]:
    urgent = [row for row in rows if row.urgency >= 0.80]
    if urgent:
        return urgent[:MINIMUM_TODAY_LIMIT]
    return rows[: min(MINIMUM_TODAY_LIMIT, len(rows))]


def _session_rows(rows: list[ReportRow]) -> list[ReportRow]:
    soon = [row for row in rows if row.urgency >= 0.65]
    if soon:
        return soon[:SESSION_LIMIT]
    return rows[: min(SESSION_LIMIT, len(rows))]


def _queue_header(now: datetime) -> list[str]:
    stamp = resolve_now(now).astimezone(local_timezone()).strftime("%Y-%m-%d %H:%M:%S")
    commands_stem = Path(COMMANDS_NOTE_NAME).stem
    return [
        "# SRS Queue",
        "",
        "> [!info] Автообновляемая очередь повторения",
        f"> Обновлено: {stamp}",
        f"> Команды пишите в [[{commands_stem}|{COMMANDS_NOTE_NAME}]] между маркерами: приложение их выполнит и очистит секцию.",
        "> Этот файл перезаписывается автоматически. Редактировать его вручную не нужно.",
        "",
    ]


def _format_row_brief(row: ReportRow, now: datetime, *, include_recommendation: bool) -> list[str]:
    parts = [
        f"- {_note_to_wikilink(row.rel_path)}",
        (
            "  - "
            f"Срочность: **{row.urgency:.2f}** · band: `{row.band}` · повторов: {row.review_count} "
            f"· последняя сложность: {row.last_difficulty}/10"
        ),
        (
            "  - "
            f"Последнее повторение: {format_runtime(row.last_ts, fallback='—')} "
            f"· {_humanize_age(row.last_ts, now)}"
        ),
    ]
    if include_recommendation:
        parts.append(f"  - Почему сейчас: {urgency_recommendation(row.urgency)}")
    return parts


def _format_command_results(store: SRSStore) -> list[str]:
    raw = store.get_meta_value("last_command_results", [])
    if not isinstance(raw, list) or not raw:
        return []

    lines = ["## Последняя обработка команд", ""]
    last_command_ts = store.get_meta_timestamp("last_command_ts")
    if last_command_ts:
        lines.append(f"_Выполнено: {format_runtime(last_command_ts, fallback='—')}_")
        lines.append("")

    for entry in raw[-20:]:
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command", "")).strip() or "(без команды)"
        status = str(entry.get("status", "error"))
        message = str(entry.get("message", "")).strip()
        icon = "✅" if status == "ok" else "⚠️"
        suffix = f" — {message}" if message else ""
        lines.append(f"- {icon} `{command}`{suffix}")
    lines.append("")
    return lines


def build_priority_queue_markdown(
    vault: Path,
    store: SRSStore,
    now: datetime,
    rows: list[ReportRow] | None = None,
) -> str:
    run_at = resolve_now(now)
    rows = build_report_rows(vault, store, run_at, limit=None) if rows is None else rows
    _, untracked = _tracked_and_untracked_notes(vault, rows)

    critical_count = sum(1 for row in rows if row.urgency >= 0.92)
    high_priority_count = sum(1 for row in rows if row.urgency >= 0.65)
    next_note = _note_to_wikilink(rows[0].rel_path) if rows else "—"
    avg_difficulty = (
        f"{sum(row.last_difficulty for row in rows) / len(rows):.1f}/10" if rows else "—"
    )

    lines = _queue_header(run_at)
    lines.extend(
        [
            "## Сводка",
            "",
            f"- Всего заметок в очереди: **{len(rows)}**",
            f"- Критично сейчас: **{critical_count}**",
            f"- Высокий приоритет: **{high_priority_count}**",
            f"- Средняя последняя сложность: **{avg_difficulty}**",
            f"- Следующая заметка: {next_note}",
            f"- Новые заметки без истории review: **{len(untracked)}**",
            "",
        ]
    )

    if rows:
        lines.extend(["## Минимум на сегодня", ""])
        for row in _minimum_today(rows):
            lines.extend(_format_row_brief(row, run_at, include_recommendation=True))
        lines.append("")

        lines.extend(["## Сессия на 30 минут", ""])
        for row in _session_rows(rows):
            lines.extend(_format_row_brief(row, run_at, include_recommendation=False))
        lines.append("")

        lines.extend(["## Сегодня в фокусе", ""])
        for row in rows[:FOCUS_LIMIT]:
            lines.extend(_format_row_brief(row, run_at, include_recommendation=False))
        lines.append("")

        lines.extend(["## Полная очередь по срочности", ""])
        for index, row in enumerate(rows, start=1):
            lines.append(f"{index}. {_note_to_wikilink(row.rel_path)}")
            lines.append(
                (
                    "   - "
                    f"Срочность: **{row.urgency:.2f}** · band: `{row.band}` · повторов: {row.review_count} "
                    f"· последняя сложность: {row.last_difficulty}/10"
                )
            )
            lines.append(
                (
                    "   - "
                    f"Последнее повторение: {format_runtime(row.last_ts, fallback='—')} "
                    f"· {_humanize_age(row.last_ts, run_at)}"
                )
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Очередь пока пуста",
                "",
                "- Ещё нет заметок с history review.",
                "- Добавьте ручное review или выполните сканирование маркеров `@!`, чтобы очередь появилась автоматически.",
                "",
            ]
        )

    if untracked:
        lines.extend(["## Новые заметки без истории", ""])
        for rel_path in untracked[:UNTRACKED_LIMIT]:
            lines.append(f"- {_note_to_wikilink(rel_path)} — пока без review-истории")
        if len(untracked) > UNTRACKED_LIMIT:
            lines.append(f"- …ещё {len(untracked) - UNTRACKED_LIMIT} заметок без истории")
        lines.append("")

    lines.extend(_format_command_results(store))
    return "\n".join(lines).rstrip() + "\n"


def ensure_commands_note(vault: Path) -> Path:
    note_path = commands_note_path(vault)
    if not note_path.exists():
        note_path.write_text(
            (
                "# SRS Commands\n\n"
                "> [!tip] Как пользоваться\n"
                "> Пишите по одной команде на строку только между маркерами ниже. "
                "Приложение выполнит команды и очистит секцию.\n\n"
                "## Поддерживаемые команды\n\n"
                "- `обновить` / `update` — сканировать маркеры, пересчитать теги и обновить очередь.\n"
                "- `сканировать` / `scan` — найти `@!` и записать review в индекс.\n"
                "- `синхронизировать` / `sync` — пересчитать band-теги по текущему индексу.\n"
                "- `отчёт` / `report` — перестроить очередь без изменения заметок.\n"
                "- `очередь` / `queue` — просто обновить markdown-очередь.\n"
                "- `ревью 7 [[folder/note]]` или `review [[folder/note]] 7` — добавить ручное review.\n\n"
                "## Inbox\n\n"
                f"{COMMANDS_REGION_START}\n\n"
                f"{COMMANDS_REGION_END}\n"
            ),
            encoding="utf-8",
        )
        return note_path

    content = note_path.read_text(encoding="utf-8")
    if COMMANDS_REGION_START in content and COMMANDS_REGION_END in content:
        return note_path

    append_block = (
        "\n\n## Inbox\n\n"
        f"{COMMANDS_REGION_START}\n\n"
        f"{COMMANDS_REGION_END}\n"
    )
    note_path.write_text(content.rstrip() + append_block, encoding="utf-8")
    return note_path


def _command_region_bounds(lines: list[str]) -> tuple[int, int]:
    start_idx: int | None = None
    end_idx: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == COMMANDS_REGION_START:
            start_idx = idx
            continue
        if stripped == COMMANDS_REGION_END and start_idx is not None:
            end_idx = idx
            break
    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise ValueError("Command note does not contain a valid SRS commands region.")
    return start_idx, end_idx


def _normalize_command_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    if text.startswith(("#", ">", "<!--", "%%", "```")):
        return None

    text = CHECKBOX_PREFIX_RE.sub("", text)
    text = LIST_PREFIX_RE.sub("", text).strip()
    if not text:
        return None
    return text.strip("`").strip()


def read_pending_commands(vault: Path) -> list[str]:
    note_path = ensure_commands_note(vault)
    lines = note_path.read_text(encoding="utf-8").splitlines()
    start_idx, end_idx = _command_region_bounds(lines)
    commands: list[str] = []
    for line in lines[start_idx + 1 : end_idx]:
        normalized = _normalize_command_line(line)
        if normalized:
            commands.append(normalized)
    return commands


def has_pending_commands(vault: Path) -> bool:
    return bool(read_pending_commands(vault))


def _strip_wikilink(reference: str) -> str:
    text = reference.strip().strip("`").strip()
    match = WIKILINK_RE.match(text)
    if match:
        text = match.group("body")
    if "|" in text:
        text = text.split("|", 1)[0]
    return text.strip().strip('"').strip("'").replace("\\", "/")


def resolve_note_reference(vault: Path, reference: str) -> str:
    candidate = _strip_wikilink(reference)
    if not candidate:
        raise ValueError("Empty note reference.")

    notes = [note.relative_to(vault).as_posix() for note in iter_notes(vault)]
    exact_lookup = {rel.casefold(): rel for rel in notes}

    probe_candidates = [candidate]
    if candidate.lower().endswith(".md"):
        probe_candidates.append(candidate[:-3])
    else:
        probe_candidates.append(f"{candidate}.md")

    for probe in probe_candidates:
        match = exact_lookup.get(probe.casefold())
        if match:
            return match

    no_suffix_lookup = {Path(rel).with_suffix("").as_posix().casefold(): rel for rel in notes}
    match = no_suffix_lookup.get(candidate.casefold())
    if match:
        return match

    basename = Path(candidate).name.casefold()
    basename_matches = [rel for rel in notes if Path(rel).stem.casefold() == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        variants = ", ".join(sorted(basename_matches)[:5])
        raise ValueError(f"Ambiguous note reference '{reference}'. Use a relative path, for example: {variants}")

    raise FileNotFoundError(f"Note not found: {reference}")


def _parse_review_command(command: str) -> tuple[int, str] | None:
    stripped = command.strip()
    lowered = stripped.casefold()
    prefixes = ("review", "ревью")
    prefix = next((item for item in prefixes if lowered.startswith(item.casefold())), None)
    if prefix is None:
        return None

    remainder = stripped[len(prefix) :].strip()
    if not remainder:
        raise ValueError("Use review command like 'review 7 [[folder/note]]'.")

    leading = re.match(r"^(10|[1-9])\s+(.+)$", remainder)
    if leading:
        return int(leading.group(1)), leading.group(2).strip()

    trailing = re.match(r"^(.+?)\s+(10|[1-9])$", remainder)
    if trailing:
        return int(trailing.group(2)), trailing.group(1).strip()

    raise ValueError("Use review command like 'review 7 [[folder/note]]' or 'review [[folder/note]] 7'.")


def execute_obsidian_command(
    vault: Path,
    store: SRSStore,
    now: datetime,
    command: str,
    emit: MessageSink | None = None,
) -> CommandExecution:
    normalized = command.strip()
    lowered = normalized.casefold()
    emit_message(f"[commands] -> {normalized}", emit)

    if lowered in UPDATE_COMMANDS:
        scan_count = scan_and_index(vault, store, now, emit=emit)
        sync_count = sync_tags(vault, store, now, emit=emit)
        return CommandExecution(
            command=normalized,
            ok=True,
            message=f"обновление завершено: scan={scan_count}, sync={sync_count}",
        )

    if lowered in SCAN_COMMANDS:
        count = scan_and_index(vault, store, now, emit=emit)
        return CommandExecution(command=normalized, ok=True, message=f"просканировано файлов: {count}")

    if lowered in SYNC_COMMANDS:
        count = sync_tags(vault, store, now, emit=emit)
        return CommandExecution(command=normalized, ok=True, message=f"обновлено тегов: {count}")

    if lowered in REPORT_COMMANDS:
        rows = build_report_rows(vault, store, now, limit=None)
        emit_message(f"[report] rebuilt {len(rows)} rows from command file", emit)
        return CommandExecution(command=normalized, ok=True, message=f"отчёт перестроен: {len(rows)} заметок")

    if lowered in QUEUE_COMMANDS:
        return CommandExecution(command=normalized, ok=True, message="очередь будет обновлена")

    review_payload = _parse_review_command(normalized)
    if review_payload is not None:
        difficulty, note_ref = review_payload
        rel_path = resolve_note_reference(vault, note_ref)
        add_manual_review(vault, store, rel_path, difficulty, now, emit=emit)
        return CommandExecution(
            command=normalized,
            ok=True,
            message=f"добавлено review для {rel_path} со сложностью {difficulty}",
        )

    raise ValueError(
        "Unknown command. Supported: обновить, scan, sync, report, queue, review 7 [[folder/note]]."
    )


def export_priority_queue(
    vault: Path,
    store: SRSStore,
    now: datetime,
    rows: list[ReportRow] | None = None,
    emit: MessageSink | None = None,
) -> QueueExportSummary:
    run_at = resolve_now(now)
    rows = build_report_rows(vault, store, run_at, limit=None) if rows is None else rows
    queue_path = queue_note_path(vault)
    markdown = build_priority_queue_markdown(vault, store, run_at, rows=rows)
    queue_path.write_text(markdown, encoding="utf-8")

    _, untracked = _tracked_and_untracked_notes(vault, rows)
    store.set_meta_timestamp("last_queue_export_ts", run_at)
    store.save()
    emit_message(f"[queue] updated {queue_path.name} with {len(rows)} items", emit)
    return QueueExportSummary(
        path=queue_path.relative_to(vault).as_posix(),
        row_count=len(rows),
        untracked_count=len(untracked),
    )


def sync_obsidian_bridge(
    vault: Path,
    store: SRSStore,
    now: datetime,
    rows: list[ReportRow] | None = None,
    emit: MessageSink | None = None,
) -> tuple[list[ReportRow], QueueExportSummary]:
    ensure_commands_note(vault)
    resolved_rows = build_report_rows(vault, store, now, limit=None) if rows is None else rows
    summary = export_priority_queue(vault, store, now, rows=resolved_rows, emit=emit)
    return resolved_rows, summary


def process_command_note(
    vault: Path,
    store: SRSStore,
    now: datetime,
    emit: MessageSink | None = None,
) -> tuple[list[CommandExecution], list[ReportRow], QueueExportSummary]:
    run_at = resolve_now(now)
    note_path = ensure_commands_note(vault)
    lines = note_path.read_text(encoding="utf-8").splitlines()
    start_idx, end_idx = _command_region_bounds(lines)
    commands: list[str] = []
    for line in lines[start_idx + 1 : end_idx]:
        normalized = _normalize_command_line(line)
        if normalized:
            commands.append(normalized)

    results: list[CommandExecution] = []
    if commands:
        emit_message(f"[commands] found {len(commands)} pending command(s)", emit)
        for command in commands:
            try:
                result = execute_obsidian_command(vault, store, run_at, command, emit=emit)
            except Exception as exc:
                result = CommandExecution(command=command, ok=False, message=str(exc))
                emit_message(f"[commands:error] {command}: {exc}", emit)
            results.append(result)

        failed_commands = [result.command for result in results if not result.ok]
        inbox_lines = ["", *failed_commands, ""] if failed_commands else [""]
        new_lines = lines[: start_idx + 1] + inbox_lines + lines[end_idx:]
        note_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
        store.set_meta_timestamp("last_command_ts", run_at)
        store.set_meta_timestamp("last_action_ts", run_at)
        store.set_meta_value("last_command_results", [item.to_payload() for item in results])
        store.save()

    rows, queue_summary = sync_obsidian_bridge(vault, store, run_at, emit=emit)
    return results, rows, queue_summary


def update_workspace(
    vault: Path,
    store: SRSStore,
    now: datetime,
    emit: MessageSink | None = None,
) -> tuple[dict[str, int], list[ReportRow], QueueExportSummary]:
    run_at = resolve_now(now)
    scan_count = scan_and_index(vault, store, run_at, emit=emit)
    sync_count = sync_tags(vault, store, run_at, emit=emit)
    rows, queue_summary = sync_obsidian_bridge(vault, store, run_at, emit=emit)
    return {"scan_count": scan_count, "sync_count": sync_count}, rows, queue_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Obsidian SRS for markdown vaults")
    p.add_argument("--vault", default=".", help="Path to Obsidian vault root")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_time_argument(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--at",
            type=parse_runtime,
            help="Optional simulation time in ISO format, e.g. '2026-04-01 09:30'.",
        )

    scan = sub.add_parser("scan", help="Find @! <difficulty> markers, index and replace with @@")
    scan.add_argument("--dry-run", action="store_true")
    add_time_argument(scan)

    sync = sub.add_parser("sync", help="Recompute urgency and update SRS tags")
    sync.add_argument("--dry-run", action="store_true")
    add_time_argument(sync)

    rev = sub.add_parser("review", help="Add review manually")
    rev.add_argument("--file", required=True, help="Note path relative to vault")
    rev.add_argument("--difficulty", required=True, type=int, choices=range(1, 11))
    add_time_argument(rev)

    rep = sub.add_parser("report", help="Show prioritized notes")
    rep.add_argument("--limit", type=int, default=DEFAULT_REPORT_LIMIT)
    add_time_argument(rep)

    queue_cmd = sub.add_parser("queue", help="Regenerate queue and command notes inside the vault")
    add_time_argument(queue_cmd)

    update_cmd = sub.add_parser(
        "update",
        help="Scan markers, sync tags, and regenerate queue and command notes inside the vault",
    )
    add_time_argument(update_cmd)

    commands_cmd = sub.add_parser("commands", help="Process commands from the Obsidian command note")
    add_time_argument(commands_cmd)
    return p


def main() -> None:
    args = build_parser().parse_args()
    vault = resolve_vault(args.vault)
    now = resolve_now(getattr(args, "at", None))
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
    elif args.cmd == "queue":
        _rows, queue_summary = sync_obsidian_bridge(vault, store, now)
        print(f"Queue updated: {queue_summary.path} ({queue_summary.row_count} notes)")
    elif args.cmd == "update":
        counts, _rows, queue_summary = update_workspace(vault, store, now)
        print(
            "Update complete: "
            f"scan={counts['scan_count']}, sync={counts['sync_count']}, "
            f"queue={queue_summary.path}"
        )
    elif args.cmd == "commands":
        results, _rows, queue_summary = process_command_note(vault, store, now)
        success_count = sum(1 for item in results if item.ok)
        print(
            "Processed commands: "
            f"total={len(results)}, ok={success_count}, queue={queue_summary.path}"
        )


if __name__ == "__main__":
    main()
