from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from srs import (
    COMMANDS_NOTE_NAME,
    QUEUE_NOTE_NAME,
    SRSStore,
    add_manual_review,
    build_report_rows,
    has_pending_commands,
    iter_notes,
    parse_runtime,
    process_command_note,
    scan_and_index,
    sync_obsidian_bridge,
    sync_tags,
    upsert_srs_tag,
)


class SRSTestCase(unittest.TestCase):
    def test_scan_uses_simulation_time_for_reviews_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            note = vault / "note.md"
            note.write_text("# Demo\n\n@! 7\n", encoding="utf-8")

            simulated = datetime(2030, 1, 2, 12, 30, tzinfo=timezone.utc)
            count = scan_and_index(vault, SRSStore(vault), simulated, emit=lambda _message: None)

            self.assertEqual(count, 1)

            refreshed = SRSStore(vault)
            reviews = refreshed.get_reviews("note.md")
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews[0].ts, simulated.isoformat())
            self.assertEqual(refreshed.get_meta_timestamp("last_scan_ts"), simulated.isoformat())

            content = note.read_text(encoding="utf-8")
            self.assertIn("@@ 7", content)
            self.assertIn("srs/", content)

    def test_sync_updates_last_sync_with_custom_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            note = vault / "note.md"
            note.write_text("# Demo\n", encoding="utf-8")

            review_time = datetime(2030, 1, 1, 8, 0, tzinfo=timezone.utc)
            add_manual_review(vault, SRSStore(vault), "note.md", 4, review_time, emit=lambda _message: None)

            sync_time = datetime(2030, 1, 4, 18, 45, tzinfo=timezone.utc)
            updated = sync_tags(vault, SRSStore(vault), sync_time, emit=lambda _message: None)

            self.assertGreaterEqual(updated, 0)

            refreshed = SRSStore(vault)
            self.assertEqual(refreshed.get_meta_timestamp("last_sync_ts"), sync_time.isoformat())

    def test_build_report_rows_without_limit_returns_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            (vault / "a.md").write_text("# A\n", encoding="utf-8")
            (vault / "b.md").write_text("# B\n", encoding="utf-8")

            first_time = datetime(2030, 1, 1, 8, 0, tzinfo=timezone.utc)
            second_time = datetime(2030, 1, 2, 8, 0, tzinfo=timezone.utc)
            store = SRSStore(vault)
            add_manual_review(vault, store, "a.md", 4, first_time, emit=lambda _message: None)
            add_manual_review(vault, SRSStore(vault), "b.md", 7, second_time, emit=lambda _message: None)

            rows = build_report_rows(vault, SRSStore(vault), second_time, limit=None)

            self.assertEqual(len(rows), 2)

    def test_sync_obsidian_bridge_creates_queue_and_command_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            (vault / "folder").mkdir()
            (vault / "a.md").write_text("# A\n", encoding="utf-8")
            (vault / "folder" / "topic note.md").write_text("# Topic\n", encoding="utf-8")

            now = datetime(2030, 1, 3, 9, 0, tzinfo=timezone.utc)
            add_manual_review(vault, SRSStore(vault), "a.md", 6, now, emit=lambda _message: None)
            add_manual_review(vault, SRSStore(vault), "folder/topic note.md", 8, now, emit=lambda _message: None)

            rows, queue_summary = sync_obsidian_bridge(vault, SRSStore(vault), now, emit=lambda _message: None)

            self.assertEqual(queue_summary.path, QUEUE_NOTE_NAME)
            self.assertEqual(queue_summary.row_count, 2)
            self.assertTrue((vault / QUEUE_NOTE_NAME).exists())
            self.assertTrue((vault / COMMANDS_NOTE_NAME).exists())

            queue_content = (vault / QUEUE_NOTE_NAME).read_text(encoding="utf-8")
            self.assertIn("# SRS Queue", queue_content)
            self.assertIn("[[a|a]]", queue_content)
            self.assertIn("[[folder/topic note|topic note]]", queue_content)
            self.assertTrue(any(row.rel_path == "a.md" for row in rows))

            visible_notes = {note.relative_to(vault).as_posix() for note in iter_notes(vault)}
            self.assertEqual(visible_notes, {"a.md", "folder/topic note.md"})

    def test_process_command_note_executes_review_and_clears_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            note = vault / "folder" / "topic note.md"
            note.parent.mkdir()
            note.write_text("# Topic\n", encoding="utf-8")

            now = datetime(2030, 1, 4, 10, 0, tzinfo=timezone.utc)
            sync_obsidian_bridge(vault, SRSStore(vault), now, emit=lambda _message: None)
            commands_note = vault / COMMANDS_NOTE_NAME
            commands_note.write_text(
                (
                    "# SRS Commands\n\n"
                    "## Inbox\n\n"
                    "<!-- SRS-COMMANDS:START -->\n"
                    "review 7 [[folder/topic note]]\n"
                    "очередь\n"
                    "<!-- SRS-COMMANDS:END -->\n"
                ),
                encoding="utf-8",
            )

            results, rows, queue_summary = process_command_note(
                vault,
                SRSStore(vault),
                now,
                emit=lambda _message: None,
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(all(result.ok for result in results))
            self.assertFalse(has_pending_commands(vault))
            self.assertEqual(queue_summary.path, QUEUE_NOTE_NAME)
            self.assertTrue(any(row.rel_path == "folder/topic note.md" for row in rows))

            refreshed = SRSStore(vault)
            reviews = refreshed.get_reviews("folder/topic note.md")
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews[0].difficulty, 7)
            self.assertEqual(refreshed.get_meta_timestamp("last_command_ts"), now.isoformat())

            commands_content = commands_note.read_text(encoding="utf-8")
            self.assertIn("<!-- SRS-COMMANDS:START -->", commands_content)
            self.assertIn("<!-- SRS-COMMANDS:END -->", commands_content)
            self.assertNotIn("review 7 [[folder/topic note]]", commands_content)

            queue_content = (vault / QUEUE_NOTE_NAME).read_text(encoding="utf-8")
            self.assertIn("## Последняя обработка команд", queue_content)
            self.assertIn("review 7 [[folder/topic note]]", queue_content)

    def test_process_command_note_update_scans_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            note = vault / "fresh.md"
            note.write_text("# Fresh\n\n@! 5\n", encoding="utf-8")
            sync_obsidian_bridge(vault, SRSStore(vault), datetime(2030, 1, 1, tzinfo=timezone.utc), emit=lambda _message: None)

            commands_note = vault / COMMANDS_NOTE_NAME
            commands_note.write_text(
                (
                    "# SRS Commands\n\n"
                    "## Inbox\n\n"
                    "<!-- SRS-COMMANDS:START -->\n"
                    "обновить\n"
                    "<!-- SRS-COMMANDS:END -->\n"
                ),
                encoding="utf-8",
            )

            run_at = datetime(2030, 1, 5, 12, 0, tzinfo=timezone.utc)
            results, _rows, _summary = process_command_note(vault, SRSStore(vault), run_at, emit=lambda _message: None)

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            refreshed = SRSStore(vault)
            reviews = refreshed.get_reviews("fresh.md")
            self.assertEqual(len(reviews), 1)
            self.assertEqual(reviews[0].ts, run_at.isoformat())
            self.assertIn("@@ 5", note.read_text(encoding="utf-8"))

    def test_upsert_srs_tag_rewrites_multiline_tags_without_orphans(self) -> None:
        content = (
            "---\n"
            "tags:\n"
            "  - project\n"
            "  - srs/red\n"
            "  - review\n"
            "aliases:\n"
            "  - Demo\n"
            "---\n"
            "# Demo\n"
        )

        updated = upsert_srs_tag(content, "green")

        self.assertIn("tags: [project, review, srs/green]", updated)
        self.assertNotIn("  - project", updated)
        self.assertNotIn("srs/red", updated)
        self.assertIn("aliases:\n  - Demo", updated)

    def test_manual_review_rejects_paths_outside_vault(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vault = root / "vault"
            vault.mkdir()
            outside = root / "outside.md"
            outside.write_text("# Outside\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                add_manual_review(
                    vault,
                    SRSStore(vault),
                    "../outside.md",
                    5,
                    datetime(2030, 1, 6, tzinfo=timezone.utc),
                    emit=lambda _message: None,
                )

            self.assertEqual(outside.read_text(encoding="utf-8"), "# Outside\n")
            self.assertFalse((vault / ".obsidian_srs" / "index.json").exists())

    def test_process_command_note_keeps_failed_commands_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            (vault / "existing.md").write_text("# Existing\n", encoding="utf-8")
            commands_note = vault / COMMANDS_NOTE_NAME
            commands_note.write_text(
                (
                    "# SRS Commands\n\n"
                    "## Inbox\n\n"
                    "<!-- SRS-COMMANDS:START -->\n"
                    "review 7 [[missing]]\n"
                    "<!-- SRS-COMMANDS:END -->\n"
                ),
                encoding="utf-8",
            )

            results, _rows, _summary = process_command_note(
                vault,
                SRSStore(vault),
                datetime(2030, 1, 7, tzinfo=timezone.utc),
                emit=lambda _message: None,
            )

            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].ok)
            self.assertTrue(has_pending_commands(vault))
            self.assertIn("review 7 [[missing]]", commands_note.read_text(encoding="utf-8"))

    def test_corrupt_index_recovery_uses_unique_backup_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            vault = Path(temp_dir)
            store_dir = vault / ".obsidian_srs"
            store_dir.mkdir()
            index_path = store_dir / "index.json"
            first_backup = store_dir / "index.broken.json"
            index_path.write_text("{bad", encoding="utf-8")
            first_backup.write_text("old backup", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                store = SRSStore(vault)

            self.assertEqual(store.data, {"notes": {}, "meta": {}})
            self.assertEqual(first_backup.read_text(encoding="utf-8"), "old backup")
            self.assertFalse(index_path.exists())
            self.assertEqual((store_dir / "index.broken.1.json").read_text(encoding="utf-8"), "{bad")

    def test_parse_runtime_accepts_iso_like_strings(self) -> None:
        parsed = parse_runtime("2030-01-05T09:15:00+00:00")

        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.year, 2030)
        self.assertEqual(parsed.month, 1)
        self.assertEqual(parsed.day, 5)
        self.assertEqual(parsed.hour, 9)
        self.assertEqual(parsed.minute, 15)


if __name__ == "__main__":
    unittest.main()
