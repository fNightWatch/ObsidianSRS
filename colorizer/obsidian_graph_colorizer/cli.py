from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .config import load_config
from .sync import sync_vault


console = Console()


def _print_stats(stats) -> None:
    t = Table(title="Obsidian Graph Colorizer")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Scanned markdown files", str(stats.scanned_files))
    t.add_row("Updated files", str(stats.updated_files))
    t.add_row("Updated properties", str(stats.updated_properties))
    t.add_row("Skipped files", str(stats.skipped_files))
    t.add_section()
    t.add_row("Graph groups added", str(stats.graph_groups_added))
    t.add_row("Graph groups updated", str(stats.graph_groups_updated))
    t.add_row("Graph groups removed", str(stats.graph_groups_removed))
    console.print(t)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="obsidian-graph-colorizer",
        description="Auto-generate note properties (including graph color) based on note content, and optionally sync Graph View color groups.",
    )
    p.add_argument("--vault", required=False, help="Path to your Obsidian vault")
    p.add_argument("--config", required=True, help="Path to config YAML (see config.example.yaml)")
    p.add_argument("--dry-run", action="store_true", help="Do not write changes to files")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sync", help="One-time sync of all notes (and optionally graph.json).")
    sub.add_parser("watch", help="Watch the vault and re-sync on changes (requires watchdog).")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(Path(args.config))
    vault = Path(args.vault or cfg.vault_path or "").expanduser()

    if not vault.exists():
        console.print(f"[red]Vault path does not exist:[/red] {vault}")
        return 2

    def run_once():
        stats = sync_vault(vault, cfg, dry_run=bool(args.dry_run))
        _print_stats(stats)

    if args.cmd == "sync":
        run_once()
        return 0

    if args.cmd == "watch":
        try:
            from .watch import watch
        except ModuleNotFoundError:
            console.print("[red]Watch mode requires 'watchdog'. Install it via requirements.txt[/red]")
            return 3

        console.print("[bold]Watching vault for changes...[/bold] (Ctrl+C to stop)")
        run_once()
        watch(vault, run_once)
        return 0

    return 1
