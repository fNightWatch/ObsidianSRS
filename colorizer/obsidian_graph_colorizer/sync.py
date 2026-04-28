from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import AppConfig
from .frontmatter import upsert_scalar_property
from .graph_json import HEX_RE, load_graph_json, save_graph_json, upsert_color_groups
from .io_utils import atomic_write_text
from .rules import pick_value
from .vault import iter_markdown_files, load_note


@dataclass
class SyncStats:
    scanned_files: int = 0
    updated_files: int = 0
    updated_properties: int = 0
    skipped_files: int = 0

    graph_groups_added: int = 0
    graph_groups_updated: int = 0
    graph_groups_removed: int = 0


def _normalize_hex_like(s: str) -> str:
    # Keep as-is, but trim whitespace. Accept '#RRGGBB' and 'RRGGBB'.
    return s.strip()


def sync_vault(
    vault: Path,
    cfg: AppConfig,
    *,
    dry_run: bool = False,
) -> SyncStats:
    stats = SyncStats()

    # If no properties are configured, do nothing.
    if not cfg.auto_properties:
        return stats

    graph_value_to_hex: Dict[str, str] = {}

    for md_path in iter_markdown_files(vault, cfg.ignore_globs):
        stats.scanned_files += 1
        note = load_note(vault, md_path)
        if note is None:
            stats.skipped_files += 1
            continue

        new_text = note.text
        file_changed = False

        # Apply each configured auto property sequentially.
        for prop_name, spec in cfg.auto_properties.items():
            desired = pick_value(note, spec.rules, spec.default)
            desired = str(desired)

            updated, changed = upsert_scalar_property(new_text, prop_name, desired)
            if changed:
                new_text = updated
                file_changed = True
                stats.updated_properties += 1

            # If this property is the graph color driver, remember it for group generation.
            if cfg.graph.enable and prop_name == cfg.graph.property_name:
                v = _normalize_hex_like(desired)
                if HEX_RE.match(v):
                    graph_value_to_hex[v] = v

        # If the user doesn't auto-generate the graph property, we can still read it from existing frontmatter.
        if cfg.graph.enable and cfg.graph.property_name not in cfg.auto_properties:
            existing = note.frontmatter.get(cfg.graph.property_name)
            if isinstance(existing, str):
                v = _normalize_hex_like(existing)
                if HEX_RE.match(v):
                    graph_value_to_hex[v] = v

        if file_changed:
            stats.updated_files += 1
            if not dry_run:
                atomic_write_text(md_path, new_text, encoding="utf-8")

    # Update global graph settings (graph.json) so Groups match the property values.
    if cfg.graph.enable and graph_value_to_hex:
        graph_path = vault / cfg.graph.graph_json_relpath
        graph = load_graph_json(graph_path)

        graph, added, updated, removed = upsert_color_groups(
            graph,
            query_template=cfg.graph.query_template,
            prop=cfg.graph.property_name,
            value_to_hexcolor=graph_value_to_hex,
            alpha=cfg.graph.alpha,
            prune_unused=cfg.graph.prune_unused_groups,
        )
        stats.graph_groups_added = added
        stats.graph_groups_updated = updated
        stats.graph_groups_removed = removed

        if not dry_run:
            save_graph_json(graph_path, graph)

    return stats
