from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple


HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def hex_to_rgb_int(hex_color: str) -> int:
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    return int(s, 16)


def _quote_value_for_query(value: str) -> str:
    # Obsidian property search supports quoted values for exact matching.
    # We quote when special characters are present (e.g. '#', spaces).
    if re.search(r"[\s\[\]#:\"]", value):
        esc = value.replace('"', '\"')
        return f'"{esc}"'
    return value


def build_group_query(template: str, prop: str, value: str) -> str:
    return template.format(prop=prop, value_expr=_quote_value_for_query(value))


def load_graph_json(path: Path) -> Dict:
    if not path.exists():
        return {"colorGroups": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_graph_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def upsert_color_groups(
    graph: Dict,
    *,
    query_template: str,
    prop: str,
    value_to_hexcolor: Dict[str, str],
    alpha: float = 1.0,
    prune_unused: bool = False,
) -> Tuple[Dict, int, int, int]:
    """Ensure graph['colorGroups'] contains a group for each value.

    Returns: (graph, added_count, updated_count, removed_count)
    """
    groups = graph.get("colorGroups")
    if not isinstance(groups, list):
        groups = []
        graph["colorGroups"] = groups

    desired_queries: Dict[str, str] = {}  # query -> hex
    for value, hex_color in value_to_hexcolor.items():
        q = build_group_query(query_template, prop, value)
        desired_queries[q] = hex_color

    existing_by_query: Dict[str, int] = {}
    for i, g in enumerate(groups):
        if isinstance(g, dict) and isinstance(g.get("query"), str):
            existing_by_query[g["query"]] = i

    added = 0
    updated = 0

    for query, hex_color in desired_queries.items():
        rgb_int = hex_to_rgb_int(hex_color)
        color_obj = {"a": float(alpha), "rgb": rgb_int}
        if query in existing_by_query:
            idx = existing_by_query[query]
            g = groups[idx]
            if isinstance(g, dict) and g.get("color") != color_obj:
                g["color"] = color_obj
                updated += 1
        else:
            groups.append({"query": query, "color": color_obj})
            added += 1

    removed = 0
    if prune_unused:
        # Conservative pruning: only remove groups whose query matches our prop pattern.
        # NOTE: This can still remove user-defined groups that use the same property.
        prefix = f"[{prop}:"
        before = len(groups)
        groups[:] = [
            g for g in groups
            if not (
                isinstance(g, dict)
                and isinstance(g.get("query"), str)
                and g["query"].startswith(prefix)
                and g["query"] not in desired_queries
            )
        ]
        removed = before - len(groups)

    graph["colorGroups"] = groups
    return graph, added, updated, removed
