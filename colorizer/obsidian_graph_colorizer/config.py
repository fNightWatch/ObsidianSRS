from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RuleMatch:
    # Match is AND across fields. Within each list, it's OR.
    path_glob: List[str] = field(default_factory=list)
    regex: List[str] = field(default_factory=list)
    tags_any: List[str] = field(default_factory=list)
    frontmatter: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rule:
    name: str
    value: str
    match: RuleMatch = field(default_factory=RuleMatch)


@dataclass
class PropertySpec:
    # For now we only support writing scalar string properties.
    default: str
    rules: List[Rule] = field(default_factory=list)


@dataclass
class GraphSpec:
    enable: bool = True
    property_name: str = "graph_color"
    graph_json_relpath: str = ".obsidian/graph.json"
    # The query used in graph groups. We use Obsidian property search syntax:
    #   [property:value]
    # If value contains special chars (like '#'), we wrap it in quotes.
    query_template: str = "[{prop}:{value_expr}]"
    alpha: float = 1.0
    prune_unused_groups: bool = False


@dataclass
class AppConfig:
    # Vault root is provided via CLI by default, but you can also set it here.
    vault_path: Optional[str] = None

    ignore_globs: List[str] = field(default_factory=lambda: [
        ".obsidian/**",
        ".trash/**",
        ".git/**",
        "node_modules/**",
    ])

    # Defines which properties we auto-generate and write to notes.
    # Example:
    #   auto_properties:
    #     graph_color:
    #       default: "#808080"
    #       rules: ...
    auto_properties: Dict[str, PropertySpec] = field(default_factory=dict)

    graph: GraphSpec = field(default_factory=GraphSpec)


def _parse_rule(obj: Dict[str, Any]) -> Rule:
    match_obj = obj.get("match", {}) or {}
    rm = RuleMatch(
        path_glob=list(match_obj.get("path_glob", []) or []),
        regex=list(match_obj.get("regex", []) or []),
        tags_any=list(match_obj.get("tags_any", []) or []),
        frontmatter=dict(match_obj.get("frontmatter", {}) or {}),
    )
    return Rule(
        name=str(obj.get("name", "unnamed")),
        value=str(obj.get("value")),
        match=rm,
    )


def _parse_property_spec(obj: Dict[str, Any]) -> PropertySpec:
    rules_in = obj.get("rules", []) or []
    return PropertySpec(
        default=str(obj.get("default", "")),
        rules=[_parse_rule(r) for r in rules_in],
    )


def load_config(path: Path) -> AppConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data)}")

    auto_props_raw = data.get("auto_properties", {}) or {}
    auto_properties: Dict[str, PropertySpec] = {}
    if isinstance(auto_props_raw, dict):
        for prop_name, spec in auto_props_raw.items():
            if not isinstance(spec, dict):
                continue
            auto_properties[str(prop_name)] = _parse_property_spec(spec)

    graph_raw = data.get("graph", {}) or {}
    graph = GraphSpec(
        enable=bool(graph_raw.get("enable", True)),
        property_name=str(graph_raw.get("property_name", "graph_color")),
        graph_json_relpath=str(graph_raw.get("graph_json_relpath", ".obsidian/graph.json")),
        query_template=str(graph_raw.get("query_template", "[{prop}:{value_expr}]")),
        alpha=float(graph_raw.get("alpha", 1.0)),
        prune_unused_groups=bool(graph_raw.get("prune_unused_groups", False)),
    )

    return AppConfig(
        vault_path=data.get("vault_path"),
        ignore_globs=list(data.get("ignore_globs", AppConfig().ignore_globs)),
        auto_properties=auto_properties,
        graph=graph,
    )
