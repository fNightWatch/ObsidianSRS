from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import re
import yaml


FRONTMATTER_DELIM = "---"


@dataclass
class FrontmatterSplit:
    has_frontmatter: bool
    yaml_text: str
    body_text: str
    newline: str


def split_frontmatter(text: str) -> FrontmatterSplit:
    # Detect newline style early.
    newline = "\r\n" if "\r\n" in text and "\n" in text else "\n"
    # Normalize for parsing but keep original for writing.
    lines = text.splitlines(keepends=True)
    if not lines:
        return FrontmatterSplit(False, "", "", newline)
    if lines[0].strip() != FRONTMATTER_DELIM:
        return FrontmatterSplit(False, "", text, newline)

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == FRONTMATTER_DELIM:
            end_idx = i
            break

    if end_idx is None:
        # Malformed frontmatter; treat as no frontmatter.
        return FrontmatterSplit(False, "", text, newline)

    yaml_text = "".join(lines[1:end_idx])
    body_text = "".join(lines[end_idx + 1 :])
    return FrontmatterSplit(True, yaml_text, body_text, newline)


def parse_yaml(yaml_text: str) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(yaml_text) or {}
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _yaml_double_quote(value: str) -> str:
    # Always double-quote to avoid YAML pitfalls (e.g., '#' being treated as a comment).
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


_KEY_RE_CACHE: Dict[str, re.Pattern] = {}


def upsert_scalar_property(text: str, key: str, value: str) -> Tuple[str, bool]:
    """Insert or update a top-level YAML scalar property.

    This function tries hard to:
    - keep the rest of the file intact
    - avoid reformatting YAML (we update a single line, or append a new line)

    Caveat: if the existing property uses a multiline YAML structure, we won't try
    to rewrite it safely — we’ll append a new top-level key, which YAML would treat
    as duplicate and invalid. So in that case we return unchanged.
    """
    split = split_frontmatter(text)
    newline = split.newline
    quoted = _yaml_double_quote(value)

    def build_new(front_yaml: str, body: str) -> str:
        STRIP_CHARS = "\r\n"
        return f"{FRONTMATTER_DELIM}{newline}{front_yaml}{FRONTMATTER_DELIM}{newline}{body.lstrip(STRIP_CHARS)}"  # keep body but avoid extra blank lines

    if not split.has_frontmatter:
        yaml_block = f"{key}: {quoted}{newline}"
        return build_new(yaml_block, split.body_text), True

    # Work line-by-line inside frontmatter, preserving everything else.
    yaml_lines = split.yaml_text.splitlines(keepends=True)

    # Ensure pattern is compiled once per key.
    pat = _KEY_RE_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf"^({re.escape(key)})\s*:\s*(.*)$")
        _KEY_RE_CACHE[key] = pat

    changed = False
    found = False
    out_lines: List[str] = []

    for line in yaml_lines:
        raw = line.rstrip("\r\n")
        m = pat.match(raw)
        if m and not raw.startswith(" ") and not raw.startswith("\t"):
            # Guard against multiline YAML (| or >) which would spill over.
            current = m.group(2).strip()
            if current in ("|", ">") or current.startswith("|") or current.startswith(">") :
                return text, False
            new_line = f"{key}: {quoted}{newline}"
            out_lines.append(new_line)
            found = True
            if new_line != line:
                changed = True
        else:
            out_lines.append(line)

    if not found:
        # Append at end of frontmatter, ensure there's a trailing newline.
        if out_lines and not out_lines[-1].endswith(("\n", "\r\n")):
            out_lines[-1] = out_lines[-1] + newline
        out_lines.append(f"{key}: {quoted}{newline}")
        changed = True

    if not changed:
        return text, False

    new_yaml = "".join(out_lines)
    return build_new(new_yaml, split.body_text), True


def get_inline_tags(body_text: str) -> List[str]:
    # Very lightweight tag extraction. This will also match tags in code blocks.
    # Improve later if needed.
    tag_re = re.compile(r"(?<![\\w/])#([A-Za-z0-9_\-\/]+)")
    return list({m.group(1) for m in tag_re.finditer(body_text)})


def get_frontmatter_tags(fm: Dict[str, Any]) -> List[str]:
    tags = fm.get("tags") or fm.get("tag") or []
    out: List[str] = []
    if isinstance(tags, str):
        # Allow 'tag1, tag2' or '#tag1 #tag2' etc.
        for part in re.split(r"[\s,]+", tags.strip()):
            if not part:
                continue
            out.append(part.lstrip("#"))
    elif isinstance(tags, list):
        for item in tags:
            if isinstance(item, str):
                out.append(item.lstrip("#"))
    return out
