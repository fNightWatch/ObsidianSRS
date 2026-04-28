from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import Rule
from .note import Note


def _fm_value_matches(actual: Any, expected: Any) -> bool:
    # Handles simple scalar match + list membership for list properties.
    if isinstance(actual, list):
        return expected in actual
    return actual == expected


def rule_matches(note: Note, rule: Rule) -> bool:
    m = rule.match

    # path_glob: any of them matches the relative path
    if m.path_glob:
        if not any(fnmatch.fnmatch(note.rel_path, g) for g in m.path_glob):
            return False

    # regex: any matches note.body
    if m.regex:
        ok = False
        for pat in m.regex:
            try:
                if re.search(pat, note.body, flags=0):
                    ok = True
                    break
            except re.error:
                # Bad regex -> rule never matches
                continue
        if not ok:
            return False

    # tags_any: any matches extracted tags
    if m.tags_any:
        wanted = {t.lstrip("#") for t in m.tags_any}
        have = {t.lstrip("#") for t in note.tags}
        if not (wanted & have):
            return False

    # frontmatter exact matches
    if m.frontmatter:
        for k, expected in m.frontmatter.items():
            if k not in note.frontmatter:
                return False
            if not _fm_value_matches(note.frontmatter.get(k), expected):
                return False

    return True


def pick_value(note: Note, rules: List[Rule], default: str) -> str:
    for rule in rules:
        if rule_matches(note, rule):
            return rule.value
    return default
