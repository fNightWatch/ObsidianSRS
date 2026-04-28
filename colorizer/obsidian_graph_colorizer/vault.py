from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set

from .frontmatter import split_frontmatter, parse_yaml, get_frontmatter_tags, get_inline_tags
from .note import Note


def _is_ignored(rel_posix: str, ignore_globs: List[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, g) for g in ignore_globs)


def iter_markdown_files(vault: Path, ignore_globs: List[str]) -> Iterator[Path]:
    for path in vault.rglob("*.md"):
        rel_posix = path.relative_to(vault).as_posix()
        if _is_ignored(rel_posix, ignore_globs):
            continue
        yield path


def load_note(vault: Path, path: Path) -> Optional[Note]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    split = split_frontmatter(text)
    fm = parse_yaml(split.yaml_text) if split.has_frontmatter else {}
    body = split.body_text

    tags = []
    tags.extend(get_frontmatter_tags(fm))
    tags.extend(get_inline_tags(body))
    # de-dup preserving order
    seen: Set[str] = set()
    uniq = []
    for t in tags:
        t2 = t.lstrip("#")
        if t2 and t2 not in seen:
            seen.add(t2)
            uniq.append(t2)

    return Note(
        path=path,
        rel_path=path.relative_to(vault).as_posix(),
        text=text,
        body=body,
        frontmatter=fm,
        tags=uniq,
    )
