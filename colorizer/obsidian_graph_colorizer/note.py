from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class Note:
    path: Path           # absolute path
    rel_path: str        # vault-relative POSIX path
    text: str            # full file content
    body: str            # content without frontmatter
    frontmatter: Dict[str, Any]
    tags: List[str]
