from dataclasses import dataclass
from pathlib import Path


@dataclass
class Session:
    PREFIX: str = "@!"
    POSTFIX: str = "@@"
    PATH: Path = Path.home()
    ENCODING: str = "cp1251"
    INIT_MODE_ON: bool = True


@dataclass
class Colors:
    red: str = "red"
    yellow: str = "yellow"
    green: str = "green"
