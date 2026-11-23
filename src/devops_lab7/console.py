"""Налаштування консолі, заголовки розділів і текстові таблиці."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

LINE_WIDTH: Final = 78
OUTPUT_DIR: Final = Path("output")
DATA_DIR: Final = Path("data")


def setup_console() -> None:
    """UTF-8 для консолі Windows (кирилиця, «лапки», тире)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print("=" * LINE_WIDTH)
    print(title)
    print("-" * LINE_WIDTH)


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Проста текстова таблиця з вирівнюванням стовпців."""
    cells = [[str(h) for h in header]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]

    def line(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(cells[0]), rule, *(line(row) for row in cells[1:])])
