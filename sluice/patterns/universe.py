"""The wider stock list the pattern study runs on (task 10.2).

docs/specs/09-patterns.md: today's S&P 500, read once from Wikipedia and kept
as a fixed file under `configs/universe/`, so that every later run uses the same
list; another list is another file. Today's members looked back on ten years
carry survivorship bias (stocks dropped from the index since are missing), and
the file says so. Used only for the pattern study, never by the pipeline.
"""

import re
from collections.abc import Callable
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import requests

SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
ROOT = Path("configs/universe")
# Wikipedia refuses requests without a user agent.
HEADERS = {"User-Agent": "sluice-research/0.1 (pattern study; one-off read)"}
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class _Constituents(HTMLParser):
    """The first cell of each row of the table with id "constituents"."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0  # tables open inside the constituents table, itself counted
        self.found = False
        self.cell: list[str] | None = None
        self.first_in_row = False
        self.symbols: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table" and (self.depth or dict(attrs).get("id") == "constituents"):
            self.depth += 1
            self.found = True
        elif self.depth == 1 and tag == "tr":
            self.first_in_row = True
        elif self.depth == 1 and tag in ("td", "th"):
            self.cell = [] if self.first_in_row else None
            self.first_in_row = False

    def handle_endtag(self, tag):
        if tag == "table" and self.depth:
            self.depth -= 1
        elif self.depth == 1 and tag in ("td", "th") and self.cell is not None:
            self.symbols.append("".join(self.cell).strip())
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def parse(html: str) -> list[str]:
    """Ticker symbols from the page's constituents table, in Yahoo's spelling (BRK.B -> BRK-B)."""
    reader = _Constituents()
    reader.feed(html)
    if not reader.found:
        raise ValueError("no table with id 'constituents' on the page")
    symbols = [s.upper() for s in reader.symbols if s and s.lower() != "symbol"]
    if bad := [s for s in symbols if not _SYMBOL.match(s)]:
        raise ValueError(f"not ticker symbols: {bad[:5]}")
    return sorted({s.replace(".", "-") for s in symbols})


def fetch(get: Callable[..., requests.Response] = requests.get) -> list[str]:
    response = get(SOURCE, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return parse(response.text)


def save(symbols: list[str], name: str, today: date, root: Path = ROOT) -> Path:
    """Write the list once; an existing list is never overwritten."""
    path = root / f"{name}.txt"
    header = [
        f"# S&P 500 members as listed on Wikipedia on {today}: {len(symbols)} symbols.",
        f"# Source: {SOURCE}",
        "# Today's members looked back on: survivorship bias (dropped stocks are missing).",
        "# Used only by the pattern study (docs/specs/09-patterns.md). A new list is a new file.",
    ]
    root.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        file.write("\n".join([*header, *symbols]) + "\n")
    return path


def load(name: str, root: Path = ROOT) -> list[str]:
    lines = (root / f"{name}.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
