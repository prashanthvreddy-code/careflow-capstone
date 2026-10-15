"""Split policy documents into passages ("chunks") for the index. M3 (strategy decided in phase 2).

Rules:
- One chunk per section. A section starts at a '##' or '###' heading, or at a 'N.0' numbered
  heading (the pre-authorisation matrix has no Markdown headings). '####' sub-headings such as
  table captions stay inside their section. Text before the first heading becomes a 'preamble'.
- A section longer than `max_words` is split into parts at block boundaries: blank lines first,
  then numbered clauses ("2.3 …"), then single lines. A Markdown table is one block and is never
  cut, even if it alone is longer than `max_words` (a half table loses its column meaning).
- Stable IDs: "<doc slug>#<section number>", e.g. "preauth-matrix#4.0" or "copay-schedule#4.2";
  "-part2" is added for later parts. The same document text always gives the same IDs, so a
  citation stays valid when the index is rebuilt.
- `embed_text` = document title + section heading + text. The heading travels with every part,
  so a later part still says which policy and section it belongs to (contextual header).
"""
import re
from datetime import date

from pydantic import BaseModel

from capstone.rag.ingestion import Document

SECTION_START = re.compile(r"^(?:#{2,3} |\d+\.0\s)")
SECTION_NUMBER = re.compile(r"^(?:SECTION\s+)?(\d+(?:\.\d+)*)", re.IGNORECASE)
CLAUSE_START = re.compile(r"^\**\d+\.\d+")


class Chunk(BaseModel):
    chunk_id: str
    doc_slug: str
    doc_title: str
    policy_id: str | None
    effective_date: date | None
    source: str
    section: str               # heading text without '#' / '**', e.g. "4.2 Cost-Share Schedule by Plan"
    part: int                  # 1-based
    n_parts: int
    text: str                  # the passage itself (first part includes the heading line)
    embed_text: str            # what gets embedded: title + section + text

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def clean_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line.lstrip("#").replace("**", "")).strip()


def section_key(heading: str, used: set[str]) -> str:
    """Short, stable key for a section: its number if it has one, else a slug of the heading."""
    match = SECTION_NUMBER.match(heading)
    key = match.group(1) if match else re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:40] or "section"
    base, n = key, 2
    while key in used:                      # two sections with the same number stay distinct
        key, n = f"{base}-{n}", n + 1
    used.add(key)
    return key


def split_sections(body: str) -> list[tuple[str, str]]:
    """[(heading, text)], heading '' for a preamble before the first section."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in body.splitlines():
        if SECTION_START.match(line):
            sections.append((clean_heading(line), [line]))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(lines).strip()) for h, lines in sections if "\n".join(lines).strip()]


def _blocks(text: str) -> list[str]:
    """Split a section into blocks; every Markdown table is exactly one block."""
    blocks: list[str] = []
    current: list[str] = []
    in_table = False

    def flush():
        if current and "\n".join(current).strip():
            blocks.append("\n".join(current).strip())
        current.clear()

    for line in text.splitlines():
        is_table = line.lstrip().startswith("|")
        if is_table != in_table:
            flush()
            in_table = is_table
        if not is_table and (not line.strip() or CLAUSE_START.match(line)):
            flush()
        if line.strip() or is_table:
            current.append(line)
    flush()
    return blocks


def _pack(blocks: list[str], max_words: int) -> list[str]:
    """Group consecutive blocks into parts of at most max_words (an oversized block stands alone)."""
    parts: list[list[str]] = []
    words = 0
    for block in blocks:
        n = len(block.split())
        if parts and words + n <= max_words:
            parts[-1].append(block)
            words += n
        else:
            parts.append([block])
            words = n
    return ["\n\n".join(p) for p in parts]


def chunk_document(doc: Document, max_words: int = 300) -> list[Chunk]:
    chunks: list[Chunk] = []
    used: set[str] = set()
    for heading, text in split_sections(doc.body):
        name = heading or "Preamble"
        key = section_key(heading, used) if heading else section_key("preamble", used)
        parts = [text] if len(text.split()) <= max_words else _pack(_blocks(text), max_words)
        for i, part in enumerate(parts, start=1):
            chunks.append(Chunk(
                chunk_id=f"{doc.slug}#{key}" + (f"-part{i}" if i > 1 else ""),
                doc_slug=doc.slug, doc_title=doc.title, policy_id=doc.policy_id,
                effective_date=doc.effective_date, source=doc.source, section=name,
                part=i, n_parts=len(parts), text=part,
                embed_text=f"{doc.title} — {name}\n\n{part}",
            ))
    return chunks


def chunk_corpus(documents: list[Document], max_words: int = 300) -> list[Chunk]:
    chunks = [c for doc in documents for c in chunk_document(doc, max_words)]
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids")     # would make citations ambiguous
    return chunks
