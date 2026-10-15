"""Read the policy corpus into `Document` objects with their metadata. M3 (Markdown); PDF in M4.

Each policy starts with a header block (title line, organisation lines, document ID, effective
date, owner…) that ends at the first '---' rule or, in the pre-authorisation matrix which has no
rule, at the first numbered section ("1.0 …"). The header is kept apart from the body so the
chunker (step 5) only splits real policy text, and the metadata is parsed out of it:

- title:          the first '# ' line (the golden set's `must_cite` uses exactly these titles)
- policy_id:      the value after any of the 6 labels the documents use (Document ID, POLICY REF…)
- effective_date: two date styles appear ("November 1, 2024" and "1 JANUARY 2024")

A document that can't be read properly is reported in `problems` and the rest still load
(slides: one bad file must not block the whole pipeline). Only `*.md` / `*.pdf` files in the
configured corpus folder are read; the evaluation folder is refused by the config loader.

PDF (M4). The PDF text has no Markdown markers: a heading runs straight into its first sentence
("4.1 Purpose and Scope This policy defines…"), each document numbers its sections in one of four
styles ("SECTION 1:", "1.0 …", "1. …", "4.1 …"), and list items / sub-clauses also start lines
with numbers. So, per document:
1. for each numbering style, walk the line starts and keep the longest clean run 1, 2, 3… (a
   number that doesn't continue the run is a list item or sub-clause, not a section);
2. the style with the longest run gives the sections; everything before the first is the header;
3. each heading is cut off from the sentence that follows it and written as a '### ' line, and the
   wrapped lines of a table are joined into one line (one block the chunker never cuts),
so the M3 chunker runs unchanged and produces the same stable chunk IDs as the Markdown twin.
"""
import hashlib
import re
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from capstone.settings import PROJECT_ROOT, get_retrieval_config

ID_LABEL = re.compile(
    r"(?:Policy ID|Document ID|Document Code|Document Ref|Document Control|Policy Ref)"
    r"\W*:\W*\s*([A-Z0-9][A-Z0-9-]+)", re.IGNORECASE)
DATE_LABEL = re.compile(r"Effective Date\W*:\W*\s*([A-Za-z]+ \d{1,2}, \d{4}|\d{1,2} [A-Za-z]+ \d{4})",
                        re.IGNORECASE)
DATE_FORMATS = ("%B %d, %Y", "%d %B %Y")
HEADER_END = re.compile(r"(?m)^(?:---\s*$|\d+\.0\s)")

# PDF section numbering styles: (regex on a line start, how to read (major, minor) from the match)
PDF_STYLES = {
    "section": re.compile(r"^SECTION\s+(\d+)[:.]\s+(\S.*)$"),
    "n.0": re.compile(r"^(\d+)\.0\s+(\S.*)$"),
    "n.": re.compile(r"^(\d+)\.\s+([A-Z].*)$"),
    "n.m": re.compile(r"^(\d+)\.([1-9]\d*)\s+([A-Z].*)$"),
}
CONNECTORS = {"and", "of", "for", "by", "to", "in", "on", "the", "&", "or", "with", "a", "an"}


class Document(BaseModel):
    slug: str                    # file name without .md, e.g. "preauth-matrix"
    title: str
    policy_id: str | None
    effective_date: date | None
    source: str                  # path relative to the project root
    header: str
    body: str
    sha256: str                  # of the whole file, to notice when a document changes

    @property
    def word_count(self) -> int:
        return len(self.body.split())


class CorpusLoad(BaseModel):
    documents: list[Document]
    problems: list[str]


def parse_date(text: str) -> date | None:
    cleaned = " ".join(text.replace("*", " ").split()).title()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()  # noqa: DTZ007 - only the date is kept
        except ValueError:
            continue
    return None


def split_header(text: str) -> tuple[str, str]:
    """(header, body). The rule line itself belongs to neither; a numbered section starts the body."""
    match = HEADER_END.search(text)
    if not match:
        return "", text
    header = text[: match.start()]
    body = text[match.end():] if match.group(0).startswith("---") else text[match.start():]
    return header.strip(), body.strip()


def parse_document(path: Path) -> tuple[Document | None, list[str]]:
    """Read one Markdown policy. Returns (document or None, problems found)."""
    problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if not first.startswith("# "):
        return None, [f"{path.name}: no '# ' title line; skipped"]
    header, body = split_header(text)
    if not body:
        return None, [f"{path.name}: empty body; skipped"]

    id_match = ID_LABEL.search(header)
    date_match = DATE_LABEL.search(header)
    effective = parse_date(date_match.group(1)) if date_match else None
    if not id_match:
        problems.append(f"{path.name}: no policy id found")
    if not effective:
        problems.append(f"{path.name}: no readable effective date")

    source = path.resolve()
    doc = Document(
        slug=path.stem, title=first[2:].strip(),
        policy_id=id_match.group(1).upper() if id_match else None, effective_date=effective,
        source=str(source.relative_to(PROJECT_ROOT)) if source.is_relative_to(PROJECT_ROOT) else str(source),
        header=header, body=body, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    return doc, problems


def _section_run(lines: list[str], style: str) -> list[tuple[int, str, str]]:
    """Longest clean 1, 2, 3… run of headings in one numbering style: [(line index, number, rest)]."""
    pattern, run = PDF_STYLES[style], []
    major = None
    for i, line in enumerate(lines):
        m = pattern.match(line.strip())
        if not m:
            continue
        if style == "n.m":
            maj, minor, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            if major is None and minor == 1:
                major = maj
            if maj == major and minor == len(run) + 1:
                run.append((i, f"{maj}.{minor}", rest))
        else:
            n, rest = int(m.group(1)), m.group(2)
            if n == len(run) + 1:
                number = f"{n}.0" if style == "n.0" else f"{n}"
                run.append((i, number, rest))
    return run


def split_heading(rest: str) -> tuple[str, str]:
    """('Purpose and Scope', 'This policy defines…') from 'Purpose and Scope This policy defines…'."""
    words = rest.split()
    if not words:
        return "", ""
    upper = sum(w.isupper() for w in words[:3] if any(c.isalpha() for c in w)) >= 2
    keep = 0
    for i, w in enumerate(words):
        if re.match(r"^\d+(\.\d+)*\.?$", w):                          # "1.1" / "1." starts the text
            break
        letters = [c for c in w if c.isalpha()]
        if upper:
            if letters and not all(c.isupper() for c in letters):
                break
        else:
            nxt = words[i + 1] if i + 1 < len(words) else ""
            starts_sentence = (w[:1].isupper() and nxt[:1].islower() and nxt.lower() not in CONNECTORS
                               and i > 0)
            if starts_sentence or not (w[:1].isupper() or w.lower() in CONNECTORS or w[:1] in "(&-"):
                break
        keep = i + 1
    return " ".join(words[:keep]).rstrip(":"), " ".join(words[keep:])


def _join_tables(lines: list[str]) -> list[str]:
    """Join consecutive lines that contain '|' into one line, so a table is one unbreakable block."""
    out: list[str] = []
    for line in lines:
        if "|" in line and out and "|" in out[-1]:
            out[-1] = f"{out[-1]} {line.strip()}"
        else:
            out.append(line)
    return out


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # imported here so Markdown-only use doesn't need it

    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def parse_pdf_document(path: Path) -> tuple[Document | None, list[str]]:
    """Read one PDF policy into the same Document shape as the Markdown parser."""
    text = pdf_text(path)
    lines = text.splitlines()
    if not text.strip():
        return None, [f"{path.name}: no text layer (scanned PDF?); skipped"]
    best = max((_section_run(lines, s) for s in PDF_STYLES), key=len)
    if len(best) < 2:
        return None, [f"{path.name}: no numbered sections found; skipped"]
    first = best[0][0]
    header = "\n".join(lines[:first]).strip()
    starts = {i: rest for i, _number, rest in best}
    body_lines: list[str] = []
    for i in range(first, len(lines)):
        if i in starts:
            rest = starts[i]
            prefix = lines[i].strip()[: lines[i].strip().find(rest)].strip()   # "SECTION 1:" / "4.1"
            heading, after = split_heading(rest)
            nxt = lines[i + 1].strip() if i + 1 < len(lines) and i + 1 not in starts else ""
            if not after and heading.isupper() and nxt:
                more, rest_of_next = split_heading(nxt)      # a long CAPS heading wrapped to next line
                if more and more.isupper():
                    heading = f"{heading} {more}"
                    lines[i + 1] = rest_of_next              # (lines is this function's own list)
            body_lines.append(f"### {prefix} {heading}".rstrip())
            if after:
                body_lines.append(after)
        else:
            body_lines.append(lines[i])
    body = "\n".join(_join_tables(body_lines)).strip()

    problems: list[str] = []
    title = header.splitlines()[0].strip() if header else ""
    if not title:
        return None, [f"{path.name}: no title line; skipped"]
    id_match = ID_LABEL.search(header)
    date_match = DATE_LABEL.search(header)
    effective = parse_date(date_match.group(1)) if date_match else None
    if not id_match:
        problems.append(f"{path.name}: no policy id found")
    if not effective:
        problems.append(f"{path.name}: no readable effective date")
    source = path.resolve()
    doc = Document(
        slug=path.stem, title=title, policy_id=id_match.group(1).upper() if id_match else None,
        effective_date=effective,
        source=str(source.relative_to(PROJECT_ROOT)) if source.is_relative_to(PROJECT_ROOT) else str(source),
        header=header, body=body, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return doc, problems


def load_corpus(folder: Path | None = None, source: str | None = None) -> CorpusLoad:
    """All policies of one copy (default: configs/retrieval.yaml), sorted by slug.

    source "markdown" reads *.md, "pdf" reads *.pdf; a broken file is reported, the rest load.
    """
    if folder is None or source is None:
        cfg = get_retrieval_config()
        folder = Path(folder) if folder else cfg.corpus_dir
        source = source or cfg.corpus.source
    folder = Path(folder)
    parse = parse_pdf_document if source == "pdf" else parse_document
    documents, problems = [], []
    for path in sorted(folder.glob("*.pdf" if source == "pdf" else "*.md")):
        try:
            doc, found = parse(path)
        except Exception as exc:  # noqa: BLE001 - any unreadable/corrupt file is reported, never fatal
            doc, found = None, [f"{path.name}: unreadable ({exc.__class__.__name__}); skipped"]
        problems.extend(found)
        if doc:
            documents.append(doc)
    return CorpusLoad(documents=documents, problems=problems)
