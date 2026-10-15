"""Keyword search (BM25) over the policy passages, with a tokenizer that keeps codes whole. M4.

Why a custom tokenizer (A4 lesson + the M4 probe): splitting on spaces keeps punctuation glued
to words, so the question's "(CARD-ECHO)?" never matches the policy's "(CARD-ECHO):", and BM25
scored worse than dense search (recall@5 0.88 vs 0.91). The rules here:
- lower-case; HTML line breaks ("<br>") become spaces;
- a token is letters/digits joined by hyphens or dots, so codes, plan names, section numbers and
  decimals stay whole:
  "card-echo", "meridian-silver", "pre-authorisation"; a hyphenated token is also emitted without
  hyphens ("preauthorisation", "copay") so "co-pay" and "copay" match;
- money and numbers lose "$" and thousands commas ("$1,500" -> "1500"), "%" is dropped;
- a short list of very common words is ignored.

BM25 (Okapi; k1 = 1.2, b = 0.75): a passage scores higher the more of the question's words it
contains, rare words count more than common ones, and long passages are not favoured just for
being long. Only passages sharing at least one word with the question are returned, best first.
The index is built in memory from the same passages as the dense index (86: milliseconds).
"""
import math
import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel

_STOP_TEXT = (
    "a an the of for to in on and or is are be been was were what how many much which who whom "
    "with under does do did i my me our we you your this that these those it its as at by from "
    "if then than can could should would will any all there their them they has have had not "
    "no yes"
)
STOPWORDS = frozenset(_STOP_TEXT.split())
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")      # "card-echo", "4.2", "3.5"


def tokenize(text: str) -> list[str]:
    """Code-aware tokens, in order (duplicates kept: BM25 counts them)."""
    text = text.lower().replace("<br>", " ")
    text = _NUMBER.sub(lambda m: " " + m.group(0).strip("$%").replace(",", "") + " ", text)
    tokens: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok:
            tokens.append(tok.replace("-", ""))
    return tokens


class KeywordHit(BaseModel):
    chunk_id: str
    score: float
    rank: int          # 1 = best


class BM25Index:
    """In-memory BM25 over (chunk_id, text) pairs."""

    def __init__(self, items: Iterable[tuple[str, str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tf: list[Counter] = []
        self.lengths: list[int] = []
        for chunk_id, text in items:
            tokens = tokenize(text)
            self.ids.append(chunk_id)
            self.tf.append(Counter(tokens))
            self.lengths.append(len(tokens))
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate chunk ids in keyword index")
        self.n = len(self.ids)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.df = Counter(t for counts in self.tf for t in counts)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[KeywordHit]:
        terms = tokenize(query)
        if not terms or not self.n:
            return []
        scored = []
        for i, counts in enumerate(self.tf):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
            for term in terms:
                f = counts.get(term, 0)
                if f:
                    score += self.idf(term) * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, self.ids[i]))
        scored.sort(key=lambda s: (-s[0], s[1]))          # ties broken by id: same result every run
        return [KeywordHit(chunk_id=cid, score=round(s, 6), rank=r)
                for r, (s, cid) in enumerate(scored[:top_k], start=1)]
