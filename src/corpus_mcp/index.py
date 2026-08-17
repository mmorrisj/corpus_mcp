"""BM25 index over chunks, with incremental rebuilds.

Why BM25 rather than embeddings: this server is meant to be pointed at a
directory and work — no model download, no API key, no GPU, no vector database
to run alongside it. For the keyword-ish queries an agent actually issues while
navigating a corpus it already knows something about, lexical retrieval is
strong, and it has the property that matters most in an agent loop: it is fast
and it never silently costs money. Semantic search is a worthwhile addition, not
a precondition for the thing being useful.

Reindexing is incremental on modification time. An agent may call `search` many
times over a long session while the user edits files underneath it; rebuilding
the whole index on every call would be wasteful, and never rebuilding would
serve stale text.
"""

from __future__ import annotations

import math
import pathlib
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass

from corpus_mcp.corpus import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    TEXT_EXTENSIONS,
    Chunk,
    Document,
    chunk_document,
    discover,
)

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have he in is it its of on that the
    to was were will with this these those or not but if then than what which
    who when where how""".split()
)

# BM25 parameters. These are the values BEIR uses for its reference baseline;
# keeping them rather than hand-tuning means behaviour is predictable and
# comparable to published results.
K1 = 0.9
B = 0.4


def normalize(token: str) -> str:
    """Fold common English inflections so `tastes` matches a query for `taste`.

    This is a suffix stripper, not a real stemmer. A full Porter implementation
    would be a hundred lines and a maintenance surface, and the long tail it
    buys (`operational` → `oper`) is as likely to hurt as help on the short
    content-word queries an agent actually issues. Plurals and the common verb
    endings are where nearly all of the practical benefit is.

    Length guards keep it from mangling short words: `bed` must not become `b`,
    and `ss` endings are left alone so `business` survives intact.
    """
    if len(token) > 4:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith(("ches", "shes", "xes", "zes", "sses")):
            return token[:-2]
        if len(token) > 6 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 5 and token.endswith("ed"):
            return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split on alphanumerics, drop stopwords, fold inflections.

    Used for both indexing and querying -- they must share this function, since
    any divergence between the two silently costs recall.
    """
    return [normalize(token) for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class CorpusIndex:
    """A searchable, self-refreshing index over a directory of documents.

    Thread-safe: the MCP server may handle concurrent tool calls, and a rebuild
    triggered by one must not be observed half-finished by another.
    """

    def __init__(
        self,
        root: pathlib.Path,
        *,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        extensions: frozenset[str] = TEXT_EXTENSIONS,
    ) -> None:
        self.root = pathlib.Path(root).resolve()
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self.extensions = extensions

        self._lock = threading.RLock()
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        # doc_id -> chunks in document order, for neighbour lookup on fetch
        self._by_document: dict[str, list[Chunk]] = {}
        self._postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self._lengths: dict[str, float] = {}
        self._idf: dict[str, float] = {}
        self._average_length = 0.0
        self._fingerprint: tuple[tuple[str, int], ...] = ()

    # -- state ------------------------------------------------------------

    @property
    def documents(self) -> list[Document]:
        with self._lock:
            return list(self._documents.values())

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "documents": len(self._documents),
                "chunks": len(self._chunks),
                "terms": len(self._postings),
            }

    # -- building ---------------------------------------------------------

    def refresh(self, *, force: bool = False) -> bool:
        """Rebuild if the corpus changed on disk. Returns True if it rebuilt.

        Change detection is a fingerprint of every indexable file's path and
        modification time, which catches edits, additions and deletions in one
        comparison. It does not catch an edit that preserves mtime -- rare
        outside deliberate tampering, and `force` covers it.
        """
        documents = discover(self.root, extensions=self.extensions)
        fingerprint = tuple((document.doc_id, document.modified_ns) for document in documents)

        with self._lock:
            if not force and fingerprint == self._fingerprint and self._documents:
                return False
            self._rebuild(documents, fingerprint)
            return True

    def _rebuild(self, documents: list[Document], fingerprint: tuple) -> None:
        self._documents = {document.doc_id: document for document in documents}
        self._chunks = {}
        self._by_document = {}
        self._postings = defaultdict(list)
        self._lengths = {}

        document_frequency: Counter[str] = Counter()

        for document in documents:
            chunks = chunk_document(
                document, chunk_chars=self.chunk_chars, overlap=self.chunk_overlap
            )
            if chunks:
                self._by_document[document.doc_id] = chunks

            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
                tokens = tokenize(chunk.text)
                self._lengths[chunk.chunk_id] = float(len(tokens))

                frequencies = Counter(tokens)
                for term, frequency in frequencies.items():
                    self._postings[term].append((chunk.chunk_id, frequency))
                document_frequency.update(frequencies.keys())

        total = len(self._chunks)
        self._average_length = sum(self._lengths.values()) / total if total else 0.0
        self._idf = {
            term: max(0.0, math.log(1 + (total - df + 0.5) / (df + 0.5)))
            for term, df in document_frequency.items()
        }
        self._fingerprint = fingerprint

    # -- querying ---------------------------------------------------------

    def search(self, query: str, limit: int) -> list[SearchHit]:
        """Top `limit` chunks for `query`, best first."""
        if limit <= 0:
            return []

        self.refresh()
        with self._lock:
            terms = tokenize(query)
            if not terms or not self._chunks:
                return []

            scores: dict[str, float] = defaultdict(float)
            for term in terms:
                idf = self._idf.get(term, 0.0)
                if idf <= 0:
                    continue
                for chunk_id, frequency in self._postings.get(term, ()):
                    length_norm = K1 * (
                        1 - B + B * (self._lengths[chunk_id] / self._average_length)
                        if self._average_length
                        else 1 - B
                    )
                    denominator = frequency + length_norm
                    if denominator:
                        scores[chunk_id] += idf * (frequency * (K1 + 1)) / denominator

            # Ties break on chunk id so repeated identical queries return a
            # stable order -- an agent re-running a search should not see the
            # results shuffle.
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
            return [SearchHit(self._chunks[chunk_id], score) for chunk_id, score in ranked]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        self.refresh()
        with self._lock:
            return self._chunks.get(chunk_id)

    def neighbours(self, chunk: Chunk, radius: int) -> list[Chunk]:
        """The chunk plus up to `radius` chunks either side, in document order.

        This is what makes the search/fetch split work: search returns a short
        snippet, and fetch widens it to surrounding context on demand instead of
        the agent having to pull the whole document.
        """
        with self._lock:
            siblings = self._by_document.get(chunk.doc_id, [])
            if not siblings:
                return [chunk]
            low = max(0, chunk.ordinal - radius)
            high = min(len(siblings), chunk.ordinal + radius + 1)
            return siblings[low:high]

    def get_document(self, doc_id: str) -> Document | None:
        self.refresh()
        with self._lock:
            return self._documents.get(doc_id)
