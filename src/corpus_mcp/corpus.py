"""Document discovery, path containment, and chunking.

Two responsibilities that are easy to get wrong and worth isolating:

**Containment.** The server is pointed at a root directory and must never read
outside it. Agents pass tool arguments derived from model output, so a document
identifier is untrusted input: `../../.ssh/id_rsa` is a thing a confused or
adversarial agent will eventually ask for. Every path crossing the boundary goes
through `resolve_within_root`, which resolves symlinks *before* comparing, since
a symlink inside the root pointing outside it defeats a prefix check done on the
unresolved path.

**Chunking.** Retrieval happens over chunks, not whole documents, because a
whole document buries the relevant paragraph and blows the agent's context
budget. Chunks carry stable identifiers so a `search` result can be turned into
a `fetch` without re-running the search.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from dataclasses import dataclass

# Extensions read as plain text. Deliberately conservative: a binary file
# decoded as UTF-8 produces garbage that pollutes the index and the agent's
# context, so anything not listed here is skipped rather than guessed at.
TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".markdown", ".rst", ".org", ".text", ".log", ".csv", ".json", ".yaml", ".yml"}
)

# Skip directories that are large, generated, or private, and never useful to
# retrieve over.
SKIP_DIRECTORIES = frozenset(
    {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}
)

DEFAULT_CHUNK_CHARS = 1_200
DEFAULT_CHUNK_OVERLAP = 200

# Files above this size are skipped. A 200MB log would otherwise stall indexing
# and dominate the index without being useful to retrieve over.
DEFAULT_MAX_FILE_BYTES = 5_000_000


class PathOutsideRoot(ValueError):
    """Raised when a requested path escapes the configured corpus root."""


def resolve_within_root(root: pathlib.Path, candidate: str | pathlib.Path) -> pathlib.Path:
    """Resolve `candidate` relative to `root`, refusing anything that escapes.

    Symlinks are resolved before the containment check, so a link inside the
    root that points outside it is rejected rather than followed. An absolute
    candidate is *not* treated as absolute -- it is interpreted relative to the
    root, because a tool argument naming `/etc/passwd` means "the document
    called /etc/passwd inside the corpus", not the real one.
    """
    root = root.resolve()
    relative = pathlib.PurePosixPath(str(candidate).replace(os.sep, "/"))
    # Strip any leading slash so an absolute-looking argument cannot escape.
    parts = [part for part in relative.parts if part not in ("/", "")]

    resolved = (root / pathlib.Path(*parts)).resolve() if parts else root
    if resolved != root and root not in resolved.parents:
        raise PathOutsideRoot(f"{candidate!r} resolves outside the corpus root")
    return resolved


@dataclass(frozen=True)
class Document:
    """One source file in the corpus."""

    doc_id: str  # POSIX-style path relative to the root; stable and human-readable
    path: pathlib.Path
    text: str
    modified_ns: int

    @property
    def title(self) -> str:
        """First markdown heading or non-empty line, for display in results."""
        for line in self.text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped.lstrip("#").strip()[:120] or self.doc_id
        return self.doc_id


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a document."""

    chunk_id: str
    doc_id: str
    text: str
    ordinal: int  # position within the document, 0-based
    start: int  # character offset into the document, for locating the span

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def discover(
    root: pathlib.Path,
    *,
    extensions: frozenset[str] = TEXT_EXTENSIONS,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[Document]:
    """Read every indexable file under `root`, sorted for deterministic output."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"corpus root {root} is not a directory")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in extensions:
            continue

        try:
            stat = path.stat()
            if stat.st_size > max_file_bytes:
                continue
            # A file that is not valid UTF-8 is skipped rather than mangled with
            # replacement characters, which would index as noise.
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        documents.append(
            Document(
                doc_id=path.relative_to(root).as_posix(),
                path=path,
                text=text,
                modified_ns=stat.st_mtime_ns,
            )
        )
    return documents


def chunk_document(
    document: Document,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a document into overlapping chunks, preferring paragraph breaks.

    The overlap exists so a passage straddling a boundary is still wholly
    present in at least one chunk; without it, the sentence that answers the
    query can be split in half and retrieved by neither side.

    Boundaries snap backwards to the nearest blank line or newline when one sits
    reasonably close, so chunks tend to end at a paragraph rather than
    mid-sentence. That costs a little uniformity in chunk size and buys snippets
    that read as prose.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if not 0 <= overlap < chunk_chars:
        raise ValueError("overlap must be non-negative and smaller than chunk_chars")

    text = document.text
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0

    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            end = _snap_to_break(text, start, end)

        body = text[start:end]
        if body.strip():
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, ordinal),
                    doc_id=document.doc_id,
                    text=body,
                    ordinal=ordinal,
                    start=start,
                )
            )
            ordinal += 1

        if end >= len(text):
            break
        # Step forward by at least one character even in the pathological case
        # where the snap lands back at the start, so this cannot loop forever.
        start = max(end - overlap, start + 1)

    return chunks


def _snap_to_break(text: str, start: int, end: int) -> int:
    """Move `end` back to a paragraph or line break if one is nearby."""
    window = max((end - start) // 4, 1)
    floor = max(start + 1, end - window)

    paragraph = text.rfind("\n\n", floor, end)
    if paragraph != -1:
        return paragraph + 2

    line = text.rfind("\n", floor, end)
    if line != -1:
        return line + 1

    space = text.rfind(" ", floor, end)
    if space != -1:
        return space + 1

    return end


def make_chunk_id(doc_id: str, ordinal: int) -> str:
    """A stable, opaque-ish identifier the agent can round-trip to `fetch`.

    The document path is hashed rather than embedded so the identifier does not
    leak the directory layout into model context, but it stays deterministic:
    the same document and ordinal always produce the same id, so ids survive a
    reindex as long as the chunking did not change.
    """
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=6).hexdigest()
    return f"{digest}:{ordinal}"
