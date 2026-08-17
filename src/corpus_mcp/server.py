"""The MCP server: tools, resources, and the output budget that makes them usable.

The design question for an MCP server is not "what can I expose" but "what will
an agent do with it". Three decisions follow from that:

**Search and fetch are separate tools.** A single `search` returning full chunks
is simpler to write and much worse to use: ten results at 1,200 characters each
is most of a context window spent before the agent has decided which one it
wants. So `search` returns short, match-centred snippets — enough to triage —
and `fetch` widens a chosen result to its surrounding context on demand. The
agent pays for detail only where it decided detail was worth having.

**Every result is bounded, and says when it was cut.** Tool output lands
directly in a context window, so an unbounded tool is a denial-of-service on the
thing calling it. Limits are clamped server-side rather than trusted from the
arguments, because the caller asking for 10,000 results is exactly the case the
cap exists for. When output is truncated the response says so, so the agent can
narrow its query instead of assuming it saw everything.

**Results carry stable identifiers and real locations.** Each hit reports its
`chunk_id`, source path and character offsets, so the agent can both fetch more
and cite precisely where something came from.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from corpus_mcp.corpus import PathOutsideRoot, resolve_within_root
from corpus_mcp.index import CorpusIndex
from corpus_mcp.snippets import DEFAULT_SNIPPET_CHARS, extract

# Server-side ceilings. The tool schemas advertise the defaults; these are the
# hard limits applied regardless of what the caller asks for.
MAX_RESULTS = 25
MAX_SNIPPET_CHARS = 2_000
MAX_CONTEXT_CHUNKS = 5
MAX_FETCH_CHARS = 20_000


@dataclass
class ServerConfig:
    root: pathlib.Path
    name: str = "corpus"
    chunk_chars: int = 1_200
    chunk_overlap: int = 200


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def build_server(config: ServerConfig) -> MCPServer:
    """Wire the index to an MCP server. Returns it unstarted, so tests can drive
    it in-process over the real protocol rather than calling functions."""
    index = CorpusIndex(
        config.root,
        chunk_chars=config.chunk_chars,
        chunk_overlap=config.chunk_overlap,
    )
    index.refresh(force=True)

    server = MCPServer(
        name=config.name,
        version="0.1.0",
        instructions=(
            "Retrieval over a local document corpus. Call `search` first to find "
            "relevant passages, then `fetch` with a returned chunk_id to read the "
            "surrounding context. Cite results by their source path."
        ),
    )

    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)

    @server.tool(
        description=(
            "Search the corpus for passages matching a query. Returns ranked "
            "snippets with a chunk_id for each; pass that id to `fetch` to read "
            "more. Prefer specific content words over full sentences."
        ),
        annotations=read_only,
    )
    def search(
        query: str,
        limit: int = 5,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    ) -> dict[str, Any]:
        limit = _clamp(limit, 1, MAX_RESULTS)
        snippet_chars = _clamp(snippet_chars, 80, MAX_SNIPPET_CHARS)

        hits = index.search(query, limit)
        results = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "source": hit.chunk.doc_id,
                "score": round(hit.score, 4),
                "snippet": extract(hit.chunk.text, query, snippet_chars),
                "location": {"start": hit.chunk.start, "end": hit.chunk.end},
            }
            for hit in hits
        ]

        payload: dict[str, Any] = {"query": query, "results": results, "count": len(results)}
        if not results:
            # An empty result is a dead end unless the agent is told why. Saying
            # the corpus is non-empty distinguishes "no match" from "nothing
            # indexed", which lead to different next moves.
            payload["note"] = (
                f"No matches. The corpus holds {index.chunk_count} chunks across "
                f"{len(index.documents)} documents; try different or broader terms."
            )
        elif len(results) == limit:
            payload["note"] = f"Showing the top {limit} results; more may exist."
        return payload

    @server.tool(
        description=(
            "Read the full text of a passage found by `search`, plus the "
            "surrounding chunks of the same document for context."
        ),
        annotations=read_only,
    )
    def fetch(chunk_id: str, context_chunks: int = 1) -> dict[str, Any]:
        context_chunks = _clamp(context_chunks, 0, MAX_CONTEXT_CHUNKS)

        chunk = index.get_chunk(chunk_id)
        if chunk is None:
            # Chunk ids change when a document is re-chunked, so a stale id from
            # earlier in a long session is an expected failure, not a bug. Say
            # what to do about it.
            return {
                "error": "unknown chunk_id",
                "detail": (
                    f"No chunk {chunk_id!r}. Ids change when a document is edited; "
                    "run `search` again to get current ids."
                ),
            }

        window = index.neighbours(chunk, context_chunks)
        text = _join(window)
        truncated = len(text) > MAX_FETCH_CHARS

        return {
            "chunk_id": chunk.chunk_id,
            "source": chunk.doc_id,
            "text": text[:MAX_FETCH_CHARS],
            "truncated": truncated,
            "location": {"start": window[0].start, "end": window[-1].end},
            "context_chunks": context_chunks,
        }

    @server.tool(
        description="List the documents currently indexed, with their sizes.",
        annotations=read_only,
    )
    def list_sources(limit: int = 100) -> dict[str, Any]:
        limit = _clamp(limit, 1, 1_000)
        index.refresh()
        documents = index.documents

        return {
            "root": str(index.root),
            "documents": [
                {"source": document.doc_id, "title": document.title, "chars": len(document.text)}
                for document in documents[:limit]
            ],
            "total": len(documents),
            "truncated": len(documents) > limit,
            "stats": index.stats(),
        }

    @server.resource(
        "corpus://{path}",
        description="Full text of a document in the corpus, by relative path.",
        mime_type="text/plain",
    )
    def document_resource(path: str) -> str:
        # Resource URIs are as untrusted as tool arguments; both go through the
        # containment check before anything is read.
        try:
            resolve_within_root(index.root, path)
        except PathOutsideRoot as exc:
            raise ValueError(str(exc)) from exc

        document = index.get_document(path)
        if document is None:
            raise ValueError(f"no document {path!r} in the corpus")
        return document.text

    return server


def _join(chunks: list) -> str:
    """Concatenate neighbouring chunks without repeating their overlap.

    Chunks deliberately overlap so no passage is split across a boundary, but
    handing the agent that overlap back means it reads the same sentences twice
    and may treat the repetition as emphasis. Since chunks carry their absolute
    offsets, the overlap is removed by slicing on position rather than by
    string matching.
    """
    if not chunks:
        return ""

    pieces = [chunks[0].text]
    cursor = chunks[0].end
    for chunk in chunks[1:]:
        if chunk.start >= cursor:
            pieces.append(chunk.text)
        else:
            pieces.append(chunk.text[cursor - chunk.start :])
        cursor = max(cursor, chunk.end)
    return "".join(pieces)
