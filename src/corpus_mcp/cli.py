"""Command line entry point.

`serve` is what an MCP client launches. `search` runs one query and prints the
result, which exists so the corpus can be sanity-checked without wiring the
server into a client first -- the fastest way to find out whether indexing is
working is to ask it something and look.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from corpus_mcp.index import CorpusIndex
from corpus_mcp.server import ServerConfig, build_server
from corpus_mcp.snippets import extract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corpus-mcp", description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True, help="corpus directory")
    parser.add_argument("--name", default="corpus", help="server name reported to the client")
    parser.add_argument("--chunk-chars", type=int, default=1_200)
    parser.add_argument("--chunk-overlap", type=int, default=200)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="run the MCP server over stdio")

    search_parser = subparsers.add_parser("search", help="run one query and print results")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("stats", help="print what is indexed and exit")

    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"corpus root {args.root} is not a directory", file=sys.stderr)
        return 2

    config = ServerConfig(
        root=args.root,
        name=args.name,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
    )

    if args.command == "serve":
        # Logging must go to stderr: stdout carries the JSON-RPC stream, and
        # anything else written there corrupts the protocol.
        print(f"corpus-mcp serving {args.root.resolve()}", file=sys.stderr)
        build_server(config).run("stdio")
        return 0

    index = CorpusIndex(
        config.root, chunk_chars=config.chunk_chars, chunk_overlap=config.chunk_overlap
    )
    index.refresh(force=True)

    if args.command == "stats":
        print(json.dumps({"root": str(index.root), **index.stats()}, indent=2))
        return 0

    hits = index.search(args.query, args.limit)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "source": hit.chunk.doc_id,
                        "score": round(hit.score, 4),
                        "snippet": extract(hit.chunk.text, args.query),
                    }
                    for hit in hits
                ],
                indent=2,
            )
        )
        return 0

    if not hits:
        print("no matches")
        return 0

    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. {hit.chunk.doc_id}  (score {hit.score:.3f}, {hit.chunk.chunk_id})")
        print(f"   {extract(hit.chunk.text, args.query)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
