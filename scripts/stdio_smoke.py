#!/usr/bin/env python3
"""Launch the installed server as a subprocess and exercise it over stdio.

The in-process tests in `tests/test_server.py` cover protocol behaviour, but
they import the server directly. This checks the things they cannot: that the
package installs, that the `corpus-mcp` console script exists and is on PATH,
that it speaks JSON-RPC over stdio the way an MCP host will launch it, and that
nothing pollutes stdout and corrupts the stream.

Usage: python scripts/stdio_smoke.py [corpus-directory]
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

EXPECTED_TOOLS = {"search", "fetch", "list_sources"}


def console_script() -> str:
    """Locate the installed `corpus-mcp` entry point.

    CI installs into the job's Python, where it lands on PATH. A local venv puts
    it beside the interpreter but PATH only sees it once the venv is activated,
    which `make` does not do. Checking next to `sys.executable` first makes this
    work either way -- and if neither finds it, the console script is genuinely
    missing, which is exactly the packaging failure this test exists to catch.
    """
    beside_interpreter = pathlib.Path(sys.executable).parent / "corpus-mcp"
    if beside_interpreter.exists():
        return str(beside_interpreter)

    found = shutil.which("corpus-mcp")
    if found:
        return found

    raise SystemExit(
        "corpus-mcp console script not found; is the package installed (`pip install -e .`)?"
    )


def result_payload(result) -> dict:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    return json.loads(result.content[0].text)


async def main(root: pathlib.Path) -> int:
    parameters = StdioServerParameters(
        command=console_script(),
        args=["--root", str(root), "serve"],
    )

    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            initialised = await session.initialize()
            print(f"connected to {initialised.server_info.name} {initialised.server_info.version}")

            tools = {tool.name for tool in (await session.list_tools()).tools}
            if tools != EXPECTED_TOOLS:
                print(f"unexpected tools: {sorted(tools)}", file=sys.stderr)
                return 1
            print(f"tools: {sorted(tools)}")

            listing = result_payload(await session.call_tool("list_sources", {}))
            if listing["total"] < 1:
                print("no documents indexed", file=sys.stderr)
                return 1
            print(f"indexed {listing['total']} documents, {listing['stats']['chunks']} chunks")

            found = result_payload(
                await session.call_tool("search", {"query": "sour coffee", "limit": 2})
            )
            if not found["results"]:
                print("search returned nothing", file=sys.stderr)
                return 1
            top = found["results"][0]
            print(f"top hit: {top['source']} (score {top['score']})")

            fetched = result_payload(
                await session.call_tool("fetch", {"chunk_id": top["chunk_id"]})
            )
            if not fetched.get("text"):
                print("fetch returned no text", file=sys.stderr)
                return 1
            print(f"fetched {len(fetched['text'])} characters from {fetched['source']}")

    print("stdio smoke test passed")
    return 0


if __name__ == "__main__":
    corpus_root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "examples/corpus")
    if not corpus_root.is_dir():
        print(f"corpus root {corpus_root} is not a directory", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(corpus_root)))
