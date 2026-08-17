"""End-to-end tests over the real MCP protocol.

These drive an actual client against an actual server in-process, so what is
exercised is the wire behaviour an MCP host will see -- tool schemas, structured
results, error shapes -- rather than the Python functions underneath. A server
whose functions are correct but whose tool surface is wrong is still broken, and
only this level catches that.
"""

import json

import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from corpus_mcp.corpus import PathOutsideRoot, resolve_within_root
from corpus_mcp.server import MAX_CONTEXT_CHUNKS, MAX_RESULTS, ServerConfig, build_server


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "coffee.md").write_text(
        "# Coffee\n\nCooler water under-extracts and tastes sour. Boiling water "
        "scalds the grounds and pulls bitter compounds from the coffee.\n"
    )
    (tmp_path / "bread.md").write_text(
        "# Bread\n\nHydration is the weight of water divided by the weight of flour.\n"
    )
    long_body = "\n\n".join(f"Part {index}. " + "content " * 60 for index in range(8))
    (tmp_path / "long.md").write_text(long_body)
    return tmp_path


@pytest.fixture
def server(corpus):
    return build_server(ServerConfig(root=corpus, chunk_chars=400, chunk_overlap=80))


def payload(result):
    """Structured content from a tool result, whichever way the SDK returns it."""
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


# --- discovery ------------------------------------------------------------


async def test_the_server_advertises_its_three_tools(server):
    async with Client(server, raise_exceptions=True) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"search", "fetch", "list_sources"}
    # Descriptions are what the model reads to decide when to call a tool, so an
    # empty one is a real defect.
    for tool in tools.values():
        assert tool.description and len(tool.description) > 30


async def test_tools_are_annotated_read_only(server):
    async with Client(server, raise_exceptions=True) as client:
        tools = (await client.list_tools()).tools

    for tool in tools:
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert tool.annotations.read_only_hint is True


async def test_the_server_ships_usage_instructions(server):
    async with Client(server, raise_exceptions=True) as client:
        assert client.instructions
        assert "search" in client.instructions.lower()


# --- search ---------------------------------------------------------------


async def test_search_returns_ranked_results_with_ids_and_sources(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("search", {"query": "sour coffee", "limit": 3}))

    assert body["count"] >= 1
    top = body["results"][0]
    assert top["source"] == "coffee.md"
    assert top["chunk_id"]
    assert "sour" in top["snippet"]
    assert top["location"]["end"] > top["location"]["start"]


async def test_search_reports_an_empty_result_usefully(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("search", {"query": "quokka xylophone"}))

    assert body["results"] == []
    # An agent that gets nothing back needs to know whether the corpus is empty
    # or its query simply missed.
    assert "chunks" in body["note"]


async def test_search_limit_is_clamped_server_side(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("search", {"query": "content", "limit": 10_000}))

    assert len(body["results"]) <= MAX_RESULTS


async def test_a_zero_or_negative_limit_still_returns_something(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("search", {"query": "content", "limit": 0}))

    assert len(body["results"]) >= 1


async def test_snippet_length_is_bounded_by_the_argument(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("search", {"query": "content", "snippet_chars": 100}))

    for result in body["results"]:
        assert len(result["snippet"]) <= 110  # plus ellipses


async def test_search_warns_when_results_are_capped(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("search", {"query": "content", "limit": 1}))

    assert "more may exist" in body["note"]


# --- fetch ----------------------------------------------------------------


async def test_fetch_widens_a_search_hit_to_its_context(server):
    async with Client(server, raise_exceptions=True) as client:
        found = payload(await client.call_tool("search", {"query": "content", "limit": 1}))
        chunk_id = found["results"][0]["chunk_id"]

        narrow = payload(
            await client.call_tool("fetch", {"chunk_id": chunk_id, "context_chunks": 0})
        )
        wide = payload(await client.call_tool("fetch", {"chunk_id": chunk_id, "context_chunks": 2}))

    assert narrow["source"] == wide["source"]
    assert len(wide["text"]) > len(narrow["text"])
    assert wide["truncated"] is False


async def test_fetched_context_does_not_repeat_the_chunk_overlap(server):
    # Chunks overlap by design; handing that overlap back would make the agent
    # read the same sentences twice.
    async with Client(server, raise_exceptions=True) as client:
        found = payload(await client.call_tool("search", {"query": "content", "limit": 5}))
        target = next(r for r in found["results"] if r["source"] == "long.md")
        wide = payload(
            await client.call_tool("fetch", {"chunk_id": target["chunk_id"], "context_chunks": 2})
        )

    text = wide["text"]
    location = wide["location"]
    # The joined text must be exactly as long as the span it claims to cover.
    assert len(text) == location["end"] - location["start"]


async def test_fetch_context_is_clamped(server):
    async with Client(server, raise_exceptions=True) as client:
        found = payload(await client.call_tool("search", {"query": "content", "limit": 1}))
        body = payload(
            await client.call_tool(
                "fetch", {"chunk_id": found["results"][0]["chunk_id"], "context_chunks": 999}
            )
        )

    assert body["context_chunks"] <= MAX_CONTEXT_CHUNKS


async def test_fetch_of_an_unknown_id_explains_the_recovery(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("fetch", {"chunk_id": "deadbeef:99"}))

    # Stale ids are expected after an edit, so this is guidance, not a crash.
    assert body["error"] == "unknown chunk_id"
    assert "search" in body["detail"]


# --- list_sources ---------------------------------------------------------


async def test_list_sources_reports_the_corpus(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("list_sources", {}))

    sources = {document["source"] for document in body["documents"]}
    assert {"coffee.md", "bread.md", "long.md"} <= sources
    assert body["total"] == 3
    assert body["truncated"] is False
    assert body["stats"]["chunks"] >= 3


async def test_list_sources_marks_truncation(server):
    async with Client(server, raise_exceptions=True) as client:
        body = payload(await client.call_tool("list_sources", {"limit": 1}))

    assert len(body["documents"]) == 1
    assert body["truncated"] is True


# --- resources ------------------------------------------------------------


async def test_a_document_is_readable_as_a_resource(server):
    async with Client(server, raise_exceptions=True) as client:
        result = await client.read_resource("corpus://coffee.md")

    assert "tastes sour" in result.contents[0].text


@pytest.mark.parametrize(
    "uri",
    [
        "corpus://../../etc/passwd",
        "corpus://../outside.md",
        "corpus://etc/passwd",
    ],
)
async def test_a_traversing_or_absent_resource_uri_is_refused(server, uri):
    # The read must fail rather than return anything from outside the corpus.
    # The error is caught inside the client context on purpose: letting it
    # escape the `async with` gets it wrapped in an ExceptionGroup by anyio.
    async with Client(server) as client:
        with pytest.raises(MCPError):
            await client.read_resource(uri)


async def test_the_containment_check_runs_before_any_read(corpus, tmp_path):
    # Belt and braces at the unit level: even a path that would resolve to a
    # real file outside the root is rejected by the resolver the handler uses.
    secret = tmp_path.parent / "outside_secret.md"
    secret.write_text("secret")

    with pytest.raises(PathOutsideRoot):
        resolve_within_root(corpus, f"../{secret.name}")


# --- live corpus ----------------------------------------------------------


async def test_a_file_added_after_start_becomes_searchable(server, corpus):
    async with Client(server, raise_exceptions=True) as client:
        before = payload(await client.call_tool("search", {"query": "quokka"}))
        assert before["results"] == []

        (corpus / "new.md").write_text("A quokka is a small marsupial from Australia.")

        after = payload(await client.call_tool("search", {"query": "quokka"}))

    assert after["results"][0]["source"] == "new.md"
