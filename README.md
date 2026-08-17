# corpus-mcp

An [MCP](https://modelcontextprotocol.io) server that gives an agent keyword
search over a directory of documents. Point it at a folder and it works — no
model download, no API key, no GPU, no vector database running alongside it.
One dependency: the MCP SDK.

```bash
pip install -e .
corpus-mcp --root ./docs serve
```

The interesting part is not the retrieval. It is the **tool design**: what an
agent can actually do with a search tool, and what makes one usable rather than
a context-window bonfire.

---

## Try it in ten seconds

```console
$ make demo
1. reference/glossary.md  (score 1.973, f700ededcfdd:0)
   # Glossary

   **Extraction** — the process of dissolving soluble compounds out of ground
   coffee. Under-extraction tastes sour and thin; over-extraction tastes bitter …

2. guides/brewing.md  (score 1.774, 71c6f092dbcb:0)
   # Pour-over brewing
   …
```

That query was *"why does my coffee taste sour"*. The document says `tastes`,
the query said `taste`, and the glossary entry that actually answers it ranks
first. Both of those are deliberate; see below.

## The tools

| Tool | Purpose |
|---|---|
| `search(query, limit, snippet_chars)` | Ranked passages as short, match-centred snippets, each with a `chunk_id` |
| `fetch(chunk_id, context_chunks)` | Full text of one passage plus its neighbours |
| `list_sources(limit)` | What is indexed, with per-document sizes |

Documents are also exposed as MCP **resources** at `corpus://<relative-path>`.

## Design decisions worth arguing with

**Search and fetch are separate tools.** One `search` returning full chunks is
simpler to write and much worse to use: ten results at 1,200 characters each is
most of a context window spent before the agent has decided which one it wants.
So `search` returns snippets — enough to triage — and `fetch` widens a chosen
result on demand. The agent pays for detail only where it decided detail was
worth having.

**Snippets are centred on the match, not the top of the chunk.** Returning the
first N characters fails constantly, because the matching sentence is usually in
the middle: the agent sees an unrelated preamble and either discards a good hit
or fetches everything to find out. The snippet window is chosen to cover as many
query-term occurrences as possible.

**Every limit is clamped server-side.** Tool output lands directly in a context
window, so an unbounded tool is a denial-of-service on the thing calling it. A
caller asking for 10,000 results is exactly the case the cap exists for, so
limits are enforced rather than trusted. When output is truncated the response
says so, so the agent can narrow its query instead of assuming it saw everything.

**Empty results explain themselves.** A bare empty list is a dead end. The
response reports how many chunks and documents exist, which distinguishes "your
query missed" from "nothing is indexed" — two situations with different next
moves.

**Stale identifiers are an expected outcome, not an error.** Chunk ids change
when a document is edited, so an id from earlier in a long session can go bad.
`fetch` says exactly that and tells the agent to search again.

**Overlap is stripped when chunks are joined.** Chunks overlap so no passage is
split across a boundary, but handing that overlap back means the agent reads the
same sentences twice and may read the repetition as emphasis. Chunks carry
absolute offsets, so the overlap is removed by position rather than by string
matching.

**BM25, not embeddings.** For the keyword-ish queries an agent issues while
navigating a corpus it already knows something about, lexical retrieval is
strong, and it has the property that matters most in an agent loop: fast, and it
never silently costs money. Semantic search is a worthwhile addition, not a
precondition for the thing being useful.

**Light stemming, not a real stemmer.** Plurals and common verb endings are
folded so `tastes` matches `taste`. A full Porter implementation is a hundred
lines and a maintenance surface, and its long tail (`operational` → `oper`) is
as likely to hurt as help on short queries. Indexing and querying share one
tokeniser, since any divergence between them silently costs recall.

## Security

The server is pointed at a root directory and never reads outside it. This
matters more than it might seem: **tool arguments come from model output**, so a
document identifier is untrusted input, and `../../.ssh/id_rsa` is a thing a
confused or adversarial agent will eventually ask for.

Every path crossing the boundary goes through one containment check that
resolves symlinks *before* comparing — a symlink inside the root pointing
outside it defeats a prefix check done on the unresolved path. Absolute-looking
arguments are interpreted relative to the root rather than as real absolute
paths. Resource URIs get the same treatment as tool arguments.

Non-UTF-8 files, oversized files, and vendor directories (`.git`,
`node_modules`, …) are skipped rather than indexed as noise.

## Connecting it to a client

Claude Desktop, or any MCP host, launches the server as a subprocess:

```json
{
  "mcpServers": {
    "my-docs": {
      "command": "corpus-mcp",
      "args": ["--root", "/absolute/path/to/docs", "serve"]
    }
  }
}
```

The corpus is re-read when it changes on disk, so files edited during a session
become searchable without a restart — reindexing is incremental on modification
time rather than rebuilding on every call.

## Development

```bash
make install   # server plus dev tools
make demo      # one query against the example corpus
make test      # 89 tests, no network required
make smoke     # launch the installed server as a subprocess and exercise it
make lint
```

Two layers of testing, because they catch different failures:

- **`tests/test_server.py` drives a real MCP client against a real server
  in-process.** What is exercised is wire behaviour — tool schemas, structured
  results, error shapes — not the Python functions underneath. A server whose
  functions are correct but whose tool surface is wrong is still broken, and
  only this level catches that.
- **`scripts/stdio_smoke.py` launches the installed console script as a
  subprocess** and talks JSON-RPC to it over stdio, the way a host does. That
  covers packaging, the entry point, and the transport — including the classic
  failure where something writes to stdout and corrupts the protocol stream.

## Limitations

- **Lexical retrieval only.** A query sharing no vocabulary with the document
  will not find it. Adding an embedding backend behind the same tool surface is
  the obvious next step.
- **Text formats only** — `.md`, `.txt`, `.rst`, `.csv`, `.json`, `.yaml` and
  friends. No PDF or DOCX extraction.
- **The whole index lives in memory** and is rebuilt in full when the corpus
  changes. Fine for the thousands-of-documents case this is built for; a corpus
  in the millions wants a real index that updates per file.
- **English only.** The stopword list and the suffix folding both assume it.
- **No access control beyond the root.** Every file under the root is visible to
  anything the server is connected to.

## License

MIT. Built by [Aion Innovations](https://aionbuilt.com).
