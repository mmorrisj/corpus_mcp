"""Snippet extraction.

A search result has a hard job: convince the agent whether this chunk is worth
fetching, in as few tokens as possible. Returning the first N characters of the
chunk fails at that regularly — the matching sentence is usually in the middle,
so the agent sees an unrelated preamble and either discards a good hit or
fetches everything to find out.

So the snippet is a window chosen to cover as many query-term occurrences as
possible, trimmed to word boundaries and marked with ellipses when it does not
start or end at the chunk edge. It costs a pass over the match positions and it
is the difference between a result list an agent can triage and one it cannot.
"""

from __future__ import annotations

import re

from corpus_mcp.index import normalize, tokenize

DEFAULT_SNIPPET_CHARS = 400


def find_matches(text: str, query: str) -> list[tuple[int, int]]:
    """Character spans in `text` matching any query term, in order.

    Matching is on whole tokens rather than substrings, so a query for "cat"
    does not highlight the middle of "concatenate". Candidates are normalised
    the same way query terms are, so the snippet centres on `tastes` when the
    query said `taste` -- if these two disagreed, the highlighted region would
    drift away from the text the ranker actually scored.
    """
    terms = set(tokenize(query))
    if not terms:
        return []

    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[A-Za-z0-9]+", text):
        if normalize(match.group(0).lower()) in terms:
            spans.append((match.start(), match.end()))
    return spans


def extract(text: str, query: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    """A snippet of at most `max_chars` centred on the densest match region."""
    text = text.strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    matches = find_matches(text, query)
    if not matches:
        # No lexical match to centre on -- fall back to the opening, which at
        # least reads as the start of something.
        return _trim(text, 0, max_chars, prefix=False, suffix=True)

    start = _best_window_start(matches, max_chars, len(text))
    return _trim(text, start, max_chars, prefix=start > 0, suffix=start + max_chars < len(text))


def _best_window_start(matches: list[tuple[int, int]], max_chars: int, length: int) -> int:
    """Window start covering the most matches, biased to centre the first one.

    A sweep over candidate windows anchored at each match: for each, count how
    many matches fall inside. Linear in the number of matches, which is all this
    needs to be.
    """
    best_start = 0
    best_count = -1

    for begin, _ in matches:
        # Anchor a little before the match so it does not sit flush against the
        # snippet's left edge, where it reads as truncated.
        start = max(0, min(begin - max_chars // 4, length - max_chars))
        end = start + max_chars
        count = sum(
            1 for match_start, match_end in matches if match_start >= start and match_end <= end
        )
        if count > best_count:
            best_count = count
            best_start = start

    return best_start


def _trim(text: str, start: int, max_chars: int, *, prefix: bool, suffix: bool) -> str:
    """Cut a window and pull its edges back to whitespace, adding ellipses."""
    end = min(start + max_chars, len(text))
    window = text[start:end]

    if prefix:
        space = window.find(" ")
        if 0 <= space < max_chars // 4:
            window = window[space + 1 :]
    if suffix and end < len(text):
        space = window.rfind(" ")
        if space > len(window) - max_chars // 4:
            window = window[:space]

    window = window.strip()
    return f"{'… ' if prefix else ''}{window}{' …' if suffix else ''}"
