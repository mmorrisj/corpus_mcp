"""Snippet windowing.

The property under test throughout: the snippet must contain the text that
caused the hit. A snippet showing an unrelated preamble is what makes an agent
fetch everything.
"""

from corpus_mcp.snippets import extract, find_matches

# A document whose only relevant sentence sits a long way from the start.
LONG = (
    "Introduction. "
    + ("filler sentence about nothing in particular. " * 30)
    + "The anchor should be set with five to seven times the depth in rope. "
    + ("more irrelevant filler follows here. " * 30)
)


def test_short_text_is_returned_whole():
    assert extract("just a little text", "little", 400) == "just a little text"


def test_the_window_centres_on_a_late_match():
    snippet = extract(LONG, "anchor rope depth", 300)

    # This is the whole point: the matching sentence has to be in there.
    assert "anchor" in snippet
    assert "five to seven times the depth" in snippet
    assert len(snippet) <= 300 + 4  # allow for the ellipsis characters


def test_a_truncated_window_is_marked_at_both_ends():
    snippet = extract(LONG, "anchor", 200)
    assert snippet.startswith("… ")
    assert snippet.endswith(" …")


def test_a_window_at_the_start_has_no_leading_ellipsis():
    text = "Anchor scope matters. " + ("filler " * 200)
    snippet = extract(text, "anchor", 200)
    assert not snippet.startswith("…")
    assert snippet.endswith("…")


def test_no_match_falls_back_to_the_opening():
    snippet = extract(LONG, "quokka", 120)
    assert snippet.startswith("Introduction")
    assert not snippet.startswith("…")


def test_the_densest_region_wins_over_a_lone_earlier_match():
    text = (
        "anchor mentioned once here. "
        + ("filler " * 100)
        + "anchor rope depth anchor rope depth all together. "
        + ("filler " * 100)
    )
    snippet = extract(text, "anchor rope depth", 200)
    assert "all together" in snippet


def test_max_chars_is_respected_and_zero_yields_nothing():
    assert extract(LONG, "anchor", 0) == ""
    assert len(extract(LONG, "anchor", 50)) <= 54


def test_find_matches_is_token_aligned_not_substring():
    spans = find_matches("concatenate the cat", "cat")
    # Only the standalone "cat" matches, not the one inside "concatenate".
    assert len(spans) == 1
    start, end = spans[0]
    assert "concatenate the cat"[start:end] == "cat"


def test_find_matches_folds_inflections():
    spans = find_matches("it tastes sour", "taste")
    assert len(spans) == 1
    start, end = spans[0]
    assert "it tastes sour"[start:end] == "tastes"


def test_find_matches_ignores_stopword_only_queries():
    assert find_matches("the quick brown fox", "the and of") == []
