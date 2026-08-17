"""Retrieval quality, tokenisation, and incremental reindexing."""

import time

import pytest

from corpus_mcp.index import CorpusIndex, normalize, tokenize


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "coffee.md").write_text(
        "# Coffee\n\nCooler water under-extracts and tastes sour. Boiling water "
        "scalds the grounds and pulls bitter compounds from the coffee.\n"
    )
    (tmp_path / "bread.md").write_text(
        "# Bread\n\nHydration is the weight of water divided by the weight of "
        "flour. A slack dough is open-crumbed.\n"
    )
    (tmp_path / "sailing.md").write_text(
        "# Sailing\n\nA boat cannot sail directly into the wind, so upwind "
        "progress is made by tacking.\n"
    )
    return tmp_path


# --- tokenisation ---------------------------------------------------------


@pytest.mark.parametrize(
    "word,expected",
    [
        ("tastes", "taste"),
        ("grams", "gram"),
        ("parties", "party"),
        ("batches", "batch"),
        ("brewing", "brew"),
        ("extracted", "extract"),
        # Guarded cases: these must survive intact.
        ("business", "business"),
        ("less", "less"),
        ("bed", "bed"),
        ("gas", "gas"),
        ("water", "water"),
    ],
)
def test_normalize_folds_inflections_without_mangling_short_words(word, expected):
    assert normalize(word) == expected


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The water, and the FLOUR!") == ["water", "flour"]


def test_a_query_and_its_inflected_form_tokenize_alike():
    # Indexing and querying must agree, or recall quietly drops.
    assert tokenize("taste") == tokenize("tastes")


# --- search ---------------------------------------------------------------


def test_search_ranks_the_relevant_document_first(corpus):
    index = CorpusIndex(corpus)
    hits = index.search("sour coffee", limit=3)

    assert hits
    assert hits[0].chunk.doc_id == "coffee.md"


def test_search_matches_across_inflection(corpus):
    index = CorpusIndex(corpus)
    # The document says "tastes"; the query says "taste".
    hits = index.search("taste", limit=3)
    assert [hit.chunk.doc_id for hit in hits] == ["coffee.md"]


def test_search_returns_nothing_for_absent_terms(corpus):
    assert CorpusIndex(corpus).search("xylophone quokka", limit=5) == []


def test_search_returns_nothing_for_a_stopword_only_query(corpus):
    assert CorpusIndex(corpus).search("the and of", limit=5) == []


def test_search_respects_the_limit(corpus):
    index = CorpusIndex(corpus)
    assert len(index.search("water", limit=1)) == 1
    assert index.search("water", limit=0) == []


def test_scores_descend(corpus):
    hits = CorpusIndex(corpus).search("water flour dough", limit=5)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_is_deterministic(corpus):
    first = CorpusIndex(corpus).search("water", limit=5)
    second = CorpusIndex(corpus).search("water", limit=5)
    assert [hit.chunk.chunk_id for hit in first] == [hit.chunk.chunk_id for hit in second]


def test_an_empty_corpus_searches_without_raising(tmp_path):
    index = CorpusIndex(tmp_path)
    assert index.search("anything", limit=5) == []
    assert index.stats()["documents"] == 0


# --- incremental refresh --------------------------------------------------


def test_refresh_is_a_noop_when_nothing_changed(corpus):
    index = CorpusIndex(corpus)
    assert index.refresh(force=True) is True
    assert index.refresh() is False


def test_a_new_file_is_picked_up(corpus):
    index = CorpusIndex(corpus)
    index.refresh(force=True)
    assert index.search("quokka", limit=3) == []

    (corpus / "extra.md").write_text("A quokka is a small marsupial.")

    # search() refreshes on its own; an agent should not have to ask.
    hits = index.search("quokka", limit=3)
    assert [hit.chunk.doc_id for hit in hits] == ["extra.md"]


def test_an_edited_file_is_reindexed(corpus):
    index = CorpusIndex(corpus)
    index.refresh(force=True)

    target = corpus / "sailing.md"
    # mtime has nanosecond resolution but a same-nanosecond write is possible on
    # a fast filesystem; nudge it so the fingerprint definitely differs.
    time.sleep(0.01)
    target.write_text("# Sailing\n\nAnchoring scope should be five to seven times the depth.\n")

    assert [hit.chunk.doc_id for hit in index.search("anchoring scope", limit=3)] == ["sailing.md"]
    assert index.search("tacking", limit=3) == []


def test_a_deleted_file_disappears_from_results(corpus):
    index = CorpusIndex(corpus)
    index.refresh(force=True)
    assert index.search("tacking", limit=3)

    (corpus / "sailing.md").unlink()
    assert index.search("tacking", limit=3) == []
    assert index.get_document("sailing.md") is None


# --- fetch support --------------------------------------------------------


def test_neighbours_returns_the_chunk_and_its_siblings(tmp_path):
    body = "\n\n".join(f"Section {index} " + "filler " * 60 for index in range(6))
    (tmp_path / "long.md").write_text(body)

    index = CorpusIndex(tmp_path, chunk_chars=400, chunk_overlap=50)
    index.refresh(force=True)

    chunks = index._by_document["long.md"]
    assert len(chunks) >= 3

    middle = chunks[1]
    window = index.neighbours(middle, radius=1)
    assert [chunk.ordinal for chunk in window] == [0, 1, 2]

    # Radius clamps at the document edges rather than raising.
    assert index.neighbours(chunks[0], radius=1)[0].ordinal == 0
    assert index.neighbours(chunks[0], radius=0) == [chunks[0]]


def test_get_chunk_returns_none_for_an_unknown_id(corpus):
    assert CorpusIndex(corpus).get_chunk("deadbeef:9") is None


def test_stats_report_the_index_contents(corpus):
    index = CorpusIndex(corpus)
    index.refresh(force=True)
    stats = index.stats()

    assert stats["documents"] == 3
    assert stats["chunks"] >= 3
    assert stats["terms"] > 10
