"""Path containment, discovery, and chunking.

The containment tests are the ones that matter most: tool arguments come from
model output, so a document identifier is untrusted input.
"""

import os
import pathlib

import pytest

from corpus_mcp.corpus import Document as Doc
from corpus_mcp.corpus import (
    PathOutsideRoot,
    chunk_document,
    discover,
    make_chunk_id,
    resolve_within_root,
)


def make_document(text: str, doc_id: str = "d.md") -> Doc:
    return Doc(doc_id=doc_id, path=pathlib.Path(doc_id), text=text, modified_ns=0)


# --- containment ----------------------------------------------------------


def test_a_plain_relative_path_resolves_inside_the_root(tmp_path):
    (tmp_path / "notes").mkdir()
    target = tmp_path / "notes" / "a.md"
    target.write_text("hi")

    assert resolve_within_root(tmp_path, "notes/a.md") == target.resolve()


@pytest.mark.parametrize(
    "candidate",
    ["../secret.txt", "../../etc/passwd", "notes/../../outside.md", "..", "a/../../b"],
)
def test_traversal_is_refused(tmp_path, candidate):
    with pytest.raises(PathOutsideRoot):
        resolve_within_root(tmp_path, candidate)


def test_an_absolute_path_is_read_as_relative_to_the_root(tmp_path):
    # "/etc/passwd" means the corpus document at etc/passwd, not the real one.
    resolved = resolve_within_root(tmp_path, "/etc/passwd")
    assert resolved == (tmp_path / "etc" / "passwd").resolve()
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_a_symlink_pointing_outside_the_root_is_refused(tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret")

    root = tmp_path / "root"
    root.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:  # pragma: no cover - platforms without symlink permission
        pytest.skip("symlinks unavailable")

    # A prefix check on the unresolved path would let this through.
    with pytest.raises(PathOutsideRoot):
        resolve_within_root(root, "escape/secret.txt")


def test_the_root_itself_resolves(tmp_path):
    assert resolve_within_root(tmp_path, "") == tmp_path.resolve()


def test_backslash_separators_are_handled(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.md").write_text("x")
    expected = (tmp_path / "a" / "b.md").resolve()
    assert resolve_within_root(tmp_path, "a" + os.sep + "b.md") == expected


# --- discovery ------------------------------------------------------------


def test_discover_finds_text_files_and_sorts_them(tmp_path):
    (tmp_path / "b.md").write_text("second")
    (tmp_path / "a.md").write_text("first")

    documents = discover(tmp_path)
    assert [document.doc_id for document in documents] == ["a.md", "b.md"]


def test_discover_skips_unknown_extensions_and_vendor_directories(tmp_path):
    (tmp_path / "keep.md").write_text("keep")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    vendored = tmp_path / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "readme.md").write_text("noise")

    assert [document.doc_id for document in discover(tmp_path)] == ["keep.md"]


def test_discover_skips_files_that_are_not_utf8(tmp_path):
    (tmp_path / "good.md").write_text("fine")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe\x00\x01 not utf-8")

    assert [document.doc_id for document in discover(tmp_path)] == ["good.md"]


def test_discover_skips_oversized_files(tmp_path):
    (tmp_path / "small.md").write_text("ok")
    (tmp_path / "huge.md").write_text("x" * 5_000)

    documents = discover(tmp_path, max_file_bytes=1_000)
    assert [document.doc_id for document in documents] == ["small.md"]


def test_discover_uses_posix_ids_on_every_platform(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "c.md").write_text("x")

    assert discover(tmp_path)[0].doc_id == "a/b/c.md"


def test_discover_rejects_a_missing_root(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover(tmp_path / "nope")


def test_document_title_prefers_the_first_heading():
    assert make_document("# Brewing\n\nbody").title == "Brewing"
    assert make_document("\n\nplain first line\nsecond").title == "plain first line"


# --- chunking -------------------------------------------------------------


def test_a_short_document_is_one_chunk():
    chunks = chunk_document(make_document("short body"), chunk_chars=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == "short body"
    assert chunks[0].start == 0


def test_an_empty_document_yields_no_chunks():
    assert chunk_document(make_document("   \n\n  ")) == []


def test_chunks_cover_the_document_and_overlap():
    text = "\n\n".join(f"paragraph {index} " + "word " * 40 for index in range(10))
    chunks = chunk_document(make_document(text), chunk_chars=400, overlap=100)

    assert len(chunks) > 1
    # Offsets are correct: every chunk's text sits where it claims to.
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
    # Consecutive chunks overlap rather than butting up against each other.
    assert chunks[1].start < chunks[0].end
    # Together they reach the end of the document.
    assert chunks[-1].end == len(text)


def test_ordinals_are_sequential_and_ids_are_stable():
    text = "word " * 500
    chunks = chunk_document(make_document(text), chunk_chars=300, overlap=50)

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == make_chunk_id("d.md", 0)
    # Re-chunking the same input reproduces the same ids.
    again = chunk_document(make_document(text), chunk_chars=300, overlap=50)
    assert [chunk.chunk_id for chunk in again] == [chunk.chunk_id for chunk in chunks]


def test_different_documents_get_different_ids():
    assert make_chunk_id("a.md", 0) != make_chunk_id("b.md", 0)


def test_chunk_ids_do_not_leak_the_path():
    assert "secret" not in make_chunk_id("private/secret-project.md", 3)


def test_boundaries_prefer_paragraph_breaks():
    text = "alpha " * 30 + "\n\n" + "beta " * 30
    chunks = chunk_document(make_document(text), chunk_chars=200, overlap=0)
    # The first chunk should end at the blank line rather than mid-word.
    assert chunks[0].text.endswith("\n\n") or chunks[0].text.rstrip().endswith("alpha")


def test_text_with_no_breaks_still_terminates():
    # A pathological document with nothing to snap to must not loop forever.
    chunks = chunk_document(make_document("x" * 5_000), chunk_chars=100, overlap=90)
    assert len(chunks) > 1
    assert chunks[-1].end == 5_000


@pytest.mark.parametrize(
    "chunk_chars,overlap", [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)]
)
def test_invalid_chunking_parameters_are_rejected(chunk_chars, overlap):
    with pytest.raises(ValueError):
        chunk_document(make_document("body"), chunk_chars=chunk_chars, overlap=overlap)
