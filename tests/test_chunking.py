"""Unit tests for document chunking and boundary splitting."""

import pytest
from services.chunking import RecursiveChunker


def test_chunker_basic_split():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
    text = "Sentence one is here. Sentence two follows it. Sentence three comes next. Sentence four is last."
    chunks = chunker.split_text(text)

    assert len(chunks) >= 2
    assert all(len(c.strip()) > 0 for c in chunks)


def test_chunker_preserves_document_metadata():
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    text = "Paragraph one with some interesting knowledge.\n\nParagraph two with more details."
    doc_chunks = chunker.chunk_document(text, filename="handbook.md", extra_metadata={"author": "Antigravity"})

    assert len(doc_chunks) >= 1
    assert doc_chunks[0].chunk_index == 0
    assert doc_chunks[0].metadata["filename"] == "handbook.md"
    assert doc_chunks[0].metadata["author"] == "Antigravity"
    assert doc_chunks[0].token_count > 0


def test_chunker_handles_empty_text():
    chunker = RecursiveChunker()
    assert chunker.split_text("") == []
    assert chunker.split_text("   \n\n  ") == []
