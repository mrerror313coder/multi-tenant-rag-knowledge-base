"""Canned deterministic mock responses for CI testing."""

from typing import List, Dict, Any


class MockCannedLLM:
    """Provides deterministic responses for test suite without making external network calls."""

    @staticmethod
    def get_grounded_response(query: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "I don't have information about that in your organization's documents."

        # Search for query keywords in retrieved chunks
        words = [w.lower() for w in query.split() if len(w) > 3]
        for c in chunks:
            text = c.get("text", "")
            if any(w in text.lower() for w in words):
                doc_name = c.get("filename", "doc.txt")
                c_idx = c.get("chunk_index", 0)
                return f"According to your documents: {text.strip()} [Doc: {doc_name}, Chunk: {c_idx}]"

        return "I don't have information about that in your organization's documents."
