"""WEEK 1 RISKiEST FIRST GATE: Cross-Tenant Vector Retrieval Isolation Test Suite.

Proves mathematically and empirically that vector queries executed for Org A
NEVER return chunks belonging to Org B, even when documents share identical topics.
"""

import pytest
import os
from services.chunking import RecursiveChunker
from app.retrieval.service import VectorRetrievalService


@pytest.fixture(scope="module")
def vector_service(tmp_path_factory):
    temp_dir = str(tmp_path_factory.mktemp("chroma_test_isolation"))
    service = VectorRetrievalService(persist_directory=temp_dir)
    service.clear_all()

    # Ingest Org A document
    chunker = RecursiveChunker()
    org_a_text = (
        "Project Phoenix is Acme Corporation's quantum accelerator. "
        "Release date: October 15, 2026. "
        "Budget: $12,500,000 USD. "
        "Lead engineer: Dr. Elena Rostova."
    )
    a_chunks = chunker.chunk_document(org_a_text, filename="acme_phoenix.txt")
    service.add_document_chunks(
        org_id="org_acme_corp",
        document_id="doc_a_001",
        filename="acme_phoenix.txt",
        chunks=a_chunks,
    )

    # Ingest Org B document with identical topic name
    org_b_text = (
        "Project Phoenix is Cyberdyne Systems' neural defense network. "
        "Release date: January 30, 2029. "
        "Budget: $840,000,000 USD. "
        "Defense director: Miles Dyson. "
        "Location: Sector 7 Cybernetics Lab."
    )
    b_chunks = chunker.chunk_document(org_b_text, filename="cyberdyne_phoenix.txt")
    service.add_document_chunks(
        org_id="org_cyberdyne",
        document_id="doc_b_001",
        filename="cyberdyne_phoenix.txt",
        chunks=b_chunks,
    )

    return service


def test_zero_leakage_on_identical_query(vector_service):
    """Asserts that identical query string returns strictly isolated tenant chunks."""
    query = "What is the release date and budget for Project Phoenix?"

    # Query as Org A
    org_a_results = vector_service.query_similar_chunks("org_acme_corp", query, top_k=5)
    assert len(org_a_results) > 0, "Org A should retrieve its own chunks."

    for chunk in org_a_results:
        assert chunk["org_id"] == "org_acme_corp", f"Cross-tenant leak! Found org_id: {chunk['org_id']}"
        assert "Cyberdyne" not in chunk["text"], "Org B company name leaked into Org A context!"
        assert "$840,000,000" not in chunk["text"], "Org B confidential budget leaked into Org A context!"
        assert "Miles Dyson" not in chunk["text"], "Org B personnel leaked into Org A context!"

    # Query as Org B
    org_b_results = vector_service.query_similar_chunks("org_cyberdyne", query, top_k=5)
    assert len(org_b_results) > 0, "Org B should retrieve its own chunks."

    for chunk in org_b_results:
        assert chunk["org_id"] == "org_cyberdyne", f"Cross-tenant leak! Found org_id: {chunk['org_id']}"
        assert "Acme Corporation" not in chunk["text"], "Org A company name leaked into Org B context!"
        assert "$12,500,000" not in chunk["text"], "Org A confidential budget leaked into Org B context!"
        assert "Dr. Elena Rostova" not in chunk["text"], "Org A personnel leaked into Org B context!"


def test_tenant_cannot_retrieve_unowned_topics(vector_service):
    """Asserts Org A cannot retrieve information that only exists in Org B documents."""
    query = "Where is the Sector 7 Cybernetics Lab located?"

    org_a_results = vector_service.query_similar_chunks("org_acme_corp", query, top_k=5)
    # Check that no Org B chunk was returned
    for chunk in org_a_results:
        assert chunk["org_id"] == "org_acme_corp"
        assert "Sector 7" not in chunk["text"]

    org_b_results = vector_service.query_similar_chunks("org_cyberdyne", query, top_k=5)
    assert any("Sector 7" in c["text"] for c in org_b_results), "Org B should retrieve its own Sector 7 details."


def test_document_deletion_purges_vectors(vector_service):
    """Asserts deleting a document purges all its vectors from ChromaDB for that tenant."""
    vector_service.delete_document_chunks("doc_a_001", "org_acme_corp")
    remaining = vector_service.query_similar_chunks("org_acme_corp", "Project Phoenix", top_k=5)
    assert len(remaining) == 0, "Chunks for deleted doc_a_001 should no longer be retrievable."

    # Verify Org B's doc is still untouched
    org_b_results = vector_service.query_similar_chunks("org_cyberdyne", "Project Phoenix", top_k=5)
    assert len(org_b_results) > 0, "Org B chunks should remain intact."
