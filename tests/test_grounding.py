"""Tests for Hallucination Prevention, Source Citations, and 'I don't know' Refusal."""

import pytest
import asyncio
from services.llm import UnifiedLLMClient


@pytest.fixture
def llm_client():
    return UnifiedLLMClient(primary_provider="mock")


@pytest.mark.asyncio
async def test_grounded_answer_with_citations(llm_client):
    """Asserts that grounded context yields an answer with valid citations."""
    context_chunks = [
        {
            "filename": "Acme_Policy_2026.md",
            "chunk_index": 0,
            "text": "Acme employees may expense up to $75/day for domestic travel meals in Expensify.",
            "org_id": "org_acme_corp",
        }
    ]
    system_prompt = "Answer only using provided context. Cite sources."
    user_prompt = "How much can I expense for domestic travel meals?"

    result = await llm_client.generate_answer(
        query=user_prompt,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=context_chunks,
        org_id="org_acme_corp",
    )

    assert "75" in result.text or "expense" in result.text.lower()
    assert len(result.citations) > 0
    assert result.citations[0]["document_name"] == "Acme_Policy_2026.md"
    assert result.citations[0]["chunk_index"] == 0


@pytest.mark.asyncio
async def test_refusal_when_context_is_empty(llm_client):
    """Asserts that the system refuses to answer when no context exists, rather than hallucinating."""
    result = await llm_client.generate_answer(
        query="What is the speed of light in vacuum?",
        system_prompt="Answer only using provided context. If absent, refuse.",
        user_prompt="What is the speed of light in vacuum?",
        context_chunks=[],
        org_id="org_acme_corp",
    )

    assert "I don't have information about that" in result.text
    assert len(result.citations) == 0


@pytest.mark.asyncio
async def test_refusal_when_context_does_not_contain_answer(llm_client):
    """Asserts refusal when retrieved context is irrelevant to the question."""
    context_chunks = [
        {
            "filename": "Dental_Coverage.md",
            "chunk_index": 0,
            "text": "Dental coverage includes two annual cleanings and $1,500 ortho max.",
            "org_id": "org_acme_corp",
        }
    ]

    result = await llm_client.generate_answer(
        query="How much does a server rack cost in Zurich?",
        system_prompt="Answer only using context.",
        user_prompt="How much does a server rack cost in Zurich?",
        context_chunks=context_chunks,
        org_id="org_acme_corp",
    )

    assert "I don't have information about that" in result.text


@pytest.mark.asyncio
async def test_urdu_query_grounding(llm_client):
    """Asserts that queries in Urdu return grounded Urdu responses with citations."""
    context_chunks = [
        {
            "filename": "Acme_Policy_2026.md",
            "chunk_index": 0,
            "text": "Project Phoenix production release date is October 15, 2026.",
            "org_id": "org_acme_corp",
        }
    ]

    result = await llm_client.generate_answer(
        query="پروجیکٹ فینکس کی لانچ کی تاریخ کیا ہے؟",
        system_prompt="Answer in Urdu if asked in Urdu.",
        user_prompt="پروجیکٹ فینکس کی لانچ کی تاریخ کیا ہے؟",
        context_chunks=context_chunks,
        org_id="org_acme_corp",
    )

    assert "آپ کی تنظیم کی دستاویزات کے مطابق" in result.text or "October" in result.text
    assert len(result.citations) > 0
    assert result.citations[0]["document_name"] == "Acme_Policy_2026.md"


@pytest.mark.asyncio
async def test_roman_urdu_grounding(llm_client):
    """Asserts that queries in Roman Urdu return grounded Roman Urdu responses with citations."""
    context_chunks = [
        {
            "filename": "The 7-Stage GenAI Production Workflow.pdf",
            "chunk_index": 0,
            "text": "The 7-Stage GenAI Production Workflow starts with Problem-Solution Alignment before writing code.",
            "org_id": "org_acme_corp",
        }
    ]

    result = await llm_client.generate_answer(
        query="GenAI Project ka workflow kia hona chaiye",
        system_prompt="Answer in Roman Urdu if asked in Roman Urdu.",
        user_prompt="GenAI Project ka workflow kia hona chaiye",
        context_chunks=context_chunks,
        org_id="org_acme_corp",
    )

    assert "Aap ke documents ke mutabiq" in result.text or "Workflow" in result.text
    assert len(result.citations) > 0
    assert result.citations[0]["document_name"] == "The 7-Stage GenAI Production Workflow.pdf"


@pytest.mark.asyncio
async def test_roman_urdu_refusal(llm_client):
    """Asserts refusal in Roman Urdu when information is not in documents."""
    result = await llm_client.generate_answer(
        query="Crypto mining policy kya hai?",
        system_prompt="Answer in Roman Urdu if asked in Roman Urdu.",
        user_prompt="Crypto mining policy kya hai?",
        context_chunks=[],
        org_id="org_acme_corp",
    )

    assert "Aap ki organization ke documents mein is baray mein koi information nahi mili" in result.text

