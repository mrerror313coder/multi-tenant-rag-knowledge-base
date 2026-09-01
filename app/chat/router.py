"""Chat and RAG query endpoints with streaming SSE, Multimodal Vision, and Voice STT."""

import os
import re
import logging
import httpx
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Organization, QueryLog
from app.schemas.models import QueryRequest, QueryResponse, CitationItem, TTSRequest
from app.auth.middleware import get_current_tenant
from app.retrieval.service import get_vector_service
from services.llm import UnifiedLLMClient
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat & RAG"])

llm_client = UnifiedLLMClient(
    primary_provider=settings.PRIMARY_LLM_PROVIDER,
    gemini_api_key=settings.GEMINI_API_KEY,
    groq_api_key=settings.GROQ_API_KEY,
    openai_api_key=settings.OPENAI_API_KEY,
)


def load_prompt_file(filename: str) -> str:
    """Loads versioned prompt template from disk."""
    path = os.path.join(settings.PROMPTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with semantic clustering and similarity scores for LLM prompt."""
    if not chunks:
        return "No relevant organization documents found."

    blocks = []
    current_cluster = None
    for i, c in enumerate(chunks, 1):
        doc_name = c.get("filename", "document.txt")
        c_idx = c.get("chunk_index", 0)
        page = c.get("page", 1)
        sim = c.get("similarity_score", 0.0)
        cluster_id = c.get("cluster_id")
        cluster_label = c.get("cluster_label", f"Cluster {cluster_id}")
        text = c.get("text", "").strip()

        if cluster_id and cluster_id != current_cluster:
            blocks.append(f"--- Semantic {cluster_label} ---")
            current_cluster = cluster_id

        sim_pct = f"{round(sim * 100, 1)}%" if sim > 0 else "High Match"
        blocks.append(f"[Chunk #{i} | Document: {doc_name} | Chunk: {c_idx} | Page: {page} | Similarity: {sim_pct}]\n{text}")

    return "\n\n".join(blocks)


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Answers user question strictly grounded in tenant's documents with zero cross-tenant leakage."""
    vector_service = get_vector_service()

    retrieved_chunks = vector_service.query_similar_chunks(
        org_id=tenant.id,
        query=request.query,
        top_k=request.top_k or settings.TOP_K,
        score_threshold=request.score_threshold,
    )

    system_prompt = load_prompt_file("system_v1.txt")
    qa_template = load_prompt_file("qa_v1.txt")

    context_str = build_context_block(retrieved_chunks)
    user_prompt = qa_template.format(context_blocks=context_str, query=request.query)

    llm_result = await llm_client.generate_answer(
        query=request.query,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=retrieved_chunks,
        org_id=tenant.id,
    )

    citation_items = [
        CitationItem(
            document_name=c.get("document_name", "doc"),
            chunk_index=c.get("chunk_index", 0),
            snippet=c.get("snippet", "")[:300],
            similarity_score=c.get("similarity_score") or c.get("similarity"),
            cluster_id=c.get("cluster_id"),
            cluster_label=c.get("cluster_label"),
            page_number=c.get("page_number", 1),
        )
        for c in llm_result.citations
    ]

    log_entry = QueryLog(
        org_id=tenant.id,
        query_text=request.query,
        response_text=llm_result.text,
        model_used=llm_result.model,
        retrieved_chunks_count=len(retrieved_chunks),
        latency_ms=llm_result.latency_ms,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
        estimated_cost_usd=(llm_result.prompt_tokens * 0.10 + llm_result.completion_tokens * 0.40) / 1_000_000,
        degraded=llm_result.degraded,
    )
    db.add(log_entry)
    db.commit()

    return QueryResponse(
        query=request.query,
        answer=llm_result.text,
        citations=citation_items,
        model_used=llm_result.model,
        tokens_used=llm_result.prompt_tokens + llm_result.completion_tokens,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
        latency_ms=llm_result.latency_ms,
        estimated_cost_usd=log_entry.estimated_cost_usd,
        degraded=llm_result.degraded,
        org_id=tenant.id,
    )


@router.post("/multimodal-query", response_model=QueryResponse)
async def query_multimodal_knowledge_base(
    query: str = Form(...),
    top_k: int = Form(3),
    image: Optional[UploadFile] = File(None),
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Answers user question with image/screenshot understanding grounded in tenant documents."""
    vector_service = get_vector_service()

    retrieved_chunks = vector_service.query_similar_chunks(
        org_id=tenant.id,
        query=query,
        top_k=top_k,
    )

    image_bytes = None
    image_mime = "image/png"
    if image:
        image_bytes = await image.read()
        image_mime = image.content_type or "image/png"

    system_prompt = load_prompt_file("system_v1.txt")
    qa_template = load_prompt_file("qa_v1.txt")

    context_str = build_context_block(retrieved_chunks)
    user_prompt = qa_template.format(context_blocks=context_str, query=query)

    llm_result = await llm_client.generate_answer(
        query=query,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_chunks=retrieved_chunks,
        org_id=tenant.id,
        image_bytes=image_bytes,
        image_mime_type=image_mime,
    )

    citation_items = [
        CitationItem(
            document_name=c.get("document_name", "doc"),
            chunk_index=c.get("chunk_index", 0),
            snippet=c.get("snippet", "")[:300],
            similarity_score=c.get("similarity_score") or c.get("similarity"),
            cluster_id=c.get("cluster_id"),
            cluster_label=c.get("cluster_label"),
            page_number=c.get("page_number", 1),
        )
        for c in llm_result.citations
    ]

    log_entry = QueryLog(
        org_id=tenant.id,
        query_text=f"[Multimodal/Image] {query}",
        response_text=llm_result.text,
        model_used=llm_result.model,
        retrieved_chunks_count=len(retrieved_chunks),
        latency_ms=llm_result.latency_ms,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
        estimated_cost_usd=(llm_result.prompt_tokens * 0.10 + llm_result.completion_tokens * 0.40) / 1_000_000,
        degraded=llm_result.degraded,
    )
    db.add(log_entry)
    db.commit()

    return QueryResponse(
        query=query,
        answer=llm_result.text,
        citations=citation_items,
        model_used=llm_result.model,
        tokens_used=llm_result.prompt_tokens + llm_result.completion_tokens,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
        latency_ms=llm_result.latency_ms,
        estimated_cost_usd=log_entry.estimated_cost_usd,
        degraded=llm_result.degraded,
        org_id=tenant.id,
    )


@router.post("/transcribe")
async def transcribe_audio_endpoint(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form("ur"),
    tenant: Organization = Depends(get_current_tenant),
):
    """Transcribes an uploaded voice message / audio query into text."""
    audio_bytes = await audio_file.read()
    mime_type = audio_file.content_type or "audio/webm"

    transcribed_text = await llm_client.transcribe_audio(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        language=language,
    )
    return {"transcription": transcribed_text, "language": language, "org_id": tenant.id}


@router.post("/stream")
async def stream_query_knowledge_base(
    request: QueryRequest,
    tenant: Organization = Depends(get_current_tenant),
):
    """Streams RAG grounded response token-by-token over Server-Sent Events (SSE)."""
    vector_service = get_vector_service()

    retrieved_chunks = vector_service.query_similar_chunks(
        org_id=tenant.id,
        query=request.query,
        top_k=request.top_k or settings.TOP_K,
        score_threshold=request.score_threshold,
    )

    system_prompt = load_prompt_file("system_v1.txt")
    qa_template = load_prompt_file("qa_v1.txt")
    context_str = build_context_block(retrieved_chunks)
    user_prompt = qa_template.format(context_blocks=context_str, query=request.query)

    return StreamingResponse(
        llm_client.stream_answer(
            query=request.query,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_chunks=retrieved_chunks,
            org_id=tenant.id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/logs")
def get_query_logs(
    limit: int = Query(50, ge=1, le=200),
    tenant: Organization = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Retrieves recent queries, latency, tokens, and cost logs for this organization."""
    logs = db.query(QueryLog).filter(QueryLog.org_id == tenant.id).order_by(QueryLog.created_at.desc()).limit(limit).all()
    return logs


@router.post("/tts")
async def generate_speech_endpoint(
    request: TTSRequest,
    tenant: Organization = Depends(get_current_tenant),
):
    """Synthesizes high-fidelity Pakistani Urdu (ur-PK) or English neural speech audio."""
    clean_text = re.sub(r"\[Doc:[^\]]+\]", "", request.text)
    clean_text = re.sub(r"[#*`_~>]", "", clean_text).strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Empty text")

    is_urdu = bool(re.search(r"[\u0600-\u06FF]", clean_text))
    voice = request.voice or ("ur-PK-UzmaNeural" if is_urdu else "en-US-AriaNeural")

    # 1. Microsoft Edge Neural TTS (Natural Pakistani Urdu voice)
    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text[:3000], voice=voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        if audio_data:
            return Response(content=audio_data, media_type="audio/mpeg")
    except Exception as e:
        logger.warning(f"Edge TTS failed: {e}. Trying Google TTS fallback.")

    # 2. Google Translate TTS Fallback
    try:
        tl = "ur" if is_urdu else "en"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                "https://translate.google.com/translate_tts",
                params={"ie": "UTF-8", "tl": tl, "client": "tw-ob", "q": clean_text[:400]},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code == 200 and r.content:
                return Response(content=r.content, media_type="audio/mpeg")
    except Exception as e:
        logger.warning(f"Google TTS fallback failed: {e}")

    raise HTTPException(status_code=500, detail="Failed to synthesize speech")
