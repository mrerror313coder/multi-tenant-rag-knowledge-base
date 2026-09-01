"""Pydantic data schemas for API requests, responses, and LLM contracts."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# --- Organization Schemas ---
class OrganizationCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Organization display name")
    org_id: Optional[str] = Field(None, min_length=2, max_length=64, description="Optional custom slug identifier")


class OrganizationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    api_key: str
    created_at: datetime
    document_count: Optional[int] = 0


# --- Document Schemas ---
class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    org_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    error_message: Optional[str] = None
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    id: str
    org_id: str
    filename: str
    status: str
    chunk_count: int
    error_message: Optional[str] = None


class ChunkDetail(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    page_number: Optional[int] = None


# --- RAG Citation & Query Schemas ---
class CitationItem(BaseModel):
    document_name: str
    chunk_index: int
    snippet: str
    similarity_score: Optional[float] = None
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    page_number: Optional[int] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question to answer")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of context chunks to retrieve")
    score_threshold: Optional[float] = Field(None, description="Optional similarity cutoff threshold")


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem] = []
    model_used: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    degraded: bool = False
    org_id: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize into speech")
    voice: Optional[str] = Field(None, description="Optional neural voice name")


# --- Evaluation Schemas ---
class GoldenEvalItem(BaseModel):
    id: str
    org_id: str
    query: str
    expected_answer_keywords: List[str]
    expected_document: Optional[str] = None
    should_refuse: bool = False


class GoldenEvalReport(BaseModel):
    total_evals: int
    retrieval_recall_pct: float
    grounding_accuracy_pct: float
    citation_precision_pct: float
    isolation_passed_pct: float
    avg_latency_ms: float
    total_cost_usd: float
    details: List[Dict[str, Any]]
