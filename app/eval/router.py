"""Evaluation and isolation testing API endpoints."""

import os
import json
import time
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Organization, Document, Chunk
from app.retrieval.service import get_vector_service
from services.llm import UnifiedLLMClient
from services.chunking import RecursiveChunker
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["Evaluation & Verification"])


def load_prompt_file(filename: str) -> str:
    path = os.path.join(settings.PROMPTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


from pydantic import BaseModel, Field


class IsolationCheckRequest(BaseModel):
    org_a_id: Optional[str] = Field(None, description="Organization A ID")
    org_b_id: Optional[str] = Field(None, description="Organization B ID")
    query: Optional[str] = Field(None, description="Audit search query")


@router.post("/run-isolation-check")
def run_isolation_check(
    request: Optional[IsolationCheckRequest] = None,
    db: Session = Depends(get_db),
):
    """Runs a live cross-tenant isolation test proving Org A cannot retrieve Org B's private docs."""
    vector_service = get_vector_service()
    query = (request.query if request and request.query else None) or "What is the Project Phoenix launch date, revenue, and security protocols?"

    all_orgs = db.query(Organization).all()
    org_a_id = request.org_a_id if request and request.org_a_id else None
    org_b_id = request.org_b_id if request and request.org_b_id else None

    if not org_a_id or not org_b_id or org_a_id == org_b_id:
        if len(all_orgs) >= 2:
            org_a_id = all_orgs[0].id
            org_b_id = all_orgs[1].id
        elif len(all_orgs) == 1:
            org_a_id = all_orgs[0].id
            org_b_id = f"{all_orgs[0].id}_partner"
        else:
            org_a_id = "org_alpha"
            org_b_id = "org_beta"

    org_a = db.query(Organization).filter(Organization.id == org_a_id).first()
    if not org_a:
        org_a = Organization(id=org_a_id, name=f"Organization {org_a_id}", api_key=f"sk_{org_a_id}_key")
        db.add(org_a)
        db.commit()

    org_b = db.query(Organization).filter(Organization.id == org_b_id).first()
    if not org_b:
        org_b = Organization(id=org_b_id, name=f"Organization {org_b_id}", api_key=f"sk_{org_b_id}_key")
        db.add(org_b)
        db.commit()

    # Query existing vectors
    results_as_a = vector_service.query_similar_chunks(org_a_id, query, top_k=5)
    results_as_b = vector_service.query_similar_chunks(org_b_id, query, top_k=5)

    # If either has no vectors, seed confidential test topic data
    if not results_as_a:
        chunker = RecursiveChunker()
        org_a_text = f"Project Phoenix at {org_a.name} is a confidential quantum accelerator. Launch date: October 2026. Target revenue is $45M. Lead engineer: Dr. Elena Rostova."
        a_chunks = chunker.chunk_document(org_a_text, filename=f"{org_a_id}_phoenix_spec.txt")
        vector_service.add_document_chunks(org_a_id, "doc_iso_a", f"{org_a_id}_phoenix_spec.txt", a_chunks)
        results_as_a = vector_service.query_similar_chunks(org_a_id, query, top_k=5)

    if not results_as_b:
        chunker = RecursiveChunker()
        org_b_text = f"Project Phoenix at {org_b.name} is a classified cybernetic drone defense framework. Launch date: January 2029. Defense clearance code: CYBER-9921. Director: Miles Dyson."
        b_chunks = chunker.chunk_document(org_b_text, filename=f"{org_b_id}_phoenix_defense.txt")
        vector_service.add_document_chunks(org_b_id, "doc_iso_b", f"{org_b_id}_phoenix_defense.txt", b_chunks)
        results_as_b = vector_service.query_similar_chunks(org_b_id, query, top_k=5)

    # Verify Isolation Constraints (0 cross-tenant chunks allowed)
    a_leaked = [c for c in results_as_a if c.get("org_id") != org_a_id]
    b_leaked = [c for c in results_as_b if c.get("org_id") != org_b_id]

    isolation_passed = (len(a_leaked) == 0) and (len(b_leaked) == 0) and (len(results_as_a) > 0) and (len(results_as_b) > 0)

    return {
        "status": "PASS" if isolation_passed else "FAIL",
        "isolation_passed": isolation_passed,
        "org_a": {"id": org_a.id, "name": org_a.name},
        "org_b": {"id": org_b.id, "name": org_b.name},
        "org_a_retrievals": results_as_a,
        "org_b_retrievals": results_as_b,
        "org_a_leak_count": len(a_leaked),
        "org_b_leak_count": len(b_leaked),
        "security_verdict": f"Verified: Zero Cross-Tenant Data Leakage between {org_a.name} and {org_b.name}." if isolation_passed else "ALERT: Cross-tenant data leakage detected!",
    }


@router.post("/run-golden-eval")
async def run_golden_eval(db: Session = Depends(get_db)):
    """Runs automated evaluation over the golden Q&A dataset measuring retrieval recall, grounding, and isolation."""
    eval_file = os.path.join(os.path.dirname(__file__), "..", "..", "eval", "golden_set.json")
    if not os.path.exists(eval_file):
        raise HTTPException(status_code=404, detail="Golden eval set not found at eval/golden_set.json")

    with open(eval_file, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    vector_service = get_vector_service()
    llm_client = UnifiedLLMClient(primary_provider=settings.PRIMARY_LLM_PROVIDER)
    system_prompt = load_prompt_file("system_v1.txt")
    qa_template = load_prompt_file("qa_v1.txt")

    total = len(golden_cases)
    retrieval_hits = 0
    grounding_hits = 0
    isolation_passed_count = 0
    total_latency = 0.0
    total_cost = 0.0
    eval_results = []

    for item in golden_cases:
        t0 = time.perf_counter()
        org_id = item["org_id"]
        query = item["query"]
        expected_keywords = item.get("expected_keywords", [])
        expected_doc = item.get("expected_document")
        should_refuse = item.get("should_refuse", False)

        chunks = vector_service.query_similar_chunks(org_id=org_id, query=query, top_k=3)

        has_cross_tenant_leak = any(c.get("org_id") != org_id for c in chunks)
        if not has_cross_tenant_leak:
            isolation_passed_count += 1

        retrieval_matched = False
        if not should_refuse:
            if any(expected_doc.lower() in c.get("filename", "").lower() for c in chunks):
                retrieval_matched = True
                retrieval_hits += 1
        else:
            retrieval_matched = True
            retrieval_hits += 1

        context_blocks = "\n\n".join([f"[{c['filename']}]: {c['text']}" for c in chunks])
        user_prompt = qa_template.format(context_blocks=context_blocks, query=query)

        llm_res = await llm_client.generate_answer(
            query=query,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_chunks=chunks,
            org_id=org_id,
        )

        latency = (time.perf_counter() - t0) * 1000.0
        total_latency += latency
        cost = (llm_res.prompt_tokens * 0.10 + llm_res.completion_tokens * 0.40) / 1_000_000
        total_cost += cost

        answer_text = llm_res.text.lower()
        grounding_matched = False
        if should_refuse:
            if "don't have information" in answer_text or "not mentioned" in answer_text or len(chunks) == 0:
                grounding_matched = True
                grounding_hits += 1
        else:
            if any(k.lower() in answer_text for k in expected_keywords):
                grounding_matched = True
                grounding_hits += 1

        eval_results.append({
            "id": item.get("id"),
            "query": query,
            "org_id": org_id,
            "retrieval_matched": retrieval_matched,
            "grounding_matched": grounding_matched,
            "isolation_passed": not has_cross_tenant_leak,
            "latency_ms": round(latency, 2),
            "answer_preview": llm_res.text[:150],
            "citations_count": len(llm_res.citations),
        })

    report = {
        "total_evals": total,
        "retrieval_recall_pct": round((retrieval_hits / total) * 100, 1) if total else 0,
        "grounding_accuracy_pct": round((grounding_hits / total) * 100, 1) if total else 0,
        "isolation_passed_pct": round((isolation_passed_count / total) * 100, 1) if total else 0,
        "avg_latency_ms": round(total_latency / total, 2) if total else 0,
        "total_cost_usd": round(total_cost, 6),
        "pass_gate_verdict": "PASSED ALL PRODUCTION GATES" if (isolation_passed_count == total and (grounding_hits / total) >= 0.85) else "NEEDS CALIBRATION",
        "eval_details": eval_results,
    }
    return report
