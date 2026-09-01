"""Standalone CLI runner for the Golden Evaluation Suite."""

import os
import sys
import json
import time
import asyncio

# Ensure utf-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.session import engine, Base, SessionLocal
from app.db.models import Organization, Document, Chunk
from app.retrieval.service import get_vector_service
from services.llm import UnifiedLLMClient
from app.main import seed_default_tenants_and_docs
from app.config import settings


def load_prompt_file(filename: str) -> str:
    path = os.path.join(settings.PROMPTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


async def run_evaluation():
    print("=" * 80)
    print("MULTI-TENANT RAG KNOWLEDGE BASE - PRODUCTION GOLDEN EVALUATION")
    print("=" * 80)

    # Initialize DB & Clean Vector DB to ensure 100% clean benchmark state
    Base.metadata.create_all(bind=engine)
    vector_service = get_vector_service()
    vector_service.clear_all()
    seed_default_tenants_and_docs(force=True)

    eval_file = os.path.join(os.path.dirname(__file__), "golden_set.json")
    with open(eval_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    vector_service = get_vector_service()
    llm_client = UnifiedLLMClient(primary_provider=settings.PRIMARY_LLM_PROVIDER)
    system_prompt = load_prompt_file("system_v1.txt")
    qa_template = load_prompt_file("qa_v1.txt")

    total = len(cases)
    retrieval_hits = 0
    grounding_hits = 0
    isolation_passed_count = 0
    total_latency = 0.0
    total_cost = 0.0

    print(f"\nRunning {total} test cases across tenants...\n")
    print(f"{'ID':<30} | {'Tenant':<15} | {'Recall':<6} | {'Ground':<6} | {'Isol':<6} | {'Latency':<7}")
    print("-" * 80)

    for item in cases:
        t0 = time.perf_counter()
        org_id = item["org_id"]
        query = item["query"]
        expected_keywords = item.get("expected_keywords", [])
        expected_doc = item.get("expected_document")
        should_refuse = item.get("should_refuse", False)

        # 1. Vector Retrieval
        chunks = vector_service.query_similar_chunks(org_id=org_id, query=query, top_k=3)

        # Tenant isolation check
        has_cross_tenant_leak = any(c.get("org_id") != org_id for c in chunks)
        if not has_cross_tenant_leak:
            isolation_passed_count += 1

        # Retrieval recall check
        retrieval_matched = False
        if not should_refuse:
            if any(expected_doc.lower() in c.get("filename", "").lower() for c in chunks):
                retrieval_matched = True
                retrieval_hits += 1
        else:
            retrieval_matched = True
            retrieval_hits += 1

        # 2. Context assembly & LLM generation
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

        # Grounding check
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

        recall_str = "PASS" if retrieval_matched else "FAIL"
        ground_str = "PASS" if grounding_matched else "FAIL"
        isol_str = "PASS" if not has_cross_tenant_leak else "LEAK"
        print(f"{item['id']:<30} | {org_id:<15} | {recall_str:<6} | {ground_str:<6} | {isol_str:<6} | {latency:.0f}ms")

    recall_pct = (retrieval_hits / total) * 100
    grounding_pct = (grounding_hits / total) * 100
    isolation_pct = (isolation_passed_count / total) * 100
    avg_latency = total_latency / total

    print("=" * 80)
    print("GOLDEN EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases:            {total}")
    print(f"Cross-Tenant Isolation:       {isolation_pct:.1f}% (Pass Gate: 100.0%) -> {'PASS [OK]' if isolation_pct == 100 else 'FAIL [X]'}")
    print(f"Retrieval Recall @ 3:         {recall_pct:.1f}% (Pass Gate: >= 90.0%) -> {'PASS [OK]' if recall_pct >= 90 else 'FAIL [X]'}")
    print(f"Grounding & Refusal Accuracy: {grounding_pct:.1f}% (Pass Gate: >= 85.0%) -> {'PASS [OK]' if grounding_pct >= 85 else 'FAIL [X]'}")
    print(f"Average Latency:             {avg_latency:.1f}ms")
    print(f"Estimated Total API Cost:    ${total_cost:.6f}")
    print("=" * 80)

    if isolation_pct == 100 and recall_pct >= 90 and grounding_pct >= 85:
        print("ALL PRODUCTION QUALITY & SECURITY GATES PASSED!")
        return 0
    else:
        print("SOME PRODUCTION GATES FAILED.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_evaluation())
    sys.exit(exit_code)
