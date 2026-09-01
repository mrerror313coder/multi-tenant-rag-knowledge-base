"""Multi-Tenant vector retrieval service using ChromaDB with strict org-scoped metadata filtering."""

import os
import re
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from services.embeddings import get_embedding_service, cluster_chunks_by_embedding, compute_cosine_similarity

logger = logging.getLogger(__name__)


class VectorRetrievalService:
    """Manages vector storage and org-isolated vector similarity search."""

    COLLECTION_NAME = "multi_tenant_knowledge_base"

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_service = get_embedding_service(settings.EMBEDDING_MODEL_NAME)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document_chunks(
        self,
        org_id: str,
        document_id: str,
        filename: str,
        chunks: List[Any],
    ) -> int:
        """Embeds and persists document chunks scoped by tenant org_id."""
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for c in chunks:
            chunk_id = f"{document_id}_{c.chunk_index}"
            ids.append(chunk_id)
            documents.append(c.text)
            metadatas.append({
                "org_id": str(org_id),  # STRICT TENANT ISOLATION KEY
                "document_id": str(document_id),
                "filename": str(filename),
                "chunk_index": int(c.chunk_index),
                "page": int(c.page_number or 1),
            })

        # Generate embeddings
        embeddings = self.embedding_service.embed_documents(documents)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Persisted {len(ids)} chunks for Org '{org_id}' doc '{filename}'.")
        return len(ids)

    @staticmethod
    def _expand_cross_lingual_query(query: str) -> str:
        """Expands Urdu terms into English concepts to maximize cross-lingual embedding recall on English documents."""
        if not re.search(r"[\u0600-\u06FF]", query):
            return query

        concept_map = {
            "سٹیج": "Stage",
            "اسٹیج": "Stage",
            "ایک": "1 one first requirements",
            "دو": "2 two second scoping",
            "تین": "3 three third planning",
            "چار": "4 four fourth data",
            "پانچ": "5 five fifth project setup",
            "چھ": "6 six sixth evaluation",
            "سات": "7 seven seventh production deploy workflow",
            "فرق": "difference compare contrast",
            "پروجیکٹ": "project",
            "پروگرام": "program",
            "بجٹ": "budget cost finance",
            "پالیسی": "policy handbook",
            "مرحلہ": "stage phase step",
            "مراحل": "stages phases steps",
            "اقدامات": "steps workflow",
            "خلاصہ": "summary overview brief",
            "وضاحت": "explanation detail explain",
        }

        matched_terms = []
        for urdu_k, eng_v in concept_map.items():
            if urdu_k in query:
                matched_terms.append(eng_v)

        if matched_terms:
            return f"{query} {' '.join(matched_terms)}"
        return query

    def query_similar_chunks(
        self,
        org_id: str,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Queries the vector store with HARD metadata filtering to guarantee ZERO cross-tenant leakage."""
        if not query or not query.strip():
            return []

        search_query = self._expand_cross_lingual_query(query)
        query_vector = self.embedding_service.embed_query(search_query)
        threshold = score_threshold if score_threshold is not None else settings.SCORE_THRESHOLD

        # Query with strict tenant filter
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k * 2, 20),  # Fetch candidate pool
            where={"org_id": str(org_id)},  # MANDATORY SECURITY BOUNDARY
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc_text, meta, dist in zip(docs, metas, dists):
                # Cosine distance to similarity: similarity = 1.0 - (distance / 2) or 1.0 - distance
                similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                
                # Check tenant isolation invariant defensively
                if meta.get("org_id") != org_id:
                    logger.critical(f"FATAL: Cross-tenant data leakage detected in ChromaDB! Expected: {org_id}, got: {meta.get('org_id')}")
                    continue

                if similarity >= threshold or len(retrieved) < 1:
                    retrieved.append({
                        "text": doc_text,
                        "filename": meta.get("filename", "unknown"),
                        "document_id": meta.get("document_id"),
                        "chunk_index": meta.get("chunk_index", 0),
                        "page": meta.get("page", 1),
                        "org_id": meta.get("org_id"),
                        "similarity_score": round(similarity, 4),
                    })

        # Apply Semantic Clustering & Cluster Diversity Ordering
        clustered_chunks = cluster_chunks_by_embedding(retrieved, query_vector=query_vector)
        return clustered_chunks[:top_k]

    def delete_document_chunks(self, document_id: str, org_id: str) -> bool:
        """Safely removes all vector chunks for a specific document and tenant."""
        try:
            self._collection.delete(
                where={"$and": [{"document_id": str(document_id)}, {"org_id": str(org_id)}]}
            )
            return True
        except Exception as e:
            logger.warning(f"Error deleting chunks for doc {document_id}: {e}")
            return False

    def delete_tenant_vectors(self, org_id: str) -> bool:
        """Purges all vector embeddings belonging to a tenant organization."""
        try:
            self._collection.delete(where={"org_id": str(org_id)})
            logger.info(f"Purged all vector embeddings for tenant '{org_id}'.")
            return True
        except Exception as e:
            logger.warning(f"Error purging tenant vectors for {org_id}: {e}")
            return False

    def clear_all(self):
        """Resets the collection (used for test isolation)."""
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.warning(f"Chroma reset warning: {e}")


# Singleton instance
_vector_service = None


def get_vector_service() -> VectorRetrievalService:
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorRetrievalService()
    return _vector_service
